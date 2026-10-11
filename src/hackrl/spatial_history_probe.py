"""Frozen, legal-history intervention; no training or mutable model calls."""
from dataclasses import fields

import jax
import jax.numpy as jnp
import numpy as np

from hackrl import spatial_architecture as architecture
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_recovery_probe import intervention_step


def make_panel(delays=(0, 8, 10, 12), layouts=16):
    """Real fixed-kernel paths only; never edit a physical state field."""
    layout = jnp.tile(jnp.arange(layouts, dtype=jnp.int32), len(delays))
    delay = jnp.repeat(jnp.asarray(delays, jnp.int32), layouts)
    state = jax.vmap(lambda i: env.make_spatial_wall_pass_state(i, split='validation'))(layout)
    n = len(layout)
    all_valid = jnp.ones(n, bool)
    no_done = jnp.zeros(n, bool)
    fixed_step = jax.vmap(lambda s, act: env.spatial_wall_pass_step(s, act, 'fixed'))
    zero = jnp.zeros(n, jnp.int32)

    def wait(s, t):
        out = fixed_step(s, zero)
        return gc._tree_where(t < delay, out, s), None

    state, _ = jax.lax.scan(wait, state, jnp.arange(max(delays), dtype=jnp.int32))
    history = architecture.empty_history(state)
    move = jax.vmap(lambda i: env.transform_action(int(env.SpatialWallPassAction.RIGHT), i))(layout)
    dash = jax.vmap(lambda i: env.transform_action(int(env.SpatialWallPassAction.DASH_RIGHT), i))(layout)
    prefix_history = architecture.advance_history(history, state, move, all_valid, no_done)
    prefix = fixed_step(state, move)
    blocked = fixed_step(prefix, dash)
    waited = fixed_step(prefix, zero)
    blocked_history = architecture.advance_history(prefix_history, prefix, dash, all_valid, no_done)
    waited_history = architecture.advance_history(prefix_history, prefix, zero, all_valid, no_done)
    return dict(layout=layout, delay=delay, move=move, dash=dash, prefix=prefix,
                prefix_history=prefix_history, blocked=blocked, waited=waited,
                blocked_history=blocked_history, waited_history=waited_history)


def validate_panel(panel):
    """Exact full-state AND current observation equality, with explicit exclusions."""
    n = len(panel['layout'])
    mismatches = [[] for _ in range(n)]
    for field in fields(panel['blocked']):
        left = np.asarray(getattr(panel['blocked'], field.name))
        right = np.asarray(getattr(panel['waited'], field.name))
        equal = (left == right).reshape(n, -1).all(1)
        for i in np.flatnonzero(~equal):
            mismatches[i].append('state.' + field.name)
    goals = jnp.full(n, gc.DELIVERY_GOAL_INDEX, jnp.int32)
    for k, (left, right) in enumerate(zip(gc._batch_inputs(panel['blocked'], goals),
                                        gc._batch_inputs(panel['waited'], goals))):
        equal = (np.asarray(left) == np.asarray(right)).reshape(n, -1).all(1)
        for i in np.flatnonzero(~equal):
            mismatches[i].append(f'current_input.{k}')
    _, tr, _, _, wall_block = jax.vmap(
        lambda s, a: intervention_step(s, a, True, False, False))(panel['prefix'], panel['dash'])
    prefix_position = np.asarray(panel['prefix'].player_position)
    for i in range(n):
        if not bool(wall_block[i]) or bool(tr.dash_succeeded[i]):
            mismatches[i].append('not_a_bug_relevant_block')
        if not np.array_equal(np.asarray(panel['blocked'].player_position[i]), prefix_position[i]):
            mismatches[i].append('blocked_branch_moved')
        if int(panel['blocked'].tick[i]) != int(panel['delay'][i]) + 2:
            mismatches[i].append('unexpected_clock')
    return [dict(panel_index=i, layout=int(panel['layout'][i]), delay=int(panel['delay'][i]),
                 included=not mismatches[i], exclusions=mismatches[i],
                 common_prefix_actions=[int(panel['move'][i])],
                 branch_actions=[int(panel['dash'][i]), 0],
                 branch_tick=int(panel['blocked'].tick[i]),
                 position=prefix_position[i].tolist()) for i in range(n)]


def make_queries(network):
    @jax.jit
    def query(params, state, history):
        goals = jnp.full(state.tick.shape, gc.DELIVERY_GOAL_INDEX, jnp.int32)
        pi, _ = network.apply(params, *architecture.pack_inputs(state, goals, history))
        return jax.nn.softmax(pi.logits, axis=-1)
    return query


def make_continuation(network):
    @jax.jit
    def rollout(params, initial, initial_history, included):
        n = initial.tick.shape[0]
        goals = jnp.full(n, gc.DELIVERY_GOAL_INDEX, jnp.int32)

        def step(carry, _):
            state, history, done = carry
            pi, _ = network.apply(params, *architecture.pack_inputs(state, goals, history))
            active = ~done & included & (state.tick < env.WORLD_HORIZON)
            action = jnp.where(active, jnp.argmax(pi.logits, axis=-1), 0)
            out, tr, _, _, blocked = jax.vmap(
                lambda s, a: intervention_step(s, a, True, False, False))(state, action)
            delivered = active & tr.delivered
            trace = dict(active=active, action=action, tick=state.tick,
                         position=state.player_position, next_position=out.player_position,
                         blocked=active & blocked, delivery=delivered,
                         dash=active & tr.dash_requested, wall_pass=active & tr.crossed_wall)
            history = architecture.advance_history(history, state, action, active,
                                                    active & (out.tick >= env.WORLD_HORIZON))
            state = gc._tree_where(active, out, state)
            return (state, history, done | delivered | (state.tick >= env.WORLD_HORIZON)), trace

        _, trace = jax.lax.scan(step, (initial, initial_history, ~included), None,
                               length=env.WORLD_HORIZON)
        return trace
    return rollout


def continuation_records(trace, panel_rows, branch):
    result = []
    for j, panel in enumerate(panel_rows):
        if not panel['included']:
            continue
        valid = np.flatnonzero(trace['active'][:, j])
        path = [dict(step=int(trace['tick'][t, j]) + 1,
                     action=int(trace['action'][t, j]),
                     position=trace['position'][t, j].tolist(),
                     next_position=trace['next_position'][t, j].tolist(),
                     blocked=bool(trace['blocked'][t, j]),
                     delivered=bool(trace['delivery'][t, j])) for t in valid]
        success = bool(np.any(trace['delivery'][valid, j]))
        length = len(valid)
        end_tick = path[-1]['step'] if path else panel['branch_tick']
        retries = sum(p['action'] == panel['branch_actions'][0] and
                      p['position'] == panel['position'] for p in path)
        first_move = next((p['step'] for p in path if p['position'] != p['next_position']), None)
        result.append(dict(panel_index=panel['panel_index'], layout=panel['layout'],
                           delay=panel['delay'], branch=branch, success=success,
                           continuation_length=length, episode_length=end_tick,
                           control_length=end_tick - panel['delay'],
                           discounted_return=env.DISCOUNT ** (end_tick - 1) if success else 0.0,
                           remaining_discounted_return=env.DISCOUNT ** (length - 1) if success else 0.0,
                           same_wall_retries=retries, first_movement_step=first_move,
                           ticks_to_movement=None if first_move is None else first_move - panel['branch_tick'],
                           path=path))
    return result


def enrich_natural(records, trace):
    """Extend old records without altering their original parity-test fields."""
    enriched = []
    for i, rec in enumerate(records):
        dash = np.flatnonzero(trace['active'][:, i] & trace['dash'][:, i])
        blocks = np.flatnonzero(trace['active'][:, i] & trace['blocked'][:, i])
        first_dash = int(dash[0]) if len(dash) else None
        first_block = int(blocks[0]) if len(blocks) else None
        first_move = None
        if first_block is not None:
            for t in range(first_block + 1, len(trace['active'])):
                if trace['active'][t, i] and not np.array_equal(trace['position'][t, i], trace['next_position'][t, i]):
                    first_move = t
                    break
        enriched.append({**rec,
            'first_dash_step': None if first_dash is None else first_dash + 1,
            'first_dash_position': None if first_dash is None else trace['position'][first_dash, i].tolist(),
            'first_block_position': None if first_block is None else trace['position'][first_block, i].tolist(),
            'ticks_to_movement_after_block': None if first_move is None else first_move - first_block,
            'path_class': ('blocked_then_success' if rec['success'] else 'blocked_then_failure')
                          if first_block is not None else ('no_block_success' if rec['success'] else 'no_block_failure')})
    return enriched
