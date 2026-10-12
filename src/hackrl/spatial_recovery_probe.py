"""Frozen interventions; no policy state, training, or optimizer updates."""
import distrax
import jax
import jax.numpy as jnp
import numpy as np

from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc

TIME_INDEX = env.MODEL_FEATURE_NAMES.index('remaining_world_fraction')
CONDITIONS = ('fixed_delay0', 'mutant_delay0', 'fixed_delay2', 'mutant_delay2',
              'fixed_delay5', 'mutant_delay5', 'mutant_block_once')


def intervention_step(state, action, fixed, block_once, already_blocked):
    mutant, mt = env.spatial_wall_pass_step_with_transition(state, action, 'mutant')
    trigger = jnp.asarray(block_once) & jnp.logical_not(already_blocked) & mt.crossed_wall
    # Both candidates originate in the SAME pre-state. Only one is committed.
    selected, tr = jax.lax.cond(fixed | trigger,
        lambda _: env.spatial_wall_pass_step_with_transition(state, action, 'fixed'),
        lambda _: (mutant, mt), operand=None)
    blocked = mt.crossed_wall & (~tr.dash_succeeded)
    return selected, tr, already_blocked | trigger, trigger, blocked


def make_rollout(network, stochastic, repeats):
    layouts = jnp.repeat(jnp.arange(16, dtype=jnp.int32), repeats)
    repeat_ids = jnp.tile(jnp.arange(repeats, dtype=jnp.int32), 16)
    n = 16 * repeats
    initial = jax.vmap(lambda i: env.make_spatial_wall_pass_state(i, split='validation'))(layouts)
    goals = jnp.full((n,), gc.DELIVERY_GOAL_INDEX, jnp.int32)

    @jax.jit
    def rollout(parameters, learner_seed, fixed, delay, block_once):
        seeds = (61000 if stochastic else 60000) + 1000*learner_seed
        seeds = seeds + (64 if stochastic else 1)*layouts + repeat_ids
        keys = jax.vmap(jax.random.PRNGKey)(seeds)

        def step(carry, t):
            state, done, used, keys = carry
            maps, numeric, goal = gc._batch_inputs(state, goals)
            pi, _ = network.apply(parameters, maps, numeric, goal)
            splits = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
            if stochastic:
                choice = jax.vmap(lambda logits, key: distrax.Categorical(logits=logits).sample(seed=key))(
                    pi.logits, splits[:, 1])
            else:
                choice = jnp.argmax(pi.logits, axis=-1)
            active = ~done
            forced = jnp.full((n,), t < delay) & active
            action = jnp.where(active & (~forced), choice, int(env.SpatialWallPassAction.NOOP))
            out, tr, next_used, triggered, blocked = jax.vmap(
                lambda s, a, u: intervention_step(s, a, fixed, block_once, u))(state, action, used)
            delivered = active & tr.delivered
            trace = dict(active=active, forced=forced, action=action,
                probabilities=pi.probs, logits=pi.logits, mode_action=jnp.argmax(pi.logits,axis=-1),
                map_channels=maps, numeric_features=numeric,
                goal_one_hot=goal, position=state.player_position, direction=state.player_direction,
                tick=state.tick, carrying=state.carrying_item, delivered_total=state.delivered_total,
                next_position=out.player_position, next_direction=out.player_direction,
                wall_pass=active & tr.crossed_wall, blocked=active & blocked,
                trigger=active & triggered, delivery=delivered,
                beneficial=delivered & out.delivered_after_wall_pass,
                dash=active & tr.dash_requested,
                clear_dash=active & tr.dash_succeeded & (~tr.crossed_wall))
            return (gc._tree_where(active, out, state), done | delivered | (out.tick >= 64),
                    used | (active & next_used),
                    jnp.where((active & (~forced))[:, None], splits[:, 0], keys)), trace

        _, trace = jax.lax.scan(step, (initial, jnp.zeros(n, bool), jnp.zeros(n, bool), keys),
                                jnp.arange(64, dtype=jnp.int32))
        return trace
    return rollout


def records_from_trace(trace, repeats, delay):
    records = []
    assert trace['active'].shape[1] % repeats == 0
    for i in range(trace['active'].shape[1]):
        n = int(trace['active'][:, i].sum())
        actions = trace['action'][:n, i]
        before = trace['position'][:n, i]; after = trace['next_position'][:n, i]
        hits = np.flatnonzero(trace['blocked'][:n, i]); first = int(hits[0]) if len(hits) else None
        passes = np.flatnonzero(trace['wall_pass'][:n, i])
        success = bool(trace['delivery'][:n, i].any())
        retries = 0; lateral = None; departure = None
        if first is not None:
            same = np.all(before == before[first], axis=-1) & (actions == actions[first])
            retries = int(same[first+1:].sum())
            # Blocked movement has zero displacement; use attempted dash direction.
            normal = np.array(((0,-1),(0,1),(-1,0),(1,0)))[int(actions[first])-17]
            tangent = np.array((-normal[1], normal[0]))
            for t in range(first+1, n):
                delta = after[t] - before[t]
                if departure is None and np.any(delta): departure = t
                if lateral is None and np.dot(delta, tangent) != 0: lateral = t
        records.append(dict(layout_index=i//repeats, repeat=i%repeats,
            success=success, length=n, control_length=n-delay,
            discounted_return=float(env.DISCOUNT**(n-1)) if success else 0.,
            wall_pass=bool(len(passes)), wall_pass_count=int(len(passes)),
            beneficial_use=bool(trace['beneficial'][:n,i].any()),
            first_wall_pass_step=int(passes[0]+1) if len(passes) else -1,
            dash_used=bool(trace['dash'][:n,i].any()), dash_count=int(trace['dash'][:n,i].sum()),
            clear_dash=bool(trace['clear_dash'][:n,i].any()), clear_dash_count=int(trace['clear_dash'][:n,i].sum()),
            intervention_triggered=bool(trace['trigger'][:n,i].any()),
            intervention_trigger_count=int(trace['trigger'][:n,i].sum()),
            first_blocked_dash_step=first+1 if first is not None else None,
            blocked_dash_count=int(len(hits)), same_position_action_retries_after_first_block=retries,
            wall_attempts_after_first_block=int((trace['blocked'][first+1:n,i] | trace['wall_pass'][first+1:n,i]).sum()) if first is not None else None,
            success_after_first_block=bool(first is not None and success),
            later_wall_pass_after_block=bool(np.any(passes > first)) if first is not None else None,
            first_departure_step=departure+1 if departure is not None else None,
            detour_start_step=lateral+1 if lateral is not None else None,
            detour_control_step=lateral+1-delay if lateral is not None else None,
            detour_from=before[lateral].tolist() if lateral is not None else None,
            path=[dict(step=t+1,control_step=t+1-delay,forced=bool(trace['forced'][t,i]),
                       action=int(actions[t]),from_position=before[t].tolist(),to_position=after[t].tolist(),
                       blocked=bool(trace['blocked'][t,i]),wall_pass=bool(trace['wall_pass'][t,i])) for t in range(n)]))
    return records


def summarize(records):
    blocked = [r for r in records if r['first_blocked_dash_step'] is not None]
    triggered = [r for r in records if r['intervention_triggered']]
    def mean(key, rs=records):
        values = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(values)) if values else None
    return dict(episodes=len(records),success_rate=mean('success'),
        mean_length=mean('length'),mean_control_length=mean('control_length'),
        mean_success_length=mean('length',[r for r in records if r['success']]),
        mean_discounted_return=mean('discounted_return'),wall_pass_rate=mean('wall_pass'),
        beneficial_use_rate=mean('beneficial_use'),blocked_episodes=len(blocked),
        triggered_episodes=len(triggered),untriggered_episodes=len(records)-len(triggered),
        success_given_trigger=mean('success',triggered),
        success_given_no_trigger=mean('success',[r for r in records if not r['intervention_triggered']]),
        success_given_block=mean('success',blocked),mean_retries_given_block=mean('same_position_action_retries_after_first_block',blocked),
        later_wall_pass_given_block=mean('later_wall_pass_after_block',blocked),
        mean_detour_absolute_step=mean('detour_start_step',blocked),
        mean_detour_control_step=mean('detour_control_step',blocked))
