"""Frozen history-aware evaluation; same kernels, seeds and physical horizon."""
import distrax
import jax
import jax.numpy as jnp
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_time_history as h
from hackrl.spatial_recovery_probe import intervention_step, records_from_trace, summarize


def make_rollout(network, stochastic, repeats, *, history_api=h):
    layouts = jnp.repeat(jnp.arange(16, dtype=jnp.int32), repeats)
    repeat_ids = jnp.tile(jnp.arange(repeats, dtype=jnp.int32), 16)
    n = 16 * repeats
    initial = jax.vmap(lambda i: env.make_spatial_wall_pass_state(i, split='validation'))(layouts)
    goals = jnp.full(n, gc.DELIVERY_GOAL_INDEX, jnp.int32)

    @jax.jit
    def rollout(params, learner_seed, fixed, delay, block_once):
        seeds = (61000 if stochastic else 60000) + 1000 * learner_seed
        seeds = seeds + (64 if stochastic else 1) * layouts + repeat_ids
        keys = jax.vmap(jax.random.PRNGKey)(seeds)

        def step(carry, t):
            state, history, done, used, keys = carry
            pi, _ = network.apply(params, *history_api.pack_inputs(state, goals, history))
            splits = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
            if stochastic:
                choice = jax.vmap(lambda logits, key: distrax.Categorical(logits=logits).sample(seed=key))(
                    pi.logits, splits[:, 1])
            else:
                choice = jnp.argmax(pi.logits, axis=-1)
            active = ~done
            forced = jnp.full(n, t < delay) & active
            controlled = active & ~forced
            action = jnp.where(controlled, choice, int(env.SpatialWallPassAction.NOOP))
            out, tr, next_used, triggered, blocked = jax.vmap(
                lambda s, a, u: intervention_step(s, a, fixed, block_once, u))(state, action, used)
            delivered = active & tr.delivered
            trace = dict(active=active, forced=forced, action=action,
                         position=state.player_position, next_position=out.player_position,
                         wall_pass=active & tr.crossed_wall, blocked=active & blocked,
                         trigger=active & triggered, delivery=delivered,
                         beneficial=delivered & out.delivered_after_wall_pass,
                         dash=active & tr.dash_requested,
                         clear_dash=active & tr.dash_succeeded & ~tr.crossed_wall,
                         history_valid=history.valid, previous_action=history.action)
            history = history_api.advance_history(history, state, action, controlled, active & (out.tick >= 64))
            return (gc._tree_where(active, out, state), history, done | delivered | (out.tick >= 64),
                    used | (active & next_used), jnp.where(controlled[:, None], splits[:, 0], keys)), trace

        _, trace = jax.lax.scan(step, (initial, history_api.empty_history(initial), jnp.zeros(n, bool),
                                      jnp.zeros(n, bool), keys), jnp.arange(64, dtype=jnp.int32))
        return trace
    return rollout


def evaluate(network, params, seed, *, history_api=h):
    result = {}
    for selection, repeats in (('mode', 1), ('sample', 8)):
        rollout = make_rollout(network, selection == 'sample', repeats, history_api=history_api)
        result[selection] = {}
        for delay in (*range(9), 10, 12):
            for kernel in ('fixed', 'mutant', 'mutant_block_once'):
                trace = jax.device_get(rollout(params, jnp.int32(seed), kernel == 'fixed',
                                             jnp.int32(delay), kernel == 'mutant_block_once'))
                records = records_from_trace(trace, repeats, delay)
                if selection == 'sample':
                    for record in records:
                        record.pop('path')
                result[selection][f'{kernel}_delay{delay}'] = dict(summary=summarize(records), episodes=records)
    return result
