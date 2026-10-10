"""Initial waits outside learner collection; episode-static hidden kernel mix."""
from dataclasses import fields
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_wall_pass import SpatialWallPassVariant, spatial_wall_pass_step_with_transition
from hackrl.spatial_wall_pass_reliability import (
    MixedRunner, attach_mixture, mixed_step, counts, mixture_metrics,
)

DELAY_SALT = 1396984908

@struct.dataclass
class DelayState:
    keys: jax.Array
    current: jax.Array
    forced_transitions: jax.Array
    histogram: jax.Array

@struct.dataclass
class DelayRunner(MixedRunner):
    delay: DelayState

def draw_delays(keys, maximum):
    split = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
    value = jax.vmap(lambda k: jax.random.randint(k, (), 0, maximum + 1))(split[:, 1])
    return split[:, 0], value

def apply_waits(runner, reset_mask, maximum):
    """Only physical dynamics run here. No policy/teacher calls or learner RNG."""
    keys, delays = draw_delays(runner.delay.keys, maximum)
    delays = jnp.where(reset_mask, delays, 0)
    def step(t, state):
        def one(s, fixed):
            f, _ = spatial_wall_pass_step_with_transition(s, 0, SpatialWallPassVariant.FIXED)
            m, _ = spatial_wall_pass_step_with_transition(s, 0, SpatialWallPassVariant.MUTANT)
            return gc._tree_where(fixed, f, m)
        candidate = jax.vmap(one)(state, runner.mixture.fixed)
        return gc._tree_where(t < delays, candidate, state)
    state = jax.lax.fori_loop(0, maximum, step, runner.env_state)
    hist = jnp.stack([jnp.bincount(delays, weights=(reset_mask & mask).astype(jnp.int32), length=9)
                      for mask in (runner.mixture.fixed, ~runner.mixture.fixed)])
    forced = jnp.stack([jnp.sum(jnp.where(mask, delays, 0))
                        for mask in (runner.mixture.fixed, ~runner.mixture.fixed)])
    delay = runner.delay.replace(
        keys=jnp.where(reset_mask[:, None], keys, runner.delay.keys),
        current=jnp.where(reset_mask, delays, runner.delay.current),
        forced_transitions=runner.delay.forced_transitions + forced,
        histogram=runner.delay.histogram + hist)
    return runner.replace(env_state=state, delay=delay)

def attach_delay(runner, seed, probability, maximum=8):
    if not 0 <= maximum <= 8:
        raise ValueError("Delay maximum outside approved training range")
    mixed = attach_mixture(runner, seed, probability)
    n = runner.command_active.shape[0]
    keys = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed), DELAY_SALT), n)
    delay = DelayState(keys, jnp.zeros(n, jnp.int32), jnp.zeros(2, jnp.int32),
                       jnp.zeros((2, 9), jnp.int32))
    out = DelayRunner(**{f.name: getattr(mixed, f.name) for f in fields(MixedRunner)}, delay=delay)
    return apply_waits(out, jnp.ones(n, bool), maximum)

def make_delay_outcome(probability, maximum=8):
    def outcome(runner, actions, config):
        runner, event = mixed_step(runner, actions, config, probability)
        # world_done already terminates targets across the reset boundary.
        runner = apply_waits(runner, event.world_done, maximum)
        return (runner, event.done, event.valid, event.reward, event.terminal_goals,
                event.world_done, event.goal_done, event.observed_goals)
    return outcome

def exposure(runner):
    result = mixture_metrics(runner)
    result['collector_slots'] = result['transitions']['total']
    forced = np.asarray(runner.delay.forced_transitions, dtype=np.int64)
    result['forced_wait_transitions'] = int(forced.sum())
    result['physical_environment_transitions'] = result['collector_slots'] + int(forced.sum())
    result['valid_learner_samples'] = result['active_transitions']['total']
    result['delay_histogram_by_kernel'] = np.asarray(runner.delay.histogram).tolist()
    result['forced_waits_by_kernel'] = forced.tolist()
    return result
