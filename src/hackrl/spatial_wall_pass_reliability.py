"""Episode-static hidden kernel mixture; all original actor/teacher RNG is preserved."""
from dataclasses import fields
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_wall_pass import SpatialWallPassVariant, spatial_wall_pass_step_with_transition

KERNEL_RNG_SALT = 0x534D4958

@struct.dataclass
class KernelMixture:
    keys: jax.Array
    fixed: jax.Array
    started: jax.Array
    completed: jax.Array
    transitions: jax.Array
    active_transitions: jax.Array
    deliveries: jax.Array

@struct.dataclass
class MixedRunner(gc.SpatialWallPassGCRunnerState):
    mixture: KernelMixture

def base_runner(runner):
    return gc.SpatialWallPassGCRunnerState(**{
        f.name: getattr(runner, f.name) for f in fields(gc.SpatialWallPassGCRunnerState)})

def counts(mask, valid=None):
    if valid is None:
        valid = jnp.ones_like(mask)
    return jnp.stack((jnp.sum(mask & valid), jnp.sum((~mask) & valid))).astype(jnp.int32)

def draw(keys, probability):
    split = jax.vmap(lambda k: jax.random.split(k, 2))(keys)
    uniform = jax.vmap(lambda k: jax.random.uniform(k, ()))(split[:, 1])
    return split[:, 0], uniform < probability

def attach_mixture(runner, seed, probability):
    if not 0 <= probability <= 1:
        raise ValueError("fixed probability must be in [0, 1]")
    if not bool(jnp.all(runner.env_state.tick == 0)):
        raise ValueError("Initial branch must be a world-reset boundary")
    root = jax.random.fold_in(jax.random.PRNGKey(seed), KERNEL_RNG_SALT)
    keys = jax.random.split(root, runner.command_active.shape[0])
    keys, fixed = draw(keys, probability)
    zero = jnp.zeros(2, dtype=jnp.int32)
    mix = KernelMixture(keys, fixed, counts(fixed), zero, zero, zero, zero)
    return MixedRunner(**{f.name: getattr(runner, f.name)
                         for f in fields(gc.SpatialWallPassGCRunnerState)}, mixture=mix)

def mixed_step(runner, actions, config, probability, *, reset_override=None):
    actions = jnp.where(runner.command_active, actions, 0)
    def step(kernel):
        return jax.vmap(lambda s,a: spatial_wall_pass_step_with_transition(s,a,kernel))(
            runner.env_state, actions)
    fixed = step(SpatialWallPassVariant.FIXED)
    mutant = step(SpatialWallPassVariant.MUTANT)
    stepped = gc._tree_where(runner.mixture.fixed, fixed, mutant)
    new_runner, event = gc.step_spatial_wall_pass_gc_workers(
        runner, actions, config, transition_override=stepped, reset_override=reset_override)
    mix = runner.mixture
    new_keys, new_fixed = draw(mix.keys, probability)
    # Only a world reset commits a draw; a goal switch cannot change a kernel.
    new_mix = mix.replace(
        keys=jnp.where(event.world_done[:,None], new_keys, mix.keys),
        fixed=jnp.where(event.world_done, new_fixed, mix.fixed),
        started=mix.started+counts(new_fixed,event.world_done),
        completed=mix.completed+counts(mix.fixed,event.world_done),
        transitions=mix.transitions+counts(mix.fixed),
        active_transitions=mix.active_transitions+counts(mix.fixed,event.valid),
        deliveries=mix.deliveries+counts(mix.fixed,event.delivery))
    return new_runner.replace(mixture=new_mix), event

def make_mixed_outcome(probability):
    def outcome(runner, actions, config):
        runner,event = mixed_step(runner, actions, config, probability)
        return (runner,event.done,event.valid,event.reward,event.terminal_goals,
                event.world_done,event.goal_done,event.observed_goals)
    return outcome

def mixture_metrics(runner):
    result = {}
    for name in ("started","completed","transitions","active_transitions","deliveries"):
        values = np.asarray(getattr(runner.mixture,name),dtype=np.int64)
        total = int(values.sum())
        result[name] = {"fixed":int(values[0]),"mutant":int(values[1]),
                        "total":total,"fixed_fraction":float(values[0]/total) if total else None}
    return result
