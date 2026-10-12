"""Uniform world-reset maps and per-map accounting, with unchanged Dual learner."""
from dataclasses import fields
import jax
import jax.numpy as jnp
from flax import struct
from hackrl import spatial_architecture as a
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_random_delay import apply_waits,attach_delay
from hackrl.spatial_wall_pass_reliability import base_runner,mixed_step
from hackrl.dual_leo import make_dual_leo_update

MAP_SALT=0x534D4150

@struct.dataclass
class MapAccounting:
    keys: jax.Array
    current: jax.Array
    started: jax.Array
    completed: jax.Array
    slots: jax.Array
    valid: jax.Array
    forced: jax.Array
    commanded: jax.Array
    achieved: jax.Array

@struct.dataclass
class Runner(a.WindowDelayRunner):
    maps: MapAccounting

def draw(keys,n):
    split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
    ids=jax.vmap(lambda k:jax.random.randint(k,(),0,n))(split[:,1])
    return split[:,0],ids

def select(panel,ids):
    return jax.tree.map(lambda x:x[ids],panel).replace(layout_index=ids)

def tally(ids,weights,n):
    return jnp.bincount(ids,weights=weights.astype(jnp.int32),length=n).astype(jnp.int32)

def goal_tally(ids,goals,valid,n):
    return tally(ids*gc.NUM_GOALS+goals,valid,n*gc.NUM_GOALS).reshape(n,gc.NUM_GOALS)

def start(runner,panel,seed,*,probability=1.,delivery=False,maximum=8):
    """Explicit phase boundary: common fresh worlds, never transplant weights."""
    n=panel.tick.shape[0];workers=runner.current_goal.shape[0]
    if isinstance(runner,Runner):keys=runner.maps.keys
    else:keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(seed),MAP_SALT),workers)
    keys,ids=draw(keys,n);base=base_runner(runner).replace(env_state=select(panel,ids))
    if delivery:base=gc.command_delivery(base)
    else:
        gv=jax.vmap(gc._goals)(base.env_state)
        goals,sampler,_,valid=jax.vmap(gc.sample_false_seen_goal,in_axes=(0,None,0))(base.sampler_keys,base.seen_goals,gv)
        base=base.replace(current_goal=goals,sampler_keys=sampler,command_active=valid,goal_steps=jnp.zeros(workers,jnp.int32))
    delayed=a.attach_history(attach_delay(base,seed,probability,maximum))
    zero=jnp.zeros(n,jnp.int32)
    maps=MapAccounting(keys,ids,tally(ids,jnp.ones(workers,bool),n),zero,zero,zero,
        tally(ids,delayed.delay.current,n),goal_tally(ids,delayed.current_goal,jnp.ones(workers,bool),n),jnp.zeros((n,gc.NUM_GOALS),jnp.int32))
    return Runner(**{f.name:getattr(delayed,f.name) for f in fields(a.WindowDelayRunner)},maps=maps)

def outcome(panel,probability,maximum=8):
    n=panel.tick.shape[0]
    def step(runner,actions,config):
        before=runner.maps;keys,ids=draw(before.keys,n)
        out,event=mixed_step(runner,actions,config,probability,reset_override=select(panel,ids))
        out=apply_waits(out,event.world_done,maximum)
        current=jnp.where(event.world_done,ids,before.current)
        weights=jnp.ones_like(event.valid)
        stats=before.replace(keys=jnp.where(event.world_done[:,None],keys,before.keys),current=current,
            started=before.started+tally(ids,event.world_done,n),
            completed=before.completed+tally(before.current,event.world_done,n),
            slots=before.slots+tally(before.current,weights,n),
            valid=before.valid+tally(before.current,event.valid,n),
            forced=before.forced+tally(ids,jnp.where(event.world_done,out.delay.current,0),n),
            commanded=before.commanded+goal_tally(current,out.current_goal,event.world_done|((config.goal_mode=='normal12')&event.goal_done),n),
            achieved=before.achieved+goal_tally(before.current,runner.current_goal,event.goal_done,n))
        history=a.advance_history(runner.history,runner.env_state,actions,event.valid,event.world_done)
        return (out.replace(maps=stats,history=history),event.done,event.valid,event.reward,event.terminal_goals,
                event.world_done,event.goal_done,event.observed_goals)
    return step

def make_update(model,teacher,config,mb,panel,probability,maximum=8):
    return make_dual_leo_update(model,teacher,config,outcome(panel,probability,maximum),gc._batch_inputs,mb,runner_inputs=a.runner_inputs)
