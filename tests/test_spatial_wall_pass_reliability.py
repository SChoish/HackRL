import json
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_wall_pass import make_spatial_wall_pass_state, SpatialWallPassVariant, spatial_wall_pass_step
from hackrl.spatial_wall_pass_reliability import attach_mixture, base_runner, mixed_step, mixture_metrics

def small(goal="delivery"):
    c=gc.SpatialWallPassGCConfig(num_envs=8,num_steps=4,num_updates=1,minibatch_size=8,goal_mode=goal)
    return c,gc.initialize_spatial_wall_pass_gc(c)[1]

def test_mixture_invisible_and_does_not_consume_inherited_rng():
    c,r=small()
    a=attach_mixture(r,132,.01); b=attach_mixture(r,132,.1)
    assert serialization.to_bytes(base_runner(a))==serialization.to_bytes(r)
    assert serialization.to_bytes(base_runner(b))==serialization.to_bytes(r)
    for x,y in zip(gc._batch_inputs(a.env_state,a.current_goal),gc._batch_inputs(b.env_state,b.current_goal)):
        np.testing.assert_array_equal(x,y)
    assert np.all(np.asarray(a.mixture.fixed)<=np.asarray(b.mixture.fixed))

def test_goal_completion_does_not_resample_but_world_reset_does():
    c,r=small("normal12")
    r=r.replace(current_goal=jnp.full(8,1,dtype=jnp.int32)) # item_absent true at reset
    r=attach_mixture(r,132,.1)
    after,event=mixed_step(r,jnp.zeros(8,dtype=jnp.int32),c,.1)
    assert bool(jnp.all(event.goal_done))
    assert not bool(jnp.any(event.world_done))
    np.testing.assert_array_equal(after.mixture.keys,r.mixture.keys)
    np.testing.assert_array_equal(after.mixture.fixed,r.mixture.fixed)
    before=after.replace(env_state=after.env_state.replace(tick=jnp.full(8,63,dtype=jnp.int32)))
    end,event=mixed_step(before,jnp.zeros(8,dtype=jnp.int32),c,.1)
    assert bool(jnp.all(event.world_done))
    assert not np.array_equal(end.mixture.keys,before.mixture.keys)
    assert int(end.mixture.completed.sum())==8
    assert int(end.mixture.started.sum())==16

def test_endpoints_match_original_steps_and_hidden_worker_kernels():
    c,r=small()
    s=make_spatial_wall_pass_state(jnp.int32(0))
    s=spatial_wall_pass_step(s,jnp.int32(2),SpatialWallPassVariant.FIXED)
    r=r.replace(env_state=jax.tree.map(lambda x:jnp.repeat(x[None],8,axis=0),s))
    for q,variant in ((0.,"mutant"),(1.,"fixed")):
        source=r.replace(env_state=r.env_state.replace(tick=jnp.zeros(8,dtype=jnp.int32)))
        mixed=attach_mixture(source,132,q)
        actions=jnp.full(8,18,dtype=jnp.int32)
        a,e=mixed_step(mixed,actions,c,q)
        b,f=gc.step_spatial_wall_pass_gc_workers(source,actions,replace(c,variant=variant))
        assert serialization.to_bytes(base_runner(a))==serialization.to_bytes(b)
        assert serialization.to_bytes(e)==serialization.to_bytes(f)
    mixed=attach_mixture(source,132,.1)
    mask=jnp.asarray([True,False]*4)
    mixed=mixed.replace(mixture=mixed.mixture.replace(fixed=mask))
    result,_=mixed_step(mixed,actions,c,.1)
    np.testing.assert_array_equal(result.env_state.player_position[:,1],np.where(mask,8,10))

def test_resume_preserves_mixture_rng_flags_counts_and_active_denominator():
    c,r=small()
    r=attach_mixture(r,132,.1)
    r=r.replace(command_active=jnp.asarray([True,False]*4))
    step=jax.jit(lambda x:mixed_step(x,jnp.zeros(8,dtype=jnp.int32),c,.1)[0])
    a=step(r); payload=serialization.to_bytes(a)
    restored=serialization.from_bytes(r,payload)
    assert serialization.to_bytes(step(restored))==serialization.to_bytes(step(a))
    metrics=mixture_metrics(a)
    assert metrics["transitions"]["total"]==8
    assert metrics["active_transitions"]["total"]==4

def test_success_transition_keeps_reward_terminal_state_and_no_early_reset():
    c,r=small()
    state=r.env_state.replace(
        player_position=r.env_state.camp_position+jnp.asarray([0,1]),
        carrying_item=jnp.ones(8,dtype=bool),
        item_present=jnp.zeros(8,dtype=bool))
    r=attach_mixture(r.replace(env_state=state),132,.1)
    after,event=mixed_step(r,jnp.full(8,22,dtype=jnp.int32),c,.1)
    assert bool(jnp.all(event.reward==1))
    assert bool(jnp.all(event.valid & event.goal_done))
    assert not bool(jnp.any(event.world_done))
    assert bool(jnp.all(event.terminal_goals[:,gc.DELIVERY_GOAL_INDEX]))
    assert bool(jnp.all(after.env_state.delivered_total==1))
    assert not bool(jnp.any(after.command_active))
    np.testing.assert_array_equal(after.mixture.fixed,r.mixture.fixed)


def test_blocked_dash_diagnostic_separates_repeat_recovery_and_no_attempt():
    import distrax
    from hackrl.spatial_wall_pass import normal_reference_actions
    normal=np.asarray(normal_reference_actions(jnp.int32(0))).tolist()
    class ScriptedPolicy:
        def __init__(self,plan):
            self.plan=jnp.asarray(plan+[0]*(64-len(plan)))
        def apply(self,params,maps,numeric,goals):
            t=jnp.clip(jnp.rint(64*(1-numeric[:,4])).astype(jnp.int32),0,63)
            actions=self.plan[t]
            logits=jnp.full((maps.shape[0],23),-1e6).at[jnp.arange(maps.shape[0]),actions].set(0.)
            return distrax.Categorical(logits=logits),jnp.zeros(maps.shape[0])
    initial=jax.tree.map(lambda x:x[None],make_spatial_wall_pass_state(jnp.int32(0)))
    def evaluate(plan):
        return gc.evaluate_spatial_wall_pass_gc_frozen(
            ScriptedPolicy(plan),{},variant="fixed",stochastic=False,
            repeats_per_state=1,seed_base=60000,learner_seed=132,
            record_episodes=True,initial_states=initial,
            diagnose_blocked_dash=True,record_paths=True)
    stuck=evaluate([2]+[18]*63)
    assert stuck["success_rate"]==0
    assert stuck["mean_retries_given_block"]==62
    assert stuck["episode_records"][0]["first_blocked_dash_step"]==2
    recovered=evaluate([2,18,1]+normal)
    assert recovered["success_rate"]==1
    assert recovered["mean_length"]==17
    assert recovered["success_after_block_rate_given_block"]==1
    safe=evaluate(normal)
    assert safe["success_rate"]==1
    assert safe["blocked_episode_count"]==0
    assert safe["success_after_block_rate_given_block"] is None
