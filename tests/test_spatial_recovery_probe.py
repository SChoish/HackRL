import jax
import jax.numpy as jnp
import numpy as np
import distrax
import pytest
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_recovery_probe as probe


def equal(a,b):
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(b)):
        np.testing.assert_array_equal(x,y)


@pytest.mark.parametrize('delay',[0,2,5])
def test_wait_only_changes_time_all_layouts(delay):
    for layout in range(16):
        s=env.make_spatial_wall_pass_state(layout,split='validation');out=s
        for _ in range(delay): out=env.spatial_wall_pass_step(out,0,'fixed')
        equal(s,out.replace(tick=s.tick))
        a=gc.spatial_wall_pass_gc_inputs(env.observe_spatial_wall_pass(s),gc.DELIVERY_GOAL_INDEX)
        b=gc.spatial_wall_pass_gc_inputs(env.observe_spatial_wall_pass(out),gc.DELIVERY_GOAL_INDEX)
        equal(a[0],b[0]);equal(a[2],b[2])
        np.testing.assert_array_equal(np.delete(a[1],probe.TIME_INDEX),np.delete(b[1],probe.TIME_INDEX))
        assert float(b[1][probe.TIME_INDEX])==(64-delay)/64


def test_first_wall_attempt_only_and_single_state_update():
    for layout in range(16):
        s=env.make_spatial_wall_pass_state(layout,split='validation')
        actions=env.exploit_reference_actions(jnp.int32(layout))
        s=env.spatial_wall_pass_step(s,actions[0],'mutant')
        out,tr,used,trigger,blocked=probe.intervention_step(s,actions[1],False,True,False)
        expected,et=env.spatial_wall_pass_step_with_transition(s,actions[1],'fixed')
        equal(out,expected);equal(tr,et)
        assert bool(used) and bool(trigger) and bool(blocked)
        assert int(out.tick)==int(s.tick)+1
        passed,pt,used2,trigger2,blocked2=probe.intervention_step(out,actions[1],False,True,used)
        equal(passed,env.spatial_wall_pass_step(out,actions[1],'mutant'))
        assert bool(pt.crossed_wall) and bool(used2) and not bool(trigger2) and not bool(blocked2)
        assert int(passed.tick)==int(s.tick)+2


def test_non_crossing_actions_do_not_consume_intervention():
    s=env.make_spatial_wall_pass_state(0,split='validation')
    for action in range(23):
        out,tr,used,trigger,blocked=probe.intervention_step(s,action,False,True,False)
        expected,et=env.spatial_wall_pass_step_with_transition(s,action,'mutant')
        if not bool(et.crossed_wall):
            equal(out,expected);assert not bool(used) and not bool(trigger) and not bool(blocked)


@pytest.mark.parametrize('variant',['fixed','mutant'])
def test_unmodified_kernels_match_on_reference_states(variant):
    for layout in (0,3,8,15):
        s=env.make_spatial_wall_pass_state(layout,split='validation')
        for action in env.normal_reference_actions(jnp.int32(layout)):
            out,tr,_,trigger,_=probe.intervention_step(s,action,variant=='fixed',False,False)
            expected,et=env.spatial_wall_pass_step_with_transition(s,action,variant)
            equal(out,expected);equal(tr,et);assert not bool(trigger)
            s=out


def test_analysis_flags_not_actor_inputs():
    s=env.make_spatial_wall_pass_state(0,split='validation')
    changed=s.replace(episode_wall_passed=jnp.bool_(True),item_collected_after_wall_pass=jnp.bool_(True),delivered_after_wall_pass=jnp.bool_(True))
    equal(gc._batch_inputs(jax.tree.map(lambda x:x[None],s),jnp.array([gc.DELIVERY_GOAL_INDEX])),
          gc._batch_inputs(jax.tree.map(lambda x:x[None],changed),jnp.array([gc.DELIVERY_GOAL_INDEX])))


class IdlePolicy:
    def apply(self,params,maps,numeric,goal):
        logits=jnp.full((maps.shape[0],23),-1000.).at[:,0].set(0.).at[:,6].set(0.)
        return distrax.Categorical(logits=logits),jnp.zeros(maps.shape[0])


def test_forced_wait_preserves_control_rng_and_absolute_horizon():
    rollout=probe.make_rollout(IdlePolicy(),True,1)
    a=jax.device_get(rollout({},jnp.int32(132),jnp.bool_(True),jnp.int32(0),jnp.bool_(False)))
    b=jax.device_get(rollout({},jnp.int32(132),jnp.bool_(True),jnp.int32(5),jnp.bool_(False)))
    np.testing.assert_array_equal(a['action'][:59],b['action'][5:])
    assert np.all(b['action'][:5]==0) and np.all(b['active'])
    assert np.all(b['tick'][:,0]==np.arange(64))
    records=probe.records_from_trace(b,1,5)
    assert all(r['length']==64 and r['control_length']==59 and not r['success'] for r in records)


def test_matrix_budget():
    assert len(probe.CONDITIONS)==7
    assert 3*5*16*7==1680
    assert 3*5*16*7*8==13440
