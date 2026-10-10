from dataclasses import replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from hackrl import spatial_wall_pass as e
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_time_history as h
from hackrl import spatial_time_history_eval as ev
from hackrl.dual_leo import init_dual_leo_teacher, make_dual_leo_update, load_dual_checkpoint, save_dual_checkpoint
from hackrl.spatial_wall_pass_reliability import base_runner
from hackrl.spatial_random_delay import exposure


def config(**kwargs):
    return gc.SpatialWallPassGCConfig(num_envs=16, num_steps=4, num_updates=512,
                                      minibatch_size=16, policy_hidden_size=32, teacher_hidden_size=32, **kwargs)


def equal(a, b):
    aa, ta = jax.tree.flatten(a); bb, tb = jax.tree.flatten(b)
    assert ta == tb
    for x, y in zip(aa, bb):
        np.testing.assert_array_equal(x, y)


@pytest.fixture(scope='module')
def initial():
    return h.initialize(config(), 'A')


def test_manifest_budget_and_closed_gate():
    s = json.loads((Path(__file__).resolve().parents[1]/'docs/manifests/spatial_time_history_v1.json').read_text())
    assert s['authorization'] == dict(validation=True, long_training=False)
    assert len(s['conditions'])*len(s['seeds'])*2 == s['planned_jobs'] == 40
    assert 20*(512+4096)*512*64 == s['planned_collector_slots']
    assert s['validation']['gpu_update_calls']*512*64 == s['validation']['diagnostic_collector_slots']


def test_start_and_real_action_history(initial):
    _, r, _, _, _ = initial
    r = gc.command_delivery(r)
    inputs = h.runner_inputs(r)
    assert not np.asarray(inputs[1][:,h.NUMERIC:]).any()
    actions = jax.vmap(e.exploit_reference_actions)(r.env_state.layout_index)[:,0]
    out, *event = h.history_outcome()(r, actions, config(goal_mode='delivery'))
    raw = gc._batch_inputs(r.env_state, r.current_goal)
    equal(out.history.maps, raw[0]); equal(out.history.numeric, raw[1])
    equal(out.history.action, actions)
    assert np.asarray(out.history.valid).all()


def test_history_disambiguates_first_block_but_not_unlimited_repetition(initial):
    _, r, _, _, _ = initial
    r = gc.command_delivery(r)
    acts = jax.vmap(e.exploit_reference_actions)(r.env_state.layout_index)
    step = h.history_outcome()
    r, *_ = step(r, acts[:,0], config(goal_mode='delivery'))
    before = h.runner_inputs(r)
    r, *_ = step(r, acts[:,1], config(goal_mode='delivery'))
    first = h.runner_inputs(r)
    r, *_ = step(r, acts[:,1], config(goal_mode='delivery'))
    second = h.runner_inputs(r)
    assert not np.array_equal(h.mask_time(before[1],False), h.mask_time(first[1],False))
    equal(first[0], second[0]); equal(h.mask_time(first[1],False),h.mask_time(second[1],False))


def test_waits_world_reset_and_padding_never_enter_history(initial):
    _, r, _, _, _ = initial
    a = h.adaptation_start(r,132)
    assert np.asarray(a.env_state.tick).max() > 0
    assert not np.asarray(a.history.valid).any()
    assert not np.asarray(h.runner_inputs(a)[1][:,h.NUMERIC:]).any()
    a = a.replace(env_state=a.env_state.replace(tick=jnp.full(16,63,jnp.int32)))
    out, *_ = h.history_outcome(adaptation=True)(a,jnp.full(16,18,jnp.int32),config(goal_mode='delivery'))
    assert not np.asarray(out.history.valid).any()
    np.testing.assert_array_equal(out.env_state.tick,out.delay.current)
    assert exposure(out)['collector_slots']==16
    pad = a.replace(command_active=jnp.zeros(16,bool),env_state=a.env_state.replace(tick=jnp.full(16,20,jnp.int32)))
    out, *_ = h.history_outcome(adaptation=True)(pad,jnp.full(16,18,jnp.int32),config(goal_mode='delivery'))
    equal(out.history,pad.history)


def test_goal_switch_keeps_real_history(initial):
    _, r, _, _, _ = initial
    # Item-absent is already true; command switches without a world reset.
    r=r.replace(current_goal=jnp.ones(16,jnp.int32))
    out, *event=h.history_outcome()(r,jnp.zeros(16,jnp.int32),config())
    assert np.asarray(event[5]).all() and not np.asarray(event[4]).any()
    assert np.asarray(out.history.valid).all()


@pytest.mark.parametrize('condition', list(h.CONDITIONS))
def test_time_masks_and_critic_isolation(initial,condition):
    model,r,teacher,leo,_=initial
    actor_time,teacher_time=h.CONDITIONS[condition]
    model=model.clone(actor_time=actor_time);teacher=teacher.clone(teacher_time=teacher_time)
    maps,numeric,goal=h.runner_inputs(r)
    # Valid prior observation exercises BOTH clock coordinates independently.
    numeric=numeric.at[:,-1].set(1.).at[:,h.NUMERIC:2*h.NUMERIC].set(numeric[:,:h.NUMERIC])
    variables=dict(params=leo.params,batch_stats=leo.batch_stats)
    p0,v0=model.apply(r.train_state.params,maps,numeric,goal)
    q0=teacher.apply(variables,maps,numeric,train=False)
    for idx in h.TIME_INDICES:
        changed=numeric.at[:,idx].set(.137)
        p1,v1=model.apply(r.train_state.params,maps,changed,goal)
        q1=teacher.apply(variables,maps,changed,train=False)
        if not actor_time:equal(p0.logits,p1.logits)
        else:assert not np.array_equal(p0.logits,p1.logits)
        if not teacher_time:
            equal(q0,q1)
            # Proves masking precedes updates of normalization statistics too.
            equal(teacher.apply(variables,maps,numeric,train=True,mutable=['batch_stats']),
                  teacher.apply(variables,maps,changed,train=True,mutable=['batch_stats']))
        else:assert not np.array_equal(q0,q1)
        assert not np.array_equal(v0,v1), 'critic must retain the clock in all conditions'
    altered=jax.tree.map(lambda x:x+1,r.train_state.params['params']['critic_encoder'])
    params={**r.train_state.params,'params':{**r.train_state.params['params'],'critic_encoder':altered}}
    equal(p0.logits,model.apply(params,maps,numeric,goal)[0].logits)


def test_all_conditions_same_initial_params_and_shapes(initial):
    _,r,_,l,_=initial
    for condition in ('B','C','D'):
        _,r2,_,l2,_=h.initialize(config(),condition)
        equal(r.train_state.params,r2.train_state.params)
        equal(l.params,l2.params);equal(l.batch_stats,l2.batch_stats)


def test_default_dual_update_adapter_parity():
    c=config()
    net,r=gc.initialize_spatial_wall_pass_gc(c)
    maps,nums,_=gc._batch_inputs(r.env_state,r.current_goal)
    teacher,l,mb=init_dual_leo_teacher(c,maps,nums,gc.NUM_GOALS,gc.NUM_ACTIONS)
    original=jax.jit(make_dual_leo_update(net,teacher,c,gc.spatial_wall_pass_outcome,gc._batch_inputs,mb))
    hooked=jax.jit(make_dual_leo_update(net,teacher,c,gc.spatial_wall_pass_outcome,gc._batch_inputs,mb,
                                     runner_inputs=lambda r:gc._batch_inputs(r.env_state,r.current_goal)))
    equal(original(r,l),hooked(r,l))


def test_full_history_checkpoint_resume_exact(initial,tmp_path):
    import shutil
    assert shutil.disk_usage(tmp_path).free > 8*1024**3
    assert shutil.disk_usage('/raid/ext_csv').free > 8*1024**3
    model,r,teacher,l,mb=initial;c=config()
    fn=jax.jit(h.make_update(model,teacher,c,mb))
    r,l,_=fn(r,l)
    save_dual_checkpoint(tmp_path/'state',r,l,dict(condition='A',config=gc.spatial_wall_pass_gc_config_payload(c)))
    restored,rl=load_dual_checkpoint(tmp_path/'state',r,l)
    equal((r,l),(restored,rl));equal(fn(r,l),fn(restored,rl))


def test_collector_and_final_bootstrap_receive_aligned_history(initial):
    model,r,teacher,l,mb=initial;c=config();seen=[]
    def inputs(runner):
        packed=h.runner_inputs(runner)
        jax.debug.callback(lambda numeric:seen.append(np.asarray(numeric).copy()),packed[1],ordered=True)
        return packed
    fn=jax.jit(make_dual_leo_update(model,teacher,c,h.history_outcome(),gc._batch_inputs,mb,
                                   runner_inputs=inputs))
    out=fn(r,l);jax.block_until_ready(out);jax.effects_barrier()
    assert len(seen)==c.num_steps+1 # collection plus final bootstrap
    for t,n in enumerate(seen):
        np.testing.assert_array_equal(n[:,h.TIME],np.full(16,1-t/64,np.float32))
        if t==0:assert not n[:,h.NUMERIC:].any()
        else:
            assert n[:,-1].all()
            np.testing.assert_array_equal(n[:,h.NUMERIC+h.TIME],np.full(16,1-(t-1)/64,np.float32))
            np.testing.assert_array_equal(n[:,2*h.NUMERIC:-1].sum(axis=1),np.ones(16))


class NoopNetwork:
    def apply(self,params,maps,numeric,goal):
        import distrax
        logits=jnp.full((maps.shape[0],gc.NUM_ACTIONS),-1e9).at[:,0].set(0.)
        return distrax.Categorical(logits=logits),jnp.zeros(maps.shape[0])


def test_evaluation_waits_initial_history_and_episode_budget():
    trace=jax.device_get(ev.make_rollout(NoopNetwork(),False,1)({},132,True,5,False))
    assert not trace['history_valid'][:6].any()
    assert trace['history_valid'][6].all()
    assert trace['forced'][:5].all() and not trace['forced'][5:].any()
    records=ev.records_from_trace(trace,1,5)
    assert len(records)==16 and all(r['length']==64 and r['control_length']==59 for r in records)
