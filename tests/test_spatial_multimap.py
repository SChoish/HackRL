from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hackrl import spatial_architecture as a,spatial_multimap as m,spatial_wall_pass_gc as gc,spatial_wall_pass as env
from hackrl.spatial_random_delay import exposure

def equal(a,b):
    aa,at=jax.tree.flatten(a);bb,bt=jax.tree.flatten(b);assert at==bt
    for x,y in zip(aa,bb):np.testing.assert_array_equal(x,y)

@pytest.fixture(scope='module')
def data():
    c=gc.SpatialWallPassGCConfig(seed=140,num_envs=16,num_steps=4,minibatch_size=16,
        policy_hidden_size=16,teacher_hidden_size=16,num_updates=512)
    model,r,t,l,mb=a.initialize(c,'history_mlp',small_test=True)
    panel=jax.vmap(lambda i:env.make_spatial_wall_pass_state(i))(jnp.arange(3))
    return c,model,r,t,l,mb,panel

def test_original_reset_override_is_exact(data):
    c,_,r,_,_,_,_=data
    r=r.replace(env_state=r.env_state.replace(tick=jnp.full(16,63,jnp.int32)))
    resets=jax.vmap(env.reset_spatial_wall_pass_worker)(jnp.arange(16))
    actions=jnp.zeros(16,jnp.int32)
    equal(gc.step_spatial_wall_pass_gc_workers(r,actions,c),gc.step_spatial_wall_pass_gc_workers(r,actions,c,reset_override=resets))

def test_map_delay_and_kernel_only_reset_on_world_end(data):
    c,_,r,_,_,_,p=data;r=m.start(r,p,c.seed,probability=.1)
    old=r
    out,*_=m.outcome(p,.1)(r,jnp.zeros(16,jnp.int32),c)
    equal(out.maps.keys,r.maps.keys);equal(out.maps.current,r.maps.current)
    equal(out.mixture.keys,r.mixture.keys);equal(out.delay.keys,r.delay.keys)
    r=r.replace(env_state=r.env_state.replace(tick=jnp.full(16,63,jnp.int32)))
    out,*_=m.outcome(p,.1)(r,jnp.zeros(16,jnp.int32),c)
    equal(out.env_state.tick,out.delay.current)
    equal(out.env_state.wall_mask,p.wall_mask[out.maps.current])
    equal(out.env_state.camp_position,p.camp_position[out.maps.current])
    assert not np.asarray(out.history.valid).any()
    assert int(out.maps.started.sum())==32 and int(out.maps.completed.sum())==16
    assert int(out.maps.slots.sum())==16
    assert int(out.maps.forced.sum())==exposure(out)['forced_wait_transitions']
    assert not np.array_equal(old.maps.keys,out.maps.keys)

def test_branches_preserve_learning_state_and_common_map_waits(data):
    c,_,r,_,_,_,p=data;r=m.start(r,p,c.seed)
    branches=[m.start(r,p,c.seed,probability=q,delivery=True) for q in (1.,0.,.1)]
    for b in branches:
        equal(b.train_state,r.train_state);equal(b.rng,r.rng);equal(b.env_keys,r.env_keys)
        equal(b.maps,branches[0].maps);equal(b.env_state,branches[0].env_state)
        equal(b.delay.keys,branches[0].delay.keys)
        assert np.all(np.asarray(b.current_goal)==gc.DELIVERY_GOAL_INDEX)
    assert np.asarray(branches[0].mixture.fixed).all()
    assert not np.asarray(branches[1].mixture.fixed).any()

@pytest.mark.parametrize('maximum',[0,8])
def test_gradient_and_accounting(data,maximum):
    c,model,r,t,l,mb,p=data;r=m.start(r,p,c.seed,maximum=maximum)
    out,teacher,metrics=jax.jit(m.make_update(model,t,c,mb,p,1.,maximum))(r,l)
    jax.block_until_ready((out,teacher,metrics))
    assert int(out.train_state.step)>int(r.train_state.step)
    assert int(teacher.step)>int(l.step)
    assert int(out.maps.slots.sum())==c.batch_size
    assert int(out.maps.valid.sum())==int(metrics['ppo_valid_transitions'])
    assert int(out.maps.forced.sum())==exposure(out)['forced_wait_transitions']
    if maximum==0:assert int(out.maps.forced.sum())==0
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(r.train_state.params),jax.tree.leaves(out.train_state.params)))

@pytest.mark.parametrize('n,repeats',[(18,1),(18,8),(36,1)])
def test_non16_evaluation_batch(n,repeats):
    from hackrl import spatial_layout_eval as ev
    from hackrl.spatial_recovery_probe import records_from_trace
    from test_spatial_time_history import NoopNetwork
    states=jax.vmap(lambda i:env.make_spatial_wall_pass_state(i%16))(jnp.arange(n))
    tr=jax.device_get(ev.make_rollout(NoopNetwork(),False,repeats)({},140,True,12,states,jnp.arange(n)))
    rows=records_from_trace(tr,repeats,12)
    assert len(rows)==n*repeats
    assert {x['layout_index'] for x in rows}==set(range(n))
    assert all(x['length']==64 and x['control_length']==52 and not x['success'] for x in rows)
