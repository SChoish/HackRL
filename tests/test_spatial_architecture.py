import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hackrl import spatial_architecture as a
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_wall_pass as e
from hackrl import spatial_time_history_eval as ev
from hackrl.dual_leo import save_dual_checkpoint, load_dual_checkpoint, make_dual_leo_update


def config():
    return gc.SpatialWallPassGCConfig(num_envs=16,num_steps=4,minibatch_size=16,
                                      policy_hidden_size=16,teacher_hidden_size=16,num_updates=512)


def equal(x,y):
    xx,xt=jax.tree.flatten(x); yy,yt=jax.tree.flatten(y)
    assert xt==yt
    for u,v in zip(xx,yy):np.testing.assert_array_equal(u,v)


@pytest.fixture(scope='module', params=a.ARCHITECTURES)
def initial(request):
    return a.initialize(config(),request.param,small_test=True)


def test_manifest():
    s=json.loads((Path(__file__).resolve().parents[1]/'docs/manifests/spatial_architecture_v1.json').read_text())
    assert s['authorization']==dict(validation=True,long_training=False)
    assert tuple(s['architectures'])==a.ARCHITECTURES
    assert len(s['architectures'])*len(s['seeds'])*3==s['planned_jobs']==60
    assert 4*5*(512+2*4096)*512*64==s['planned_collector_slots']
    assert 28*512*64==s['validation']['diagnostic_collector_slots']
    assert 4*2*11*3*16*9==s['validation']['diagnostic_evaluation_episodes']


def test_window_alignment_and_forgetting(initial):
    _,r,_,_,_=initial
    history=r.history
    for t in range(10):
        state=r.env_state.replace(tick=jnp.full(16,t,jnp.int32))
        history=a.advance_history(history,state,jnp.full(16,t,jnp.int32),jnp.ones(16,bool),jnp.zeros(16,bool))
    equal(history.action,jnp.broadcast_to(jnp.arange(3,10),(16,7)))
    packed=a.pack_inputs(state,r.current_goal,history)
    equal(packed[1][:,:7,4],jnp.broadcast_to(1-jnp.arange(3,10)/64,(16,7)))
    assert not np.asarray(packed[1][:,-1,a.NUMERIC:-1]).any()
    equal(a.advance_history(history,state,jnp.zeros(16,jnp.int32),jnp.zeros(16,bool),jnp.zeros(16,bool)),history)
    cleared=a.advance_history(history,state,jnp.ones(16,jnp.int32),jnp.ones(16,bool),jnp.ones(16,bool))
    assert not np.asarray(cleared.valid).any()


def test_padding_and_auxiliary_information_isolation(initial):
    model,r,t,l,_=initial
    maps,nums,goal=a.runner_inputs(r)
    p,v=model.apply(r.train_state.params,maps,nums,goal)
    variables=dict(params=l.params,batch_stats=l.batch_stats)
    q=t.apply(variables,maps,nums,train=False)
    # Poison masked left padding; even biases and positional encodings must not leak it.
    mm=maps.at[:,:-1].set(17.)
    nn=nums.at[:,:-1,:-1].set(9.)
    pp,vv=model.apply(r.train_state.params,mm,nn,goal)
    equal(p.logits,pp.logits);equal(v,vv)
    equal(q,t.apply(variables,mm,nn,train=False))
    # Valid history changes are invisible to critic and teacher (including renorm updates).
    nn=nn.at[:,:-1,-1].set(1.)
    pp,vv=model.apply(r.train_state.params,mm,nn,goal)
    equal(v,vv)
    equal(t.apply(variables,maps,nums,train=True,mutable=['batch_stats']),
          t.apply(variables,mm,nn,train=True,mutable=['batch_stats']))
    if model.architecture=='current_mlp':equal(p.logits,pp.logits)
    else:assert not np.array_equal(p.logits,pp.logits)
    # Critic cannot affect actor forward.
    changed={**r.train_state.params,'params':{**r.train_state.params['params'],
      'critic_encoder':jax.tree.map(lambda x:x+1.,r.train_state.params['params']['critic_encoder'])}}
    equal(p.logits,model.apply(changed,maps,nums,goal)[0].logits)


def test_causal_block_has_no_future_information():
    block=a.CausalBlock(16)
    x=jax.random.normal(jax.random.PRNGKey(1),(2,8,16));valid=jnp.ones((2,8),bool)
    p=block.init(jax.random.PRNGKey(0),x,valid)
    y=block.apply(p,x,valid)
    yy=block.apply(p,x.at[:,5:].add(100.),valid)
    equal(y[:,:5],yy[:,:5])
    assert not np.array_equal(y[:,5:],yy[:,5:])


def test_auxiliary_initializations_identical():
    previous=None
    for name in a.ARCHITECTURES:
        _,r,_,l,_=a.initialize(config(),name,small_test=True)
        p={k:v for k,v in r.train_state.params['params'].items() if k!='actor'}
        current=(p,l.params,l.batch_stats)
        if previous is not None:equal(current,previous)
        previous=current


def test_wait_and_goal_reset_contract(initial):
    _,r,_,_,_=initial
    for probability in (0.,.1):
        delayed=a.adaptation_start(r,132,probability)
        assert not np.asarray(delayed.history.valid).any()
        delayed=delayed.replace(env_state=delayed.env_state.replace(tick=jnp.full(16,63,jnp.int32)))
        out,*_=a.history_outcome(adaptation=True,probability=probability)(delayed,jnp.zeros(16,jnp.int32),config())
        assert not np.asarray(out.history.valid).any()
        equal(out.env_state.tick,out.delay.current)
    switched=r.replace(current_goal=jnp.ones(16,jnp.int32))
    out,*_=a.history_outcome()(switched,jnp.zeros(16,jnp.int32),config())
    assert np.asarray(out.history.valid[:,-1]).all()


def test_gradient_optimizer_and_disk_restore(initial,tmp_path):
    import shutil
    assert shutil.disk_usage(tmp_path).free>8*1024**3
    assert shutil.disk_usage('/raid/ext_csv').free>8*1024**3
    model,r,t,l,mb=initial
    fn=jax.jit(a.make_update(model,t,config(),mb))
    rr,ll,m=fn(r,l);jax.block_until_ready((rr,ll,m))
    assert int(rr.train_state.step)>int(r.train_state.step)
    assert int(ll.step)>int(l.step)
    assert any(not np.array_equal(x,y) for x,y in zip(jax.tree.leaves(r.train_state.params['params']['actor']),
                                                    jax.tree.leaves(rr.train_state.params['params']['actor'])))
    save_dual_checkpoint(tmp_path/'checkpoint',rr,ll,dict(architecture=model.architecture))
    restored,rl=load_dual_checkpoint(tmp_path/'checkpoint',rr,ll)
    equal((rr,ll),(restored,rl));equal(fn(rr,ll),fn(restored,rl))


def test_bootstrap_uses_successor_window():
    model,r,t,l,mb=a.initialize(config(),'current_mlp',small_test=True);seen=[]
    def inputs(runner):
        packed=a.runner_inputs(runner)
        jax.debug.callback(lambda n:seen.append(np.asarray(n).copy()),packed[1],ordered=True)
        return packed
    fn=jax.jit(make_dual_leo_update(model,t,config(),a.history_outcome(),gc._batch_inputs,mb,runner_inputs=inputs))
    jax.block_until_ready(fn(r,l));jax.effects_barrier()
    assert len(seen)==5
    for tick,numeric in enumerate(seen):
        equal(numeric[:,-1,4],jnp.full(16,1-tick/64))
        assert not numeric[:,-1,a.NUMERIC:-1].any()
        if tick:equal(numeric[:,-2,4],jnp.full(16,1-(tick-1)/64))


def test_evaluation_wait_history():
    from test_spatial_time_history import NoopNetwork
    trace=jax.device_get(ev.make_rollout(NoopNetwork(),False,1,history_api=a)({},132,True,5,False))
    assert not trace['history_valid'][:6].any()
    assert trace['history_valid'][6,:,-1].all()
    records=ev.records_from_trace(trace,1,5)
    assert len(records)==16 and all(r['length']==64 and r['control_length']==59 for r in records)
