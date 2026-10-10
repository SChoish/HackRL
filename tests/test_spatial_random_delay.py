import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from hackrl import spatial_wall_pass_gc as gc
from hackrl.spatial_wall_pass import spatial_wall_pass_step_with_transition
from hackrl.spatial_wall_pass_reliability import attach_mixture, mixed_step, base_runner
from hackrl.spatial_random_delay import attach_delay, apply_waits, make_delay_outcome, exposure

def setup(goal='delivery'):
    c=gc.SpatialWallPassGCConfig(num_envs=16,num_steps=4,num_updates=1,minibatch_size=16,goal_mode=goal)
    return c,gc.initialize_spatial_wall_pass_gc(c)[1]

def test_waits_are_physical_time_only_and_do_not_touch_learner():
    c,r=setup()
    a=attach_delay(r,132,.1)
    delays=np.asarray(a.delay.current)
    assert set(delays).issubset(range(9)) and np.any(delays>0)
    for name in ('rng','sampler_keys','train_state','global_update','env_steps','seen_goals','goal_steps'):
        assert serialization.to_bytes(getattr(a,name))==serialization.to_bytes(getattr(r,name))
    np.testing.assert_array_equal(a.env_state.tick,delays)
    assert serialization.to_bytes(a.env_state.replace(tick=r.env_state.tick))==serialization.to_bytes(r.env_state)
    expected=r.env_state
    for t in range(8):
        stepped=jax.vmap(lambda s:spatial_wall_pass_step_with_transition(s,0,'fixed')[0])(expected)
        expected=gc._tree_where(jnp.asarray(t<delays),stepped,expected)
    assert serialization.to_bytes(expected)==serialization.to_bytes(a.env_state)
    x=exposure(a)
    assert x['collector_slots']==0 and x['valid_learner_samples']==0
    assert x['physical_environment_transitions']==int(delays.sum())

def test_zero_delay_matches_existing_mixed_collection():
    c,r=setup()
    a=attach_delay(r,132,.1,0); b=attach_mixture(r,132,.1)
    fn=make_delay_outcome(.1,0)
    for _ in range(65):
        a,*event=fn(a,jnp.zeros(16,jnp.int32),c)
        b,_=mixed_step(b,jnp.zeros(16,jnp.int32),c,.1)
    assert serialization.to_bytes(base_runner(a))==serialization.to_bytes(base_runner(b))
    assert serialization.to_bytes(a.mixture)==serialization.to_bytes(b.mixture)
    assert int(a.delay.forced_transitions.sum())==0

def test_independent_streams_and_goal_switch_does_not_wait():
    c,r=setup('normal12')
    a=attach_delay(r,132,0.); b=attach_delay(r,132,.1)
    np.testing.assert_array_equal(a.delay.keys,b.delay.keys)
    np.testing.assert_array_equal(a.delay.current,b.delay.current)
    a=a.replace(current_goal=jnp.ones(16,jnp.int32))
    out,*events=make_delay_outcome(0.)(a,jnp.zeros(16,jnp.int32),c)
    assert bool(jnp.all(events[5])) # goal_done
    assert serialization.to_bytes(out.delay)==serialization.to_bytes(a.delay)

def test_reset_waits_are_outside_slot_and_mask_accounting_and_resume_exact():
    c,r=setup()
    a=attach_delay(r,132,.1)
    a=a.replace(env_state=a.env_state.replace(tick=jnp.full(16,63,jnp.int32)),
                command_active=jnp.arange(16)%2==0)
    fn=jax.jit(lambda r:make_delay_outcome(.1)(r,jnp.zeros(16,jnp.int32),c))
    restored=serialization.from_bytes(a,serialization.to_bytes(a))
    out,*events=fn(a)
    assert serialization.to_bytes(fn(restored))==serialization.to_bytes(fn(a))
    np.testing.assert_array_equal(out.env_state.tick,out.delay.current)
    assert bool(jnp.all(events[4])) # world_done
    x=exposure(out)
    assert x['collector_slots']==16 and x['valid_learner_samples']==8
    assert x['forced_wait_transitions']==int(a.delay.current.sum()+out.delay.current.sum())
    assert int(out.delay.histogram.sum())==32
    assert int(out.mixture.completed.sum())==16
    assert int(out.env_steps-a.env_steps)==16

def test_training_wait_distribution_and_kernel_independence():
    from hackrl.spatial_random_delay import draw_delays, DELAY_SALT
    from hackrl.spatial_wall_pass_reliability import draw, KERNEL_RNG_SALT
    n=90000
    keys=jax.random.split(jax.random.fold_in(jax.random.PRNGKey(132),DELAY_SALT),n)
    _,values=draw_delays(keys,8)
    _,fixed=draw(jax.random.split(jax.random.fold_in(jax.random.PRNGKey(132),KERNEL_RNG_SALT),n),.1)
    hist=np.bincount(np.asarray(values),minlength=9)
    assert np.max(np.abs(hist/n-1/9))<.005
    for d in range(9):
        assert abs(float(np.asarray(fixed)[np.asarray(values)==d].mean())-.1)<.02
