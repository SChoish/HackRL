import json
from pathlib import Path
import distrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hackrl import spatial_architecture as a
from hackrl import spatial_layout_eval as e
from hackrl import spatial_time_history_eval as old
from hackrl import spatial_wall_pass as env


class TestPolicy:
    def apply(self,params,maps,numeric,goal):
        del params,maps,goal
        x=numeric[:,-1,:13]
        logits=jnp.sin(x.sum(-1)[:,None]+jnp.arange(23)[None,:])
        return distrax.Categorical(logits=logits),jnp.zeros(x.shape[0])


@pytest.mark.parametrize('sample,repeats',[(False,1),(True,8)])
@pytest.mark.parametrize('delay',[0,12])
def test_external_initial_state_preserves_original_evaluator(sample,repeats,delay):
    model=TestPolicy()
    initial=jax.vmap(lambda i:env.make_spatial_wall_pass_state(i,split='validation'))(jnp.arange(16))
    fresh=e.make_rollout(model,sample,repeats)
    legacy=old.make_rollout(model,sample,repeats,history_api=a)
    for fixed in (False,True):
        actual=jax.device_get(fresh({},jnp.int32(132),fixed,jnp.int32(delay),initial,jnp.arange(16)))
        expected=jax.device_get(legacy({},jnp.int32(132),fixed,jnp.int32(delay),False))
        for key in actual:np.testing.assert_array_equal(actual[key],expected[key],err_msg=key)
        assert np.all(actual['active'].sum(0)<=64)
        assert np.all(actual['forced'].sum(0)==delay)


def test_return_definitions_failure_and_joint():
    from hackrl.spatial_recovery_probe import records_from_trace
    # Synthetic trace tests metric arithmetic only, not scientific evidence.
    n=16;t=20
    trace={k:np.zeros((t,n),bool) for k in ('active','forced','wall_pass','blocked','trigger','delivery','beneficial','dash','clear_dash')}
    trace.update(action=np.zeros((t,n),int),position=np.zeros((t,n,2),int),next_position=np.zeros((t,n,2),int))
    trace['active'][:]=True;trace['forced'][:12]=True;trace['delivery'][-1,:8]=True;trace['wall_pass'][14,:]=True
    maps=[dict(map_id=str(i),cell=['short','large_gain'],fixture=dict(family='test'),oracle=dict(L0=14,Lcross=8)) for i in range(16)]
    mutant=e.decorate(trace,maps,1,12,'mutant','mode')
    assert mutant[0]['control_return']==env.DISCOUNT**7
    assert mutant[0]['discounted_return']==env.DISCOUNT**19
    assert mutant[0]['regret']==0
    assert mutant[8]['control_return']==0 and mutant[8]['regret']==env.DISCOUNT**7
    fixed=[{**r,'kernel':'fixed','success':False,'control_return':0.,'discounted_return':0.,'regret':env.DISCOUNT**13} for r in mutant]
    pairs=e.pair_rows(mutant+fixed)
    assert all(x['joint']==0 for x in pairs)


def test_primary_bootstrap_uses_paired_five_seeds_and16_maps():
    maps=[dict(map_id=str(i),cell=['short','large_gain' if i<8 else 'small_gain'],oracle=dict(A=4 if i<8 else 2)) for i in range(16)]
    results=[]
    for arch in ('current_mlp','history_mlp'):
        for seed in range(132,137):
            episodes=[]
            for m in maps:
                for kernel in ('fixed','mutant'):
                    crossing=arch=='history_mlp' and seed==132
                    episodes.append(dict(selection='mode',delay=12,map_id=m['map_id'],repeat=0,cell='/'.join(m['cell']),
                        family='unit',kernel=kernel,success=True,wall_pass=crossing,
                        control_return=.9,discounted_return=.8,regret=.05))
            results.append(dict(architecture=arch,seed=seed,fixed_percent=10,episodes=episodes))
    r=e.primary_bootstrap(results,maps,repeats=10000)
    assert r['learner_seeds']==list(range(132,137))
    assert r['metrics']['joint']['seed_differences']==[1,0,0,0,0]
    assert r['metrics']['joint']['paired_mean']==.2
    assert r['metrics']['control_return_gap']['seed_only95']==[0,0]
    assert r==e.primary_bootstrap(results,maps,repeats=10000)


def test_revision_preserves_old_design_and_separate_budgets():
    root=Path(__file__).resolve().parents[1]
    oldspec=json.loads((root/'docs/manifests/spatial_layout_panel_v1.json').read_text())
    spec=json.loads((root/'docs/manifests/spatial_layout_transfer_v2.json').read_text())
    assert not oldspec['authorization']['policy_evaluation']
    assert spec['authorization']['frozen_evaluation'] and not spec['authorization']['training']
    assert spec['matrix']['mode_episodes']==40*32*2*2
    assert spec['matrix']['sample_episodes']==2*5*32*2*2*8
    assert spec['matrix']['total_episodes']==15360
