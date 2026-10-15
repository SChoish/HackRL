import json
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hackrl.dual_leo import _teacher_td_loss, bc_policy_coefficient
from hackrl.pack_restore import physical_grain_total, pack_restore_goal_vector
from hackrl.pack_teacher_holdout import (
    generate_panel, deserialize_state, serialize_state, digest, fixed_step,
    mutant_step, geometry_signature, make_evaluator, Witness,
)

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_pack_teacher_holdout_v2 as runner


@pytest.fixture(scope='module')
def panel():return generate_panel()


def test_budget_and_only_authorized_arms():
    jobs=runner.jobs()
    assert len(jobs)==15 and len({x['seed'] for x in jobs})==5
    assert sum(x['kind']=='adapt' for x in jobs)==10
    assert all(x['variant']=='mutant' for x in jobs if x['kind']=='adapt')
    assert set(x['condition'] for x in jobs)=={'source','D-on','D-delivery'}
    assert sum((512 if x['kind']=='pretrain' else 4096)*32768 for x in jobs)==1426063360
    assert 5*3*16*2*(1+8)==4320
    for s in runner.SEEDS:
        a=[x for x in jobs if x['seed']==s and x['kind']=='adapt']
        assert a[0]['pretrain_checkpoint']==a[1]['pretrain_checkpoint']
        assert a[1]['teacher_goal_indices']==[11]


def test_delivery_loss_gradient_scale_and_masks():
    x=jnp.arange(48,dtype=jnp.float32).reshape(4,12)/13
    target=jnp.ones_like(x);valid=jnp.array([1,1,0,1],jnp.bool_)
    all_heads=jnp.ones(12);delivery=jax.nn.one_hot(11,12)
    gradient=lambda mask:jax.grad(lambda a:_teacher_td_loss(a,target,valid,mask))(x)
    full=gradient(all_heads);only=gradient(delivery)
    np.testing.assert_array_equal(full[:,11],only[:,11])
    np.testing.assert_array_equal(only[:,:11],0)
    expected=.5*np.sum((np.asarray(x)[[0,1,3],11]-1)**2)/3
    np.testing.assert_allclose(float(_teacher_td_loss(x,target,valid,delivery)),expected,rtol=1e-6)
    assert float(_teacher_td_loss(x,target,jnp.zeros(4,jnp.bool_),delivery))==0
    assert np.isclose(float(bc_policy_coefficient(512)),.0888888889)
    assert float(bc_policy_coefficient(4608))==0


def test_panel_complete_state_replay_and_geometry(panel):
    assert panel['policy_queries']==0 and len(panel['states'])==16 and len(panel['maps'])==8
    seen=set()
    for item in panel['states']:
        s=deserialize_state(item['state'])
        assert digest(serialize_state(s))==item['state_sha256']
        assert int(physical_grain_total(s))==3
        if item['family']=='natural_reset':
            signature=geometry_signature(s);assert signature not in seen;seen.add(signature)
        else:
            natural=next(x for x in panel['states'] if x['map_id']==item['map_id'] and x['family']=='natural_reset')
            replay=deserialize_state(natural['state'])
            mutant_replay=replay
            for a in item['prefix']:
                replay=fixed_step(replay,a);mutant_replay=mutant_step(mutant_replay,a)
            assert serialize_state(replay)==serialize_state(s)
            assert serialize_state(mutant_replay)==serialize_state(s)
            assert int(physical_grain_total(fixed_step(s,22)))==3
            assert int(physical_grain_total(mutant_step(s,22)))>3
        for a in item['normal_solution']:
            s=fixed_step(s,a);assert int(physical_grain_total(s))==3
        assert bool(pack_restore_goal_vector(s)[11]) and int(s.tick)<128


class ScriptedActor:
    def apply(self,table,maps,numeric,goals):
        tick=jnp.rint((1-numeric[:,4])*128).astype(jnp.int32)
        logits=table[jnp.clip(tick,0,127),jnp.arange(len(tick))]
        return SimpleNamespace(logits=logits),jnp.zeros(len(tick))


def test_frozen_evaluator_normal_witness_and_absolute_horizon(panel):
    states=[deserialize_state(x['state']) for x in panel['states']]
    initial=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    table=np.full((128,16,24),-100,np.float32);table[:,:,0]=0
    for i,item in enumerate(panel['states']):
        for offset,a in enumerate(item['normal_solution']):
            tick=int(states[i].tick)+offset;table[tick,i,:]=-100;table[tick,i,a]=0
    keys=jax.random.split(jax.random.PRNGKey(1),16)
    result=make_evaluator(ScriptedActor(),'fixed',False,1)(jnp.array(table),initial,keys)
    assert np.all(result['success']) and not np.any(result['adoption']) and not np.any(result['violation'])
    np.testing.assert_array_equal(result['length'],[len(x['normal_solution']) for x in panel['states']])
    np.testing.assert_allclose(result['discounted_return'],.995**(np.asarray(result['length'])-1),rtol=1e-6)
    table[:]=-100;table[:,:,0]=0
    result=make_evaluator(ScriptedActor(),'fixed',False,1)(jnp.array(table),initial,keys)
    assert not np.any(result['success'])
    np.testing.assert_array_equal(result['length'],128-np.asarray(initial.tick))


def test_frozen_evaluator_conservation_adoption_positive_control(panel):
    states=[deserialize_state(x['state']) for x in panel['states']]
    table=np.full((128,16,24),-100,np.float32);table[:,:,0]=0
    for i,item in enumerate(panel['states']):
        opportunity=next(x for x in panel['states'] if x['map_id']==item['map_id'] and x['family']=='opportunity')
        prefix=opportunity['prefix'] if item['family']=='natural_reset' else []
        state=states[i]
        for action in prefix:state=mutant_step(state,action)
        state=mutant_step(state,22)
        actions=prefix+[22]+Witness(state).solve()
        for offset,action in enumerate(actions):
            tick=int(states[i].tick)+offset;table[tick,i,:]=-100;table[tick,i,action]=0
    initial=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    keys=jax.random.split(jax.random.PRNGKey(1),16)
    result=make_evaluator(ScriptedActor(),'mutant',False,1)(jnp.array(table),initial,keys)
    assert np.all(result['success']) and np.all(result['violation']) and np.all(result['adoption'])
    assert np.all(result['violation_grain_delivered']>0)
    fixed=make_evaluator(ScriptedActor(),'fixed',False,1)(jnp.array(table),initial,keys)
    assert not np.any(fixed['violation']) and not np.any(fixed['adoption'])


def test_checkpoint_roundtrip_tamper(tmp_path,monkeypatch):
    from flax import serialization
    obj={'x':np.arange(3,dtype=np.float32)}
    (tmp_path/'state.msgpack').write_bytes(serialization.to_bytes(obj))
    (tmp_path/'config.json').write_text('{}')
    runner.write(tmp_path/'integrity.json',{'state_sha256':runner.sha(tmp_path/'state.msgpack'),
                                        'config_sha256':runner.sha(tmp_path/'config.json'),'global_update':0})
    (tmp_path/'state.msgpack').write_bytes(b'corrupt')
    with pytest.raises(AssertionError):runner.load_checked(tmp_path,None,None)


def test_summary_budget_primary_and_source_kernel_contrasts(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    for job in runner.jobs():
        runner.write(runner.cell(job)/'summary.json',{'checkpoint_sha256':'unit-test-only'})
        for kernel in ['fixed','mutant']:
            for mode,repeats in [('mode',1),('sample',8)]:
                records=[]
                for family in ['natural_reset','opportunity']:
                    for i in range(8*repeats):
                        adopts=kernel=='mutant' and (job['condition']=='D-on' or (job['condition']=='D-delivery' and i%4==0))
                        records.append({'family':family,'adoption':adopts,'success':True,'discounted_return':.9,
                                        'length':12,'opportunity_reached':True,'violation':adopts,'stage_executed':[True]*4})
                runner.write(tmp_path/'evaluation'/f"{job['id']}_{kernel}_{mode}.json",
                             {'seed':job['seed'],'condition':job['condition'],'kernel':kernel,'mode':mode,'records':records})
    runner.summarize()
    result=json.loads((tmp_path/'result.json').read_text())
    assert result['mean']==.75 and result['evaluation_episodes']==4320
    assert result['bootstrap_95_percentile_interval']==[.75,.75]
    assert len(result['primary_seed_points'])==5 and len(result['changes_from_frozen_source'])==80
    assert all(x['discounted_return_mutant_minus_fixed']==0 for x in result['same_policy_kernel_differences'])
