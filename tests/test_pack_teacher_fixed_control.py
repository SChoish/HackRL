import json
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_pack_teacher_fixed_control_v1 as run
import probe_pack_teacher_fixed_control_v1 as probe
from hackrl.pack_teacher_holdout import deserialize_state, make_evaluator, Witness, mutant_step
from hackrl.pack_teacher_fixed_trace import make_traced_evaluator
from hackrl.dual_leo import _teacher_td_loss, bc_policy_coefficient


def test_exact_budget_config_and_arm():
    jobs=run.jobs();assert len(jobs)==5 and [j['seed'] for j in jobs]==list(range(200,205))
    assert 5*4096*512*64==run.SPEC['budget']['collector_slots']==671088640
    assert 5*16*2*(1+8)==1440
    for j in jobs:
        cfg=json.loads(json.dumps(asdict(run.base._job_config(j,goal_mode='deliver_3',variant='fixed',updates=4096))))
        old=run.read(run.OLD/f"D-delivery/seed{j['seed']}/checkpoints/adapt_4096/config.json")
        assert {k:(old[k],v) for k,v in cfg.items() if v!=old[k]}=={'variant':('mutant','fixed')}
        assert run.base._arm(j)==run.read(run.OLD/f"D-delivery/seed{j['seed']}/checkpoints/adapt_4096/arm.json")


def test_delivery_loss_scaling_schedule():
    x=jnp.arange(48,dtype=jnp.float32).reshape(4,12)/13;target=jnp.ones_like(x);valid=jnp.array([1,1,0,1],jnp.bool_)
    gradient=lambda mask:jax.grad(lambda v:_teacher_td_loss(v,target,valid,mask))(x)
    all_heads=gradient(jnp.ones(12));delivery=gradient(jax.nn.one_hot(11,12))
    np.testing.assert_array_equal(delivery[:,11],all_heads[:,11]);np.testing.assert_array_equal(delivery[:,:11],0)
    assert np.isclose(float(bc_policy_coefficient(512)),.0888888889)
    assert float(bc_policy_coefficient(4608))==0


class ScriptedActor:
    def apply(self,table,maps,numeric,goals):
        tick=jnp.rint((1-numeric[:,4])*128).astype(jnp.int32)
        return SimpleNamespace(logits=table[jnp.clip(tick,0,127),jnp.arange(len(tick))]),jnp.zeros(len(tick))


@pytest.mark.parametrize('kernel',['fixed','mutant'])
@pytest.mark.parametrize('sample',[False,True])
def test_trace_does_not_change_synthetic_evaluation(kernel,sample):
    # Two stored fixtures, scripted actor only. No historical learned policy is queried.
    panel=run.read(run.OLD/'holdout_v2.json');items=panel['states'][:2]
    states=[deserialize_state(x['state']) for x in items]
    initial=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
    table=np.full((128,2,24),-6,np.float32);table[:,:,0]=-2
    for i,item in enumerate(items):
        for offset,a in enumerate(item['normal_solution']):table[int(states[i].tick)+offset,i,a]=4
    table=jnp.asarray(table);keys=jax.random.split(jax.random.PRNGKey(17),2)
    expected=make_evaluator(ScriptedActor(),kernel,sample,1)(table,initial,keys)
    actual,trace=make_traced_evaluator(ScriptedActor(),kernel,sample)(table,initial,keys)
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    flat=run.flatten_trace(jax.device_get(trace))
    assert len(flat['action'])==int(np.asarray(actual['length']).sum())
    np.testing.assert_array_equal(flat['after_tick'],flat['before_tick']+1)
    np.testing.assert_array_equal(flat['created'],np.maximum(flat['total_after']-flat['total_before'],0))
    np.testing.assert_array_equal(flat['grain_charged'],np.minimum(flat['balance_before']+flat['created'],flat['delivered']))
    for i in range(2):
        assert int(flat['grain_charged'][flat['episode_index']==i].sum())==int(actual['violation_grain_delivered'][i])


def test_probe_pending_failed_conflict(tmp_path):
    run.write(tmp_path/'run_manifest.json',{'experiment_id':'pack_teacher_fixed_control_v1'})
    h=probe.sha(tmp_path/'run_manifest.json')
    assert probe.status(tmp_path,h)=={'terminal':False,'outcome':'pending'}
    run.write(tmp_path/'failure.json',{'reason':'synthetic test failure'})
    assert probe.status(tmp_path,h)['science_complete'] is False
    run.write(tmp_path/'complete.json',{})
    with pytest.raises(AssertionError):probe.status(tmp_path,h)


def test_bundle_validation_without_rollout(tmp_path):
    saved={'trace_transitions':1,'records':[{'length':1,'violation_grain_delivered':1}]}
    arrays={k:np.array([v]) for k,v in {'episode_index':0,'before_tick':5,'after_tick':6,
            'total_before':2,'total_after':3,'created':1,'grain_charged':1,'balance_before':0,'delivered':1}.items()}
    arrays['metadata_json']=np.frombuffer(json.dumps(saved).encode(),np.uint8)
    path=tmp_path/'bundle.npz';np.savez_compressed(path,**arrays)
    assert run.load_bundle(path)==saved
    arrays['after_tick']=np.array([8]);np.savez_compressed(path,**arrays)
    with pytest.raises(AssertionError):run.load_bundle(path)


def test_positive_creation_and_delivery_trace():
    item=run.read(run.OLD/'holdout_v2.json')['states'][3]
    state=deserialize_state(item['state'])
    steps=[22]+Witness(mutant_step(state,22)).solve()
    initial=jax.tree.map(lambda x:x[None],state)
    table=np.full((128,1,24),-100,np.float32);table[:,:,0]=0
    for offset,a in enumerate(steps):table[int(state.tick)+offset,0,:]=-100;table[int(state.tick)+offset,0,a]=0
    table=jnp.asarray(table);keys=jax.random.PRNGKey(1)[None]
    expected=make_evaluator(ScriptedActor(),'mutant',False,1)(table,initial,keys)
    actual,trace=make_traced_evaluator(ScriptedActor(),'mutant',False)(table,initial,keys)
    for k in expected:np.testing.assert_array_equal(actual[k],expected[k])
    assert bool(actual['adoption'][0]) and bool(actual['success'][0])
    flat=run.flatten_trace(jax.device_get(trace))
    assert flat['created'].sum()>0 and flat['grain_charged'].sum()>0
    assert flat['stage'][0,2] and not flat['before_anchor_present'][0] and flat['after_anchor_present'][0]


def test_no_repeat_uncommitted_evaluation(tmp_path):
    bundle=tmp_path/'new.npz';intent=tmp_path/'new.intent'
    run.claim_evaluation(bundle,intent,{'episodes':16})
    with pytest.raises(RuntimeError):run.claim_evaluation(bundle,intent,{'episodes':16})
    bundle.write_bytes(b'synthetic committed fixture')
    run.claim_evaluation(bundle,intent,{'episodes':16})


def test_probe_valid_complete_and_tamper(tmp_path):
    jobs=run.jobs();run.write(tmp_path/'run_manifest.json',{'experiment_id':'pack_teacher_fixed_control_v1','jobs':jobs})
    finals={}
    for j in jobs:
        cell=tmp_path/j['condition']/f"seed{j['seed']}";checkpoint=cell/'checkpoints/adapt_4096';checkpoint.mkdir(parents=True)
        for name in ['state.msgpack','config.json','metadata.json','arm.json']:(checkpoint/name).write_bytes(b'fixture-only')
        files={p.name:probe.sha(p) for p in checkpoint.iterdir()};finals[j['id']]=files['state.msgpack']
        run.write(checkpoint/'integrity.json',{'checkpoint_files':files})
        run.write(cell/'summary.json',{'complete':True,'global_update':4608,'collector_slots':134217728,'checkpoint_sha256':files['state.msgpack']})
    ev={};tr={};(tmp_path/'evaluation').mkdir()
    for i in range(20):
        p=tmp_path/'evaluation'/f'{i}.json';run.write(p,{'fixture':i});ev[str(p)]=probe.sha(p)
        p=p.with_suffix('.npz');p.write_bytes(b'fixture-only');tr[str(p)]=probe.sha(p)
    run.write(tmp_path/'result.json',{'complete':True,'new_evaluation_episodes':1440,'new_collector_slots':671088640})
    run.write(tmp_path/'complete.json',{'result_sha256':probe.sha(tmp_path/'result.json'),'final_checkpoint_sha256':finals,'evaluation_sha256':ev,'trace_sha256':tr})
    h=probe.sha(tmp_path/'run_manifest.json')
    assert probe.status(tmp_path,h)['verified_final_checkpoints']==5
    Path(next(iter(tr))).write_bytes(b'tamper')
    with pytest.raises(AssertionError):probe.status(tmp_path,h)
