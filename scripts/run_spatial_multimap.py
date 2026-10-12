#!/usr/bin/env python3
"""GPU validation -> three fixed development jobs -> conditionally20 main jobs."""
from dataclasses import replace
import argparse,hashlib,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from hackrl import spatial_architecture as a,spatial_multimap as m,spatial_layout_panel as panel
from hackrl import spatial_layout_eval as ev,spatial_wall_pass_gc as gc
from hackrl.spatial_recovery_probe import records_from_trace,summarize
from hackrl.spatial_random_delay import exposure
from hackrl.dual_leo import save_dual_checkpoint,load_dual_checkpoint
from run_spatial_history_probe import sha,tree_hash,write
from run_spatial_random_delay import counters

REPO=Path(__file__).resolve().parents[1]
SPEC=json.loads((REPO/'docs/manifests/spatial_multimap_v1.json').read_text())
EXEC=json.loads((REPO/'docs/manifests/spatial_multimap_execution_v1.json').read_text())
ROOT=Path(SPEC['root']);PANEL=Path(SPEC['panel_root']);STOP=False;PRETRAIN_WAIT=0

def journal(name,record):
    with (ROOT/name).open('a') as f:f.write(json.dumps(dict(utc=time.time(),**record))+'\n');f.flush();os.fsync(f.fileno())

def capacity(event):
    s=SPEC['storage'];projected=s['checkpoint_bound_bytes']*s['max_retained_and_overlap']+s['artifact_allowance_bytes']
    reserve=max(s['minimum_reserve_bytes'],int(projected*s['reserve_fraction']))
    d=dict(event=event,home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
           existing_run_bytes=sum(x.stat().st_size for x in ROOT.rglob('*') if x.is_file()),
           existing_panel_bytes=sum(x.stat().st_size for x in PANEL.rglob('*') if x.is_file()),
           projected_remaining_bytes=projected,reserve_bytes=reserve,filesystem='/raid/ext_csv')
    journal('capacity.jsonl',d)
    if d['raid']['free']<projected+reserve:raise RuntimeError('Insufficient RAID reserve; no deletion or partial write allowed')

def provenance():
    assert jax.default_backend()=='gpu'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    receipt=json.loads((PANEL/'integrity.json').read_text())
    for name,h in receipt['output_sha256'].items():assert sha(PANEL/name)==h,name
    for name,h in receipt['input_after'].items():assert sha(name)==h,name
    assert json.loads((PANEL/'feasibility_result.json').read_text())['status']=='panel_locked'
    paths=list(REPO.glob('src/hackrl/*.py'))+[Path(__file__).resolve(),REPO/'docs/manifests/spatial_multimap_execution_v1.json']
    return dict(execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        panel_sha256=sha(PANEL/'maps_locked.json'),spec=SPEC,execution_contract=EXEC,
        files={str(x.relative_to(REPO)):sha(x) for x in paths})

def maps_for(split):
    # Scientific test inference is separately guarded in main(); no learner sees test states.
    maps=[x for x in json.loads((PANEL/'maps_locked.json').read_text())['maps'] if x['split']==split]
    states=jax.tree.map(lambda *x:jnp.stack(x),*[panel.kernel_state(x['fixture']) for x in maps])
    return maps,states

def finite(tree):
    for x in jax.tree.leaves(tree):
        x=np.asarray(x)
        if x.dtype.kind in 'fc':assert np.isfinite(x).all()

def config(seed,q=None):
    return gc.SpatialWallPassGCConfig(seed=seed,num_updates=512 if q is None else 4096,
        goal_mode='normal12' if q is None else 'delivery',variant='fixed' if q is None else 'mutant')

def template(seed,q,train):
    c=config(seed,q);model,r,t,l,mb=a.initialize(c,'history_mlp')
    r=m.start(r,train,seed,probability=1. if q is None else q,delivery=q is not None,maximum=PRETRAIN_WAIT if q is None else 8)
    return c,model,r,t,l,mb

class Evaluator:
    def __init__(self,model):
        self.fns={s:ev.make_rollout(model,s=='sample',8 if s=='sample' else 1) for s in ('mode','sample')}
    def __call__(self,r,seed,maps,states,adaptation):
        before=tree_hash(r);out={}
        for selection,repeats in (('mode',1),('sample',8)):
            out[selection]={}
            for delay in (0,12) if adaptation else (0,):
                for kernel in ('fixed','mutant') if adaptation else ('fixed',):
                    tr=jax.device_get(self.fns[selection](r.train_state.params,jnp.int32(seed),kernel=='fixed',jnp.int32(delay),states,jnp.arange(len(maps))))
                    rows=records_from_trace(tr,repeats,delay)
                    for e in rows:
                        item=maps[e['layout_index']];o=item['oracle'];opt=o['L0'] if kernel=='fixed' else min(o['L0'],o['Lcross'])
                        control=gc.DISCOUNT**(e['control_length']-1) if e['success'] else 0.
                        oracle=gc.DISCOUNT**(opt-1)
                        assert control<=oracle+1e-12
                        e.update(map_id=item['map_id'],wall_group=item['wall_group_sha256'],value_cell=item['value_cell'],
                            control_return=control,oracle_control_return=oracle,regret=oracle-control,
                            successful_exploitation=bool(e['success'] and e['wall_pass']))
                        if selection=='sample':e.pop('path')
                    out[selection][f'{kernel}_delay{delay}']=dict(summary=summarize(rows),episodes=rows)
        assert before==tree_hash(r)
        return out

def save(path,r,l,c,q,prov,parent=None):
    capacity('checkpoint '+str(path));finite((r,l));assert not path.exists()
    temp=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    payload=dict(config=gc.spatial_wall_pass_gc_config_payload(c),q=q,panel_sha256=prov['panel_sha256'],pretrain_wait_maximum=PRETRAIN_WAIT)
    save_dual_checkpoint(temp,r,l,payload)
    assert (temp/'state.msgpack').stat().st_size<SPEC['storage']['checkpoint_bound_bytes']
    meta=json.loads((temp/'metadata.json').read_text());meta.update(execution_sha=prov['execution_sha'],
        state_sha256=sha(temp/'state.msgpack'),parent_pretrain_sha256=parent,counters=counters(r,l),
        exposure=exposure(r),per_map={k:np.asarray(getattr(r.maps,k)).tolist() for k in ('started','completed','slots','valid','forced','commanded','achieved')})
    write(temp/'metadata.json',meta);temp.rename(path)

def check(path,c,q,prov,parent=None):
    meta=json.loads((path/'metadata.json').read_text())
    assert meta['execution_sha']==prov['execution_sha'] and meta['state_sha256']==sha(path/'state.msgpack')
    assert meta['parent_pretrain_sha256']==parent
    assert json.loads((path/'config.json').read_text())==dict(config=gc.spatial_wall_pass_gc_config_payload(c),q=q,panel_sha256=prov['panel_sha256'],pretrain_wait_maximum=PRETRAIN_WAIT)
    return meta

def validate(prov,train,dev):
    dest=ROOT/'validation';dest.mkdir(exist_ok=True)
    if (dest/'verified_complete.json').exists():
        receipt=json.loads((dest/'verified_complete.json').read_text());assert receipt['execution_sha']==prov['execution_sha']
        assert receipt['result_sha256']==sha(dest/'result.json');return
    assert not (dest/'normal').exists(),'Preserve incomplete validation; manual technical review required'
    c,model,r,t,l,mb=template(140,None,train);fn=jax.jit(m.make_update(model,t,c,mb,train,1.,PRETRAIN_WAIT))
    weights=r.train_state.params['params'];count=lambda x:sum(np.size(v) for v in jax.tree.leaves(x))
    parameters=dict(actor=count(weights['actor']),critic=count({k:v for k,v in weights.items() if k!='actor'}),teacher=count(l.params))
    assert parameters==dict(actor=1505527,critic=2103585,teacher=1460494)
    r1,l1,metrics=fn(r,l);jax.block_until_ready((r1,l1,metrics));finite((r1,l1,metrics))
    assert tree_hash(r.train_state.params)!=tree_hash(r1.train_state.params)
    assert int(r1.train_state.step)-int(r.train_state.step)==int(metrics['ppo_applied_minibatches'])>0
    assert int(l1.step)-int(l.step)==int(metrics['teacher_applied_minibatches'])>0
    save(dest/'normal',r1,l1,c,None,prov)
    restored,rl=load_dual_checkpoint(dest/'normal',r1,l1)
    assert tree_hash((restored,rl))==tree_hash((r1,l1))
    first=fn(r1,l1);second=fn(restored,rl);jax.block_until_ready((first,second))
    assert tree_hash(first)==tree_hash(second)
    branches=[]
    for q in SPEC['fixed_probabilities']:
        branch=m.start(r1.replace(global_update=jnp.int32(512)),train,140,probability=q,delivery=True)
        assert tree_hash(branch.train_state)==tree_hash(r1.train_state)
        branches.append(branch);cc=config(140,q);bf=jax.jit(m.make_update(model,t,cc,mb,train,q))
        path=dest/f'q{round(q*100)}';save(path,branch,l1,cc,q,prov)
        br,bl=load_dual_checkpoint(path,branch,l1)
        xx=bf(branch,l1);yy=bf(br,bl);jax.block_until_ready((xx,yy));finite(xx)
        assert tree_hash(xx)==tree_hash(yy)
    assert all(tree_hash(b.env_state)==tree_hash(branches[0].env_state) and tree_hash(b.maps)==tree_hash(branches[0].maps) for b in branches)
    evaluated=Evaluator(model)(r1,140,*dev,False)
    evaluation_count=sum(len(v['episodes']) for selector in evaluated.values() for v in selector.values())
    assert evaluation_count==162
    write(dest/'result.json',dict(status='passed',execution_sha=prov['execution_sha'],update_calls=9,
        disposable_collector_slots=9*32768,evaluation_episodes=162,test_episodes=0,restore_max_abs=0,
        smoke_not_science=True,parameters=parameters,evaluation=evaluated))
    write(dest/'verified_complete.json',dict(execution_sha=prov['execution_sha'],result_sha256=sha(dest/'result.json')))
    print('GPU VALIDATION PASSED',flush=True);jax.clear_caches()

def cell(stage,seed,q):return ROOT/stage/f'seed_{seed}'/('pretraining' if q is None else f'fixed_{round(q*100):03d}pct')

def verify_job(stage,seed,q,prov):
    dest=cell(stage,seed,q);c=config(seed,q)
    parent=None if q is None else sha(cell(stage,seed,None)/'checkpoints/update_512/state.msgpack')
    for u in EXEC['pretrain_evaluation_updates' if q is None else 'adaptation_evaluation_updates']:
        path=dest/'checkpoints'/f'update_{u}';meta=check(path,c,q,prov,parent)
        assert meta['global_update']==u+(0 if q is None else 512)
        assert meta['exposure']['collector_slots']==u*c.batch_size
        assert sum(meta['per_map']['slots'])==u*c.batch_size
        assert sum(meta['per_map']['valid'])==meta['exposure']['valid_learner_samples']
        assert sum(meta['per_map']['forced'])==meta['exposure']['forced_wait_transitions']
        assert json.loads((path/'evaluation_integrity.json').read_text())['sha256']==sha(path/'evaluation.json')
    summary=json.loads((dest/'summary.json').read_text())
    assert summary['collector_slots']==c.num_updates*c.batch_size
    assert summary['parent_pretrain_sha256']==parent
    assert summary['state_sha256']==sha(Path(summary['final_checkpoint'])/'state.msgpack')
    return summary

def summarize_tests(finals):
    rows=[];total=0
    for f in finals:
        results=json.loads(Path(f['path']).read_text())['results']
        for selector in ('mode','sample'):
            for delay in (0,12):
                fixed=results[selector][f'fixed_delay{delay}']['episodes']
                mutant=results[selector][f'mutant_delay{delay}']['episodes'];total+=len(fixed)+len(mutant)
                keyed={(x['map_id'],x['repeat']):x for x in fixed}
                pairs=[]
                for x in mutant:
                    y=keyed[x['map_id'],x['repeat']]
                    pairs.append(dict(value_cell=x['value_cell'],fixed_success=float(y['success']),
                        mutant_success=float(x['success']),crossing=float(x['wall_pass']),
                        successful_exploitation=float(x['success'] and x['wall_pass']),
                        joint=float(x['success'] and x['wall_pass'] and y['success']),
                        fixed_return=y['discounted_return'],mutant_return=x['discounted_return'],
                        kernel_return_gap=x['discounted_return']-y['discounted_return'],
                        fixed_regret=y['regret'],mutant_regret=x['regret']))
                for group in ('large_gain','small_gain','harmful','positive'):
                    selected=[x for x in pairs if x['value_cell']==group or (group=='positive' and x['value_cell']!='harmful')]
                    rows.append(dict(seed=f['seed'],q=f['q'],selection=selector,delay=delay,group=group,
                        **{k:float(np.mean([x[k] for x in selected])) for k in selected[0] if k!='value_cell'}))
    assert total==SPEC['final_evaluation']['episodes']
    comparison={}
    rng=np.random.default_rng(EXEC['bootstrap']['seed']);ids=rng.integers(0,5,(EXEC['bootstrap']['replicates'],5))
    for q0,name in ((0.,'q10_minus_q0'),(1.,'q10_minus_q100')):
        first=[next(x for x in rows if x['seed']==s and x['q']==.1 and x['selection']=='mode' and x['delay']==0 and x['group']=='positive') for s in SPEC['main_seeds']]
        second=[next(x for x in rows if x['seed']==s and x['q']==q0 and x['selection']=='mode' and x['delay']==0 and x['group']=='positive') for s in SPEC['main_seeds']]
        comparison[name]={}
        for metric in ('joint','successful_exploitation','fixed_success','mutant_success','mutant_return','fixed_return','kernel_return_gap'):
            difference=np.array([x[metric]-y[metric] for x,y in zip(first,second)])
            comparison[name][metric]=dict(paired_mean=float(difference.mean()),seed_differences=difference.tolist(),
                seed_only95=np.percentile(difference[ids].mean(1),[2.5,97.5]).tolist())
    return dict(seed_rows=rows,comparison=comparison,test_episodes=total,unit='five learners; not episodes',
                primary='q10_minus_q0 joint; mode positive24 test maps delay0')

def run_cell(stage,seed,q,train,dev,prov):
    dest=cell(stage,seed,q);cpdir=dest/'checkpoints';cpdir.mkdir(parents=True,exist_ok=True)
    c,model,r,t,l,mb=template(seed,q,train);parent=None
    if q is not None:
        source=cell(stage,seed,None)/'checkpoints/update_512';parent=sha(source/'state.msgpack')
        check(source,config(seed),None,prov)
        # Same Runner shapes permit restoring the entire normal learner before the phase reset.
        r,l=load_dual_checkpoint(source,r,l)
        r=m.start(r,train,seed,probability=q,delivery=True)
    origin=dict(parent_pretrain_sha256=parent,initial_state_sha256=tree_hash((r,l)),execution_sha=prov['execution_sha'])
    if (dest/'origin.json').exists():assert json.loads((dest/'origin.json').read_text())==origin
    else:write(dest/'origin.json',origin)
    existing={int(x.name.split('_')[1]):x for x in cpdir.glob('update_*') if x.name.split('_')[1].isdigit()}
    for path in existing.values():check(path,c,q,prov,parent)
    done=max(existing,default=0)
    if existing:r,l=load_dual_checkpoint(existing[done],r,l)
    schedule=EXEC['pretrain_evaluation_updates' if q is None else 'adaptation_evaluation_updates']
    evaluator=Evaluator(model)
    def checkpoint(u):
        path=cpdir/f'update_{u}'
        if not path.exists():save(path,r,l,c,q,prov,parent)
        if u in schedule:
            ep=path/'evaluation.json'
            if not ep.exists():write(ep,dict(results=evaluator(r,seed,*dev,q is not None)))
            integrity=path/'evaluation_integrity.json'
            if integrity.exists():assert json.loads(integrity.read_text())['sha256']==sha(ep)
            else:write(integrity,dict(sha256=sha(ep)))
        write(dest/'latest.json',dict(update=u,path=str(path)))
    checkpoint(done)
    assert int(r.global_update)==done+(512 if q is not None else 0)
    fn=jax.jit(m.make_update(model,t,c,mb,train,1. if q is None else q,PRETRAIN_WAIT if q is None else 8))
    for u in range(done+1,c.num_updates+1):
        if STOP:checkpoint(u-1);return False
        attempt=uuid.uuid4().hex
        journal('attempts.jsonl',dict(event='attempt',stage=stage,seed=seed,q=q,update=u,id=attempt,slots=c.batch_size))
        started=time.monotonic();old_ppo=int(r.train_state.step);old_teacher=int(l.step);forced=exposure(r)['forced_wait_transitions']
        r,l,metrics=fn(r,l);jax.block_until_ready((r,l,metrics));finite(metrics)
        assert int(r.train_state.step)-old_ppo==int(metrics['ppo_applied_minibatches'])
        assert int(l.step)-old_teacher==int(metrics['teacher_applied_minibatches'])
        assert int(r.maps.slots.sum())==u*c.batch_size
        assert int(r.maps.valid.sum())==exposure(r)['valid_learner_samples']
        assert int(r.maps.forced.sum())==exposure(r)['forced_wait_transitions']
        entry=dict(event='completed',id=attempt,stage=stage,seed=seed,q=q,update=u,seconds=time.monotonic()-started,
            valid=int(metrics['ppo_valid_transitions']),slots=c.batch_size,forced=exposure(r)['forced_wait_transitions']-forced)
        journal('attempts.jsonl',entry)
        if u%32==0 or u==done+1:
            journal('progress.jsonl',entry|dict(counters=counters(r,l),exposure=exposure(r)));print(json.dumps(entry),flush=True)
        if u%64==0:capacity('update')
        if u in schedule:checkpoint(u)
    final=cpdir/f'update_{c.num_updates}';ep=json.loads((final/'evaluation.json').read_text())['results']
    summary=dict(stage=stage,seed=seed,q=q,updates=c.num_updates,collector_slots=c.num_updates*c.batch_size,
        final_checkpoint=str(final),state_sha256=sha(final/'state.msgpack'),parent_pretrain_sha256=parent,
        mode=ep['mode']['fixed_delay0']['summary']['success_rate'],sample=ep['sample']['fixed_delay0']['summary']['success_rate'],
        counters=counters(r,l),exposure=exposure(r))
    write(dest/'summary.json',summary);print(json.dumps(dict(event='job_complete',**summary)),flush=True)
    jax.clear_caches();return True

def main():
    global PRETRAIN_WAIT
    ROOT.mkdir(parents=True,exist_ok=True);prov=provenance();capacity('startup')
    manifest=ROOT/'run_manifest.json'
    if manifest.exists():assert json.loads(manifest.read_text())==prov
    else:write(manifest,prov)
    trainmaps,train=maps_for('train');dev=maps_for('development')
    write(ROOT/'training_map_order.json',[x['map_id'] for x in trainmaps])
    validate(prov,train,dev)
    if ARGS.validate_only:return
    approval_path=ROOT/'launch_approval.json'
    if not approval_path.exists():
        print('VALIDATION COMPLETE; PRETRAIN WAIT CHOICE REQUIRED BEFORE LEARNING',flush=True);return
    approval=json.loads(approval_path.read_text())
    assert approval['authorized'] is True and approval['execution_sha']==prov['execution_sha']
    assert approval['pretrain_wait_maximum']==0
    PRETRAIN_WAIT=approval['pretrain_wait_maximum']
    runtime=ROOT/'learning_contract.json'
    if runtime.exists():assert json.loads(runtime.read_text())==approval
    else:write(runtime,approval)
    def stop(*_):
        global STOP
        STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    for seed in SPEC['development_seeds']:
        if STOP or not run_cell('development',seed,None,train,dev,prov):return
    summaries=[verify_job('development',s,None,prov) for s in SPEC['development_seeds']]
    passed=all(s['mode']>=.9 and s['sample']>=.9 for s in summaries)
    write(ROOT/'development_gate.json',dict(passed=passed,seeds=summaries,test_episodes=0,execution_sha=prov['execution_sha']))
    if not passed:print('DEVELOPMENT GATE FAILED; MAIN20 CLOSED',flush=True);return
    for seed in SPEC['main_seeds']:
        for q in (None,*SPEC['fixed_probabilities']):
            if STOP or not run_cell('main',seed,q,train,dev,prov):return
    jobs=[verify_job('main',s,q,prov) for s in SPEC['main_seeds'] for q in (None,*SPEC['fixed_probabilities'])]
    assert sum(x['collector_slots'] for x in jobs+summaries)==SPEC['budget']['collector_slots']
    # Only now may a policy see held-out test maps. Evaluate all final pretrain and adaptation states once.
    test=maps_for('test');finals=[]
    for seed in SPEC['main_seeds']:
        for q in (None,*SPEC['fixed_probabilities']):
            dest=cell('main',seed,q);summary=json.loads((dest/'summary.json').read_text());path=Path(summary['final_checkpoint'])
            c,model,r,t,l,mb=template(seed,q,train);check(path,c,q,prov,summary['parent_pretrain_sha256'])
            r,l=load_dual_checkpoint(path,r,l);ep=dest/'final_test.json'
            if not ep.exists():write(ep,dict(results=Evaluator(model)(r,seed,*test,True)))
            receipt=dest/'final_test_integrity.json'
            if receipt.exists():assert json.loads(receipt.read_text())['sha256']==sha(ep)
            else:write(receipt,dict(sha256=sha(ep)))
            finals.append(dict(seed=seed,q=q,path=str(ep),sha256=sha(ep)));jax.clear_caches()
    summary=summarize_tests(finals)
    write(ROOT/'aggregate_result.json',summary)
    write(ROOT/'verified_complete.json',dict(execution_sha=prov['execution_sha'],jobs=23,test_episodes=25920,final_tests=finals,
        aggregate_sha256=sha(ROOT/'aggregate_result.json')))
    print('ALL23 JOBS COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate-only',action='store_true');ARGS=parser.parse_args()
    try:main()
    except Exception:
        import traceback
        ROOT.mkdir(parents=True,exist_ok=True);write(ROOT/'failed.json',dict(utc=time.time(),error=traceback.format_exc()))
        raise
