#!/usr/bin/env python3
"""Ten approved random-delay branches; full-state resume, no extension."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import jax
import numpy as np
from flax import serialization
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_recovery_probe as probe
from hackrl.dual_leo import load_dual_checkpoint, save_dual_checkpoint, make_dual_leo_update
from hackrl.spatial_wall_pass_reliability import attach_mixture, make_mixed_outcome, MixedRunner
from hackrl.spatial_random_delay import attach_delay, make_delay_outcome, exposure
import run_spatial_wall_pass_reliability as parent
import run_spatial_recovery_probe as frozen

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/spatial_random_delay_v1.json'
SPEC=json.loads(SPEC_PATH.read_text())
ROOT=Path(SPEC['run_root'])
UPDATES=tuple(SPEC['evaluation']['updates'])
SEEDS=tuple(SPEC['seeds'])
PROBS=tuple(SPEC['fixed_probabilities'])
sha=parent.sha
write=parent.write
ROLL={}

def name(seed,q): return f'fixed_{round(q*100):02d}pct_seed_{seed}'

@lru_cache(None)
def provenance():
    assert SPEC['total_collector_slots']==10*4096*512*64
    assert SEEDS==(132,133,134,135,136) and PROBS==(0.,.1)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit execution sources before launch')
    files=sorted(p for group in ('src/hackrl/*.py','scripts/*.py','scripts/*spatial*sh') for p in REPO.glob(group))
    original=json.loads((Path(SPEC['source_root'])/'run_manifest.json').read_text())
    assert original['execution_code_sha']==SPEC['source_training_sha']
    return dict(spec=SPEC,execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        manifest_sha256=sha(SPEC_PATH),source_file_sha256={str(p.relative_to(REPO)):sha(p) for p in files},
        pretrained_sha256={str(s):sha(parent.source(s)/'state.msgpack') for s in SEEDS})

def capacity(event):
    size=max((parent.source(s)/'state.msgpack').stat().st_size for s in SEEDS)+2*1024**2
    existing=sum(1 for p in ROOT.glob('fixed_*pct_seed_*/checkpoints/adapt_*/metadata.json'))
    projected=max(0,80-existing)*size+2*size+SPEC['storage']['artifact_allowance_bytes']
    reserve=max(8*1024**3,int(np.ceil(.2*projected)))
    record=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        chosen_filesystem='/raid/ext_csv',checkpoint_bytes_estimate=size,retained_checkpoint_count=80,
        projected_remaining_write_bytes=projected,safety_reserve_bytes=reserve,
        home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        existing_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        existing_source_run_bytes=sum(p.stat().st_size for p in Path(SPEC['source_root']).rglob('*') if p.is_file()))
    if record['raid']['free']<projected+reserve: raise RuntimeError('RAID peak plus reserve unavailable')
    with (ROOT/'capacity.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    return record

def counters(r,l):
    stats={}
    for path,x in jax.tree_util.tree_flatten_with_path(l.batch_stats)[0]:
        if 'steps' in str(path) or 'count' in str(path): stats[str(path)]=np.asarray(x).tolist()
    return dict(ppo_optimizer_steps=int(r.train_state.step),teacher_optimizer_steps=int(l.step),batch_renorm_counters=stats)

def evaluate(net,params,seed):
    result={}
    for selection,repeats in [('mode',1),('sample',8)]:
        if selection not in ROLL:
            full=probe.make_rollout(net,selection=='sample',repeats)
            keep=('active','forced','action','position','next_position','blocked','wall_pass','delivery','beneficial','dash','clear_dash','trigger')
            ROLL[selection]=jax.jit(lambda p,s,f,d,b,full=full,keep=keep:{k:v for k,v in full(p,s,f,d,b).items() if k in keep})
        result[selection]={}
        for delay in (*range(9),10,12):
            for kernel in ('fixed','mutant','mutant_block_once'):
                trace=jax.device_get(ROLL[selection](params,np.int32(seed),kernel=='fixed',np.int32(delay),kernel=='mutant_block_once'))
                records=probe.records_from_trace(trace,repeats,delay)
                if selection=='sample':
                    for rec in records: rec.pop('path')
                result[selection][f'{kernel}_delay{delay}']=dict(summary=probe.summarize(records),episodes=records)
    return result

def frozen_baselines():
    net=gc.SpatialWallPassGCActorCritic(512)
    for seed in SEEDS:
        for family in ('mutant_only','mixed_10pct'):
            dest=ROOT/'baselines'/f'{family}_{seed}.json'
            path=frozen.checkpoint(family,seed)/'state.msgpack'
            digest=sha(path)
            if dest.exists():
                assert json.loads(dest.read_text())['source_sha256']==digest
                continue
            raw=serialization.msgpack_restore(path.read_bytes())
            result=evaluate(net,raw['gc']['train_state']['params'],seed)
            assert sha(path)==digest
            old=json.loads((Path('/raid/ext_csv/HackRL/runs/spatial_recovery_probe_v1')/f'{family}_{seed}'/'result.json').read_text())
            for mode in ('mode','sample'):
                for kernel in ('fixed','mutant','mutant_block_once'):
                    key=kernel if kernel=='mutant_block_once' else kernel+'_delay0'
                    expected=old['results'][mode][key]['episodes']
                    actual=result[mode][kernel+'_delay0']['episodes']
                    assert len(expected)==len(actual)
                    for a,b in zip(actual,expected): assert all(a[k]==b[k] for k in a),(family,seed,mode,kernel)
            write(dest,dict(source_sha256=digest,execution_sha=provenance()['execution_sha'],natural_parity=True,results=result))
            print(json.dumps(dict(event='baseline_complete',family=family,seed=seed)),flush=True)

def valid_checkpoint(path,seed,q):
    m=json.loads((path/'metadata.json').read_text())
    assert m['execution_sha']==provenance()['execution_sha'] and m['seed']==seed and m['q']==q
    assert m['state_sha256']==sha(path/'state.msgpack')

def save(path,r,l,c,q):
    capacity('checkpoint_'+str(int(r.global_update)))
    if path.exists(): raise RuntimeError('Refuse checkpoint overwrite')
    for x in jax.tree.leaves((r,l)):
        a=np.asarray(x)
        if a.dtype.kind in 'fc' and not np.isfinite(a).all(): raise RuntimeError('Nonfinite state')
    pending=path.with_name(path.name+f'.pending.{os.getpid()}')
    if pending.exists(): raise RuntimeError('Preserve unfinished pending checkpoint for investigation')
    save_dual_checkpoint(pending,r,l,gc.spatial_wall_pass_gc_config_payload(c))
    m=json.loads((pending/'metadata.json').read_text())
    m.update(seed=c.seed,q=q,state_sha256=sha(pending/'state.msgpack'),execution_sha=provenance()['execution_sha'],exposure=exposure(r),counters=counters(r,l))
    write(pending/'metadata.json',m)
    pending.rename(path)
    write(path.parent.parent/'latest.json',dict(checkpoint=str(path),global_update=int(r.global_update)))

def smoke():
    c,net,r,teacher,l,mb=parent.templates(132)
    results=[]
    for q in PROBS:
        for maximum in (0,8):
            a=attach_delay(r,132,q,maximum)
            fn=jax.jit(make_dual_leo_update(net,teacher,c,make_delay_outcome(q,maximum),gc._batch_inputs,mb))
            out,leo,metrics=fn(a,l); jax.block_until_ready(out.global_update)
            repeated=fn(serialization.from_bytes(a,serialization.to_bytes(a)),serialization.from_bytes(l,serialization.to_bytes(l)))
            parent.compare_trees((out,leo,metrics),repeated)
            assert int(out.global_update)==513 and exposure(out)['collector_slots']==32768
            assert int(metrics['ppo_valid_transitions'])==exposure(out)['valid_learner_samples']
            assert int(out.train_state.step-a.train_state.step)==int(metrics['ppo_applied_minibatches'])
            assert int(leo.step-l.step)==int(metrics['teacher_applied_minibatches'])
            parity=None
            if maximum==0:
                ref=jax.jit(make_dual_leo_update(net,teacher,c,make_mixed_outcome(q),gc._batch_inputs,mb))
                b,bl,bm=ref(attach_mixture(r,132,q),l)
                from dataclasses import fields
                mixed=MixedRunner(**{f.name:getattr(out,f.name) for f in fields(MixedRunner)})
                parity=parent.compare_trees((mixed,leo,metrics),(b,bl,bm),1e-6)
            results.append(dict(q=q,delay_max=maximum,zero_delay_max_difference=parity,exposure=exposure(out),counters=counters(out,leo)))
            print(json.dumps(dict(event='smoke_passed',q=q,delay_max=maximum)),flush=True)
    write(ROOT/'smoke.json',dict(status='passed',provenance=provenance(),records=results,collector_slots=327680))

def run_cell(seed,q):
    cell=ROOT/name(seed,q); cell.mkdir(parents=True,exist_ok=True)
    c,net,r,teacher,l,mb=parent.templates(seed)
    origin=dict(seed=seed,q=q,source_sha256=sha(parent.source(seed)/'state.msgpack'),commanded_full_state_sha256=parent.digest(r,l),initial_counters=counters(r,l),execution_sha=provenance()['execution_sha'])
    parent.old._write_or_check(cell/'origin.json',origin,'delay branch origin')
    runner=attach_delay(r,seed,q)
    cps=cell/'checkpoints'; cps.mkdir(exist_ok=True)
    present=[u for u in UPDATES if (cps/f'adapt_{u}').exists()]
    for u in present: valid_checkpoint(cps/f'adapt_{u}',seed,q)
    done=max(present,default=0)
    if present: runner,l=load_dual_checkpoint(cps/f'adapt_{done}',runner,l)
    assert int(runner.global_update)==512+done
    def finish_checkpoint(u):
        path=cps/f'adapt_{u}'
        if not path.exists(): save(path,runner,l,c,q)
        ep=path/'evaluation.json'
        if not ep.exists():
            before=parent.digest(runner,l)
            result=evaluate(net,runner.train_state.params,seed)
            assert before==parent.digest(runner,l)
            write(ep,dict(update=u,results=result,exposure=exposure(runner),counters=counters(runner,l)))
        write(path/'evaluation_integrity.json',dict(sha256=sha(ep)))
    finish_checkpoint(done)
    fn=jax.jit(make_dual_leo_update(net,teacher,c,make_delay_outcome(q),gc._batch_inputs,mb))
    for u in range(done+1,4097):
        start=time.monotonic()
        runner,l,metrics=fn(runner,l); jax.block_until_ready(runner.global_update)
        if not all(np.isfinite(np.asarray(x)).all() for x in metrics.values()): raise RuntimeError('Nonfinite metrics')
        host=parent.old._host_metrics(metrics,u,time.monotonic()-start)
        if u%32==0:
            host.update(exposure=exposure(runner),counters=counters(runner,l))
            with (cell/'updates.jsonl').open('a') as f: f.write(json.dumps(host)+'\n')
            print(json.dumps(dict(event='update',cell=cell.name,update=u,seconds=host['seconds'],exposure=host['exposure'])),flush=True)
        if u%64==0: capacity(cell.name+'_'+str(u))
        if u in UPDATES: finish_checkpoint(u)
    assert exposure(runner)['collector_slots']==134217728
    write(cell/'summary.json',dict(status='complete',seed=seed,q=q,updates=4096,collector_slots=134217728,exposure=exposure(runner),counters=counters(runner,l),final_evaluation_path=str(cps/'adapt_4096/evaluation.json'),final_evaluation_sha256=sha(cps/'adapt_4096/evaluation.json')))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    if os.environ.get('HACKRL_DEVICE')!='cuda' or not any(d.platform=='gpu' for d in jax.devices()): raise RuntimeError('Use GPU entrypoint')
    ROOT.mkdir(parents=True,exist_ok=True)
    p=provenance(); capacity('startup')
    parent.old._write_or_check(ROOT/'run_manifest.json',p,'delay experiment')
    if args.smoke: smoke(); return
    if not (ROOT/'smoke.json').exists(): smoke()
    sm=json.loads((ROOT/'smoke.json').read_text()); assert sm['status']=='passed' and sm['provenance']==p
    jax.clear_caches()
    frozen_baselines()
    for seed in SEEDS:
        for q in PROBS:
            run_cell(seed,q)
            ROLL.clear(); jax.clear_caches()
    cells={name(s,q):json.loads((ROOT/name(s,q)/'summary.json').read_text()) for s in SEEDS for q in PROBS}
    assert sum(c['collector_slots'] for c in cells.values())==SPEC['total_collector_slots']
    capacity('complete')
    write(ROOT/'result.json',dict(status='complete',jobs_completed=10,cells=cells,execution_sha=p['execution_sha'],collector_slots=SPEC['total_collector_slots'],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    subprocess.run([os.sys.executable,str(REPO/'scripts/check_spatial_random_delay_complete.py')],check=True)
    write(ROOT/'verified_complete.json',dict(execution_sha=p['execution_sha'],result_sha256=sha(ROOT/'result.json')))

if __name__=='__main__': main()
