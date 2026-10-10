#!/usr/bin/env python3
"""Fixed-contract M/L follow-up. Existing S training/evaluation code is reused."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import jax
import numpy as np
from flax import serialization
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import load_dual_checkpoint, save_dual_checkpoint, make_dual_leo_update
from hackrl.spatial_random_delay import attach_delay, make_delay_outcome, exposure
import run_spatial_random_delay as adapt
import run_spatial_wall_pass_fixed_gate as fixed

REPO=Path(__file__).resolve().parents[1]
MANIFEST=REPO/'docs/manifests/spatial_delay_scale_v1.json'
SPEC=json.loads(MANIFEST.read_text())
ROOT=Path(SPEC['run_root'])
SROOT=Path(SPEC['s_root'])
sha=adapt.sha
write=adapt.write

def source_s(seed,update=512):
    return Path(SPEC['s_pretrain_root'])/f'dual_seed_{seed}/checkpoints/update_{update}'

def source(size,seed): return ROOT/size/f'pretraining/dual_seed_{seed}/checkpoints/update_512'

def pre_config(size,seed):
    recorded=json.loads((source_s(seed)/'config.json').read_text())
    c=replace(gc.config_from_spatial_wall_pass_gc_payload(recorded),policy_hidden_size=SPEC['policy_widths'][size],teacher_hidden_size=512)
    assert c.num_updates==512 and c.goal_mode=='normal12' and c.variant=='fixed'
    assert (c.num_envs,c.num_steps)==(512,64)
    return c

def branch_config(size,seed):
    return replace(adapt.parent.config(seed),policy_hidden_size=SPEC['policy_widths'][size],teacher_hidden_size=512)

@lru_cache(None)
def provenance():
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit execution sources before launch')
    srun=json.loads((SROOT/'run_manifest.json').read_text())
    receipt=json.loads((SROOT/'verified_complete.json').read_text())
    assert srun['execution_sha']==receipt['execution_sha']==SPEC['s_execution_sha']
    assert receipt['result_sha256']==sha(SROOT/'result.json')
    # No learning, dynamics, observation or delay code has changed since S.
    for key,digest in srun['source_file_sha256'].items():
        if key.startswith('src/') or key=='scripts/run_spatial_random_delay.py':
            assert sha(REPO/key)==digest,key
    return dict(spec=SPEC,execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        s_result_sha256=sha(SROOT/'result.json'),s_manifest_sha256=sha(SROOT/'run_manifest.json'),
        source_file_sha256={str(p.relative_to(REPO)):sha(p) for group in ('src/hackrl/*.py','scripts/*spatial*.py','scripts/*spatial*.sh','docs/manifests/spatial_delay_scale_v1.json') for p in REPO.glob(group)})

def capacity(event):
    storage=SPEC['storage']
    existing=sum(1 for p in ROOT.glob('*/**/checkpoints/*/metadata.json'))
    projected=max(0,210-existing)*storage['conservative_checkpoint_bytes']+2*storage['conservative_checkpoint_bytes']+storage['artifact_allowance_bytes']
    reserve=max(storage['reserve_min_bytes'],int(np.ceil(.2*projected)))
    rec=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),chosen_filesystem='/raid/ext_csv',
        projected_remaining_write_bytes=projected,safety_reserve_bytes=reserve,
        existing_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        existing_s_run_bytes=sum(p.stat().st_size for p in SROOT.rglob('*') if p.is_file()),
        home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict())
    if rec['raid']['free']<projected+reserve: raise RuntimeError('RAID peak plus reserve unavailable')
    with (ROOT/'capacity.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
    return rec

def finite(tree):
    for x in jax.tree.leaves(tree):
        a=np.asarray(x)
        if a.dtype.kind in 'fc' and not np.isfinite(a).all(): raise RuntimeError('Nonfinite state')

def validate_cp(path,c):
    m=json.loads((path/'metadata.json').read_text())
    assert m['execution_sha']==provenance()['execution_sha']
    assert m['state_sha256']==sha(path/'state.msgpack')
    assert json.loads((path/'config.json').read_text())==gc.spatial_wall_pass_gc_config_payload(c)
    integrity=path/'evaluation_integrity.json'
    if integrity.exists():
        assert json.loads(integrity.read_text())['sha256']==sha(path/'evaluation.json')

def pretrain(size,seed):
    c=pre_config(size,seed)
    net,r,t,l,mb=fixed._initialize('dual',c)
    counts=fixed._parameter_counts('dual',r,l)
    cell=ROOT/size/f'pretraining/dual_seed_{seed}'
    cps=cell/'checkpoints'; cps.mkdir(parents=True,exist_ok=True)
    adapt.parent.old._write_or_check(cell/'origin.json',dict(size=size,seed=seed,initial_state_sha256=adapt.parent.digest(r,l),parameter_counts=counts,execution_sha=provenance()['execution_sha']),'fresh pretraining')
    present=[u for u in SPEC['pretrain_checkpoints'] if (cps/f'update_{u}').exists()]
    for u in present: validate_cp(cps/f'update_{u}',c)
    done=max(present,default=0)
    if present:r,l=load_dual_checkpoint(cps/f'update_{done}',r,l)
    assert int(r.global_update)==done
    def checkpoint(u):
        path=cps/f'update_{u}'
        if not path.exists():
            capacity(f'{size}_{seed}_pretrain_{u}');finite((r,l))
            pending=path.with_name(path.name+f'.pending.{os.getpid()}')
            if pending.exists(): raise RuntimeError('Preserve incomplete checkpoint')
            save_dual_checkpoint(pending,r,l,gc.spatial_wall_pass_gc_config_payload(c))
            m=json.loads((pending/'metadata.json').read_text())
            m.update(execution_sha=provenance()['execution_sha'],state_sha256=sha(pending/'state.msgpack'),size=size,seed=seed,parameter_counts=counts,counters=adapt.counters(r,l))
            write(pending/'metadata.json',m);pending.rename(path)
        ep=path/'evaluation.json'
        if not ep.exists():
            before=adapt.parent.digest(r,l)
            e=fixed._evaluate(net,r,c,u)
            assert before==adapt.parent.digest(r,l)
            write(ep,e)
        write(path/'evaluation_integrity.json',dict(sha256=sha(ep)))
        write(cell/'latest.json',dict(checkpoint=str(path),global_update=int(r.global_update)))
    checkpoint(done)
    fn=jax.jit(make_dual_leo_update(net,t,c,gc.spatial_wall_pass_outcome,gc._batch_inputs,mb))
    for u in range(done+1,513):
        start=time.monotonic();r,l,m=fn(r,l);jax.block_until_ready(r.global_update)
        finite(m)
        if u%32==0:
            rec=dict(event='pretrain_update',size=size,seed=seed,update=u,seconds=time.monotonic()-start,
                     counters=adapt.counters(r,l),metrics={k:np.asarray(v).tolist() for k,v in m.items()})
            with (cell/'updates.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print(json.dumps(rec),flush=True)
        if u%64==0:capacity(f'{size}_{seed}_pretrain_{u}')
        if u in SPEC['pretrain_checkpoints']:checkpoint(u)
    final=json.loads((cps/'update_512/evaluation.json').read_text())
    flag=all(final['fixed'][mode]['success_rate']>=.9 for mode in ('mode','sample'))
    write(cell/'summary.json',dict(status='complete',updates=512,collector_slots=16777216,size=size,seed=seed,
        normal_capability_flag=flag,normal_mode_success=final['fixed']['mode']['success_rate'],normal_sample_success=final['fixed']['sample']['success_rate'],parameter_counts=counts,counters=adapt.counters(r,l),final_state_sha256=sha(cps/'update_512/state.msgpack')))
    print(json.dumps(dict(event='pretrain_complete',size=size,seed=seed,normal_capability_flag=flag)),flush=True)

def templates(size,seed):
    c=branch_config(size,seed)
    net,r,t,l,mb=adapt.parent.old._initialize_templates(c)
    cp=source(size,seed)
    validate_cp(cp,pre_config(size,seed))
    adapt.parent.old._validate_branch_config(cp,replace(c,sample_repeats_per_state=64))
    r,l=load_dual_checkpoint(cp,r,l)
    before=adapt.parent.old._command_only_digest(r,l)
    r=gc.command_delivery(r)
    assert before==adapt.parent.old._command_only_digest(r,l) and int(r.global_update)==512
    return c,net,r,t,l,mb

@contextmanager
def adaptation_context(size):
    """Bind only I/O and full-state initialization; reuse S run_cell unchanged."""
    saved=(adapt.ROOT,adapt.provenance,adapt.capacity,adapt.parent.templates,adapt.parent.source)
    adapt.ROOT=ROOT/size/'adaptation';adapt.provenance=provenance;adapt.capacity=capacity
    adapt.parent.templates=lambda seed:templates(size,seed)
    adapt.parent.source=lambda seed:source(size,seed)
    adapt.ROLL.clear()
    try:yield
    finally:
        adapt.ROOT,adapt.provenance,adapt.capacity,adapt.parent.templates,adapt.parent.source=saved
        adapt.ROLL.clear()

def smoke():
    records=[]
    # S is an externally stored reference, not a second copy of the new runner.
    c=pre_config('S',132);net,r,t,l,mb=fixed._initialize('dual',c)
    ref,rl=load_dual_checkpoint(source_s(132,0),r,l)
    adapt.parent.compare_trees((r,l),(ref,rl))
    c=branch_config('S',132);net,r,t,l,mb=adapt.parent.old._initialize_templates(c)
    r,l=load_dual_checkpoint(source_s(132),r,l);r=gc.command_delivery(r)
    a=attach_delay(r,132,.1)
    saved,sl=load_dual_checkpoint(SROOT/'fixed_10pct_seed_132/checkpoints/adapt_0',a,l)
    adapt.parent.compare_trees((a,l),(saved,sl))
    fn=jax.jit(make_dual_leo_update(net,t,c,make_delay_outcome(.1),gc._batch_inputs,mb))
    adapt.parent.compare_trees(fn(a,l),fn(saved,sl))
    records.append(dict(size='S',external_state_and_update_parity=True))
    jax.clear_caches()
    for size in SPEC['new_sizes']:
        c=pre_config(size,132);net,r,t,l,mb=fixed._initialize('dual',c)
        counts=fixed._parameter_counts('dual',r,l)
        print(json.dumps(dict(event='model_initialized',size=size,parameter_counts=counts)),flush=True)
        for phase,q in [('pretrain',None),('adapt',0.),('adapt',.1)]:
            cc=c if phase=='pretrain' else branch_config(size,132)
            a=r if phase=='pretrain' else attach_delay(gc.command_delivery(r.replace(global_update=np.int32(512))),132,q)
            outcome=gc.spatial_wall_pass_outcome if phase=='pretrain' else make_delay_outcome(q)
            fn=jax.jit(make_dual_leo_update(net,t,cc,outcome,gc._batch_inputs,mb))
            restored=serialization.from_bytes({'gc':a,'leo':l},serialization.to_bytes({'gc':a,'leo':l}))
            out,leo,m=fn(a,l);jax.block_until_ready(out.global_update)
            adapt.parent.compare_trees((out,leo,m),fn(restored['gc'],restored['leo']))
            finite((out,leo,m))
            assert int(out.train_state.step-a.train_state.step)==int(m['ppo_applied_minibatches'])
            assert int(leo.step-l.step)==int(m['teacher_applied_minibatches'])
            if phase=='adapt':assert exposure(out)['collector_slots']==32768
            record=dict(size=size,phase=phase,q=q,passed=True,parameter_counts=counts,counters=adapt.counters(out,leo),state_bytes=len(serialization.to_bytes({'gc':out,'leo':leo})))
            assert record['state_bytes']<SPEC['storage']['conservative_checkpoint_bytes']
            records.append(record);print(json.dumps(dict(event='smoke_passed',**record)),flush=True)
            jax.clear_caches()
    write(ROOT/'smoke.json',dict(status='passed',provenance=provenance(),records=records,diagnostic_collector_slots=458752))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-only',action='store_true');args=parser.parse_args()
    if os.environ.get('HACKRL_DEVICE')!='cuda' or not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU entrypoint required')
    ROOT.mkdir(parents=True,exist_ok=True);p=provenance();capacity('startup')
    adapt.parent.old._write_or_check(ROOT/'run_manifest.json',p,'scale execution')
    if not (ROOT/'smoke.json').exists():smoke()
    sm=json.loads((ROOT/'smoke.json').read_text());assert sm['status']=='passed' and sm['provenance']==p
    if args.smoke_only:return
    jax.clear_caches()
    for seed in SPEC['seeds']:
        for size in SPEC['new_sizes']:
            pretrain(size,seed);jax.clear_caches()
            with adaptation_context(size):
                for q in SPEC['fixed_probabilities']:
                    for cp in adapt.ROOT.glob(adapt.name(seed,q)+'/checkpoints/adapt_*'):
                        if '.pending.' not in cp.name:validate_cp(cp,branch_config(size,seed))
                    adapt.run_cell(seed,q);adapt.ROLL.clear();jax.clear_caches()
    subprocess.run([sys.executable,str(REPO/'scripts/finalize_spatial_delay_scale.py')],check=True)

if __name__=='__main__':main()
