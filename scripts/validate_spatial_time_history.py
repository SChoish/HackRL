#!/usr/bin/env python3
"""Bounded validation only. This entrypoint cannot launch the 40-job study."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from hackrl import spatial_time_history as h
from hackrl import spatial_time_history_eval as evaluation
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import load_dual_checkpoint, save_dual_checkpoint
from hackrl.spatial_random_delay import exposure
import run_spatial_random_delay as shared

REPO=Path(__file__).resolve().parents[1]
SPEC=json.loads((REPO/'docs/manifests/spatial_time_history_v1.json').read_text())
ROOT=Path(SPEC['validation_root'])
write=shared.write
sha=shared.sha


def capacity(event):
    s=SPEC['storage'];bound=s['checkpoint_bound_bytes']
    remaining=max(0,s['retained_validation_checkpoints']-len(list(ROOT.glob('*/checkpoints/*/metadata.json'))))
    projected=(remaining+s['overlap_checkpoints'])*bound+s['artifact_allowance_bytes']
    reserve=max(s['reserve_min_bytes'],int(np.ceil(s['reserve_fraction']*projected)))
    rec=dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
             chosen_filesystem='/raid/ext_csv',projected_remaining_write_bytes=projected,reserve_bytes=reserve,
             existing_validation_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
             existing_scale_run_bytes=sum(p.stat().st_size for p in Path('/raid/ext_csv/HackRL/runs/spatial_delay_scale_v1').rglob('*') if p.is_file()),
             home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict())
    if rec['raid']['free']<projected+reserve:raise RuntimeError('Insufficient RAID peak plus reserve')
    with (ROOT/'capacity.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')


def provenance():
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit validation code before execution')
    paths=[p for glob in ('src/hackrl/*.py','scripts/*spatial*time*','tests/test_spatial_time_history.py',
                         'docs/manifests/spatial_time_history_v1.json') for p in REPO.glob(glob) if p.is_file()]
    return dict(execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                spec=SPEC,source_file_sha256={str(p.relative_to(REPO)):sha(p) for p in paths})


def finite(tree):
    for x in jax.tree.leaves(tree):
        a=np.asarray(x)
        if a.dtype.kind in 'fc' and not np.isfinite(a).all():raise RuntimeError('Nonfinite smoke output')


def execute(fn,r,l,label):
    # Durable accounting includes diagnostic retries after any interruption.
    with (ROOT/'update_calls.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,collector_slots=32768,event='attempt',utc=time.time()))+'\n')
        f.flush();os.fsync(f.fileno())
    out=fn(r,l);jax.block_until_ready(out)
    return out


def check_networks(model,r,teacher,l,condition):
    maps,numeric,goal=h.runner_inputs(r)
    numeric=numeric.at[:,-1].set(1.).at[:,h.NUMERIC:2*h.NUMERIC].set(numeric[:,:h.NUMERIC])
    variables=dict(params=l.params,batch_stats=l.batch_stats)
    p0,v0=model.apply(r.train_state.params,maps,numeric,goal)
    q0=teacher.apply(variables,maps,numeric,train=False)
    out=[]
    for idx in h.TIME_INDICES:
        changed=numeric.at[:,idx].set(.137)
        p1,v1=model.apply(r.train_state.params,maps,changed,goal)
        q1=teacher.apply(variables,maps,changed,train=False)
        delta=lambda x,y:float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        pd,qd,vd=delta(p0.logits,p1.logits),delta(q0,q1),delta(v0,v1)
        if not h.CONDITIONS[condition][0]:assert pd==0.
        else:assert pd>0.
        if not h.CONDITIONS[condition][1]:
            assert qd==0.
            shared.parent.compare_trees(
                teacher.apply(variables,maps,numeric,train=True,mutable=['batch_stats']),
                teacher.apply(variables,maps,changed,train=True,mutable=['batch_stats']))
        else:assert qd>0.
        assert vd>0.
        out.append(dict(time_coordinate=idx,actor_max_delta=pd,teacher_max_delta=qd,critic_max_delta=vd))
    altered=jax.tree.map(lambda x:x+1.,r.train_state.params['params']['critic_encoder'])
    params={**r.train_state.params,'params':{**r.train_state.params['params'],'critic_encoder':altered}}
    assert delta(p0.logits,model.apply(params,maps,numeric,goal)[0].logits)==0.
    return out


def save(condition,name,r,l,c,p):
    capacity(condition+'_'+name);finite((r,l))
    dest=ROOT/condition/'checkpoints'/name
    payload=dict(condition=condition,config=gc.spatial_wall_pass_gc_config_payload(c),
                 input_contract=SPEC['history'],mask=SPEC['mask'],architecture=SPEC['architecture'])
    if dest.exists():
        meta=json.loads((dest/'metadata.json').read_text())
        assert meta['execution_sha']==p['execution_sha'] and meta['state_sha256']==sha(dest/'state.msgpack')
        assert json.loads((dest/'config.json').read_text())==payload
        rr,ll=load_dual_checkpoint(dest,r,l);shared.parent.compare_trees((r,l),(rr,ll))
    else:
        pending=dest.with_name(name+f'.pending.{os.getpid()}')
        if pending.exists():raise RuntimeError('Preserve incomplete diagnostic checkpoint')
        save_dual_checkpoint(pending,r,l,payload)
        assert (pending/'state.msgpack').stat().st_size<SPEC['storage']['checkpoint_bound_bytes']
        meta=json.loads((pending/'metadata.json').read_text())
        meta.update(execution_sha=p['execution_sha'],state_sha256=sha(pending/'state.msgpack'),science_checkpoint=False,
                    counters=shared.counters(r,l))
        write(pending/'metadata.json',meta);pending.rename(dest)
    return dest


def main():
    assert SPEC['authorization']==dict(validation=True,long_training=False)
    if os.environ.get('HACKRL_DEVICE')!='cuda' or not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Real GPU required')
    ROOT.mkdir(parents=True,exist_ok=True);p=provenance();capacity('startup')
    shared.parent.old._write_or_check(ROOT/'manifest.json',p,'validation provenance')
    results=[];initial_hashes=[]
    for condition in h.CONDITIONS:
        c=gc.SpatialWallPassGCConfig(seed=132,num_updates=512)
        model,r,teacher,l,mb=h.initialize(c,condition)
        initial_hashes.append(hashlib.sha256(serialization.to_bytes(dict(ppo=r.train_state.params,teacher=l.params,stats=l.batch_stats))).hexdigest())
        counts=dict(ppo=sum(x.size for x in jax.tree.leaves(r.train_state.params)),teacher=sum(x.size for x in jax.tree.leaves(l.params)))
        mask_initial=check_networks(model,r,teacher,l,condition)
        fn=jax.jit(h.make_update(model,teacher,c,mb))
        before=shared.counters(r,l)
        r1,l1,m1=execute(fn,r,l,condition+'_normal1');finite((r1,l1,m1))
        assert int(r1.train_state.step-r.train_state.step)==32
        assert int(l1.step-l.step)==128
        cp=save(condition,'normal_before',r1,l1,c,p)
        restored,rl=load_dual_checkpoint(cp,r1,l1)
        r2,l2,m2=execute(fn,r1,l1,condition+'_normal2')
        resumed=execute(fn,restored,rl,condition+'_normal2_restored')
        normal_difference=shared.parent.compare_trees((r2,l2,m2),resumed)
        save(condition,'normal_after',r2,l2,c,p)
        cc=replace(c,goal_mode='delivery',variant='mutant',num_updates=4096)
        a=h.adaptation_start(r2.replace(global_update=jnp.int32(512)),132)
        assert not np.asarray(a.history.valid).any()
        cp=save(condition,'adapt_before',a,l2,cc,p)
        ar,al=load_dual_checkpoint(cp,a,l2)
        af=jax.jit(h.make_update(model,teacher,cc,mb,adaptation=True))
        out,leo,metrics=execute(af,a,l2,condition+'_adapt')
        resumed=execute(af,ar,al,condition+'_adapt_restored')
        adapt_difference=shared.parent.compare_trees((out,leo,metrics),resumed)
        assert exposure(out)['collector_slots']==32768 and int(out.global_update)==513
        assert int(out.train_state.step-a.train_state.step)==int(metrics['ppo_applied_minibatches'])
        assert int(leo.step-l2.step)==int(metrics['teacher_applied_minibatches'])
        save(condition,'adapt_after',out,leo,cc,p)
        mask_updated=check_networks(model,out,teacher,leo,condition)
        before_eval=shared.parent.digest(out,leo)
        evaluations=evaluation.evaluate(model,out.train_state.params,132)
        assert before_eval==shared.parent.digest(out,leo)
        episodes=sum(len(v['episodes']) for mode in evaluations.values() for v in mode.values())
        assert episodes==4752
        write(ROOT/condition/'evaluation.json',dict(diagnostic_only=True,results=evaluations))
        record=dict(condition=condition,status='passed',parameter_counts=counts,
                    masks_initial=mask_initial,masks_after_updates=mask_updated,
                    normal_resume_max_abs=normal_difference,adapt_resume_max_abs=adapt_difference,
                    initial_counters=before,normal_counters=shared.counters(r2,l2),adapt_counters=shared.counters(out,leo),
                    adaptation_exposure=exposure(out),diagnostic_evaluation_episodes=episodes,
                    evaluation_sha256=sha(ROOT/condition/'evaluation.json'),evaluation_state_immutable=True)
        results.append(record);write(ROOT/condition/'validation.json',record)
        print(json.dumps(record),flush=True);jax.clear_caches()
    assert len(set(initial_hashes))==1
    assert len({json.dumps(r['parameter_counts'],sort_keys=True) for r in results})==1
    calls=len((ROOT/'update_calls.jsonl').read_text().splitlines())
    write(ROOT/'result.json',dict(status='passed',execution_sha=p['execution_sha'],provenance=p,
          completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),conditions=results,
          initial_parameter_and_stats_hashes=initial_hashes,diagnostic_update_calls=20,
          diagnostic_collector_slots=655360,diagnostic_evaluation_episodes=19008,
          attempted_update_calls_including_retries=calls,attempted_collector_slots_upper_bound=calls*32768,
          scientific_training_started=False,long_training_authorized=False))
    capacity('complete')
    write(ROOT/'verified_complete.json',dict(execution_sha=p['execution_sha'],result_sha256=sha(ROOT/'result.json'),long_training_authorized=False))
    print('Validation complete. Production training remains closed.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if ROOT.exists():write(ROOT/'failed.json',dict(error=repr(error),long_training_authorized=False))
        raise
