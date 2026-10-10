#!/usr/bin/env python3
"""Bounded, full-batch architecture validation. No main-training entrypoint."""
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
from hackrl import spatial_architecture as a
from hackrl import spatial_time_history_eval as evaluation
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import load_dual_checkpoint, save_dual_checkpoint
from hackrl.spatial_random_delay import exposure
import run_spatial_random_delay as shared

REPO=Path(__file__).resolve().parents[1]
SPEC=json.loads((REPO/'docs/manifests/spatial_architecture_v1.json').read_text())
ROOT=Path(SPEC['validation_root'])
write=shared.write
sha=shared.sha


def tree_digest(tree):
    return hashlib.sha256(serialization.to_bytes(tree)).hexdigest()


def capacity(event):
    s=SPEC['storage']
    retained=len(list(ROOT.glob('*/checkpoints/*/metadata.json')))
    projected=(max(0,s['retained_validation_checkpoints']-retained)+s['overlap_checkpoints'])*s['checkpoint_bound_bytes']+s['artifact_allowance_bytes']
    reserve=max(s['reserve_min_bytes'],int(np.ceil(s['reserve_fraction']*projected)))
    record=dict(event=event,utc=time.time(),chosen_filesystem='/raid/ext_csv',projected_remaining_write_bytes=projected,
        reserve_bytes=reserve,home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        existing_validation_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        existing_scale_bytes=sum(p.stat().st_size for p in Path('/raid/ext_csv/HackRL/runs/spatial_delay_scale_v1').rglob('*') if p.is_file()))
    if record['raid']['free']<projected+reserve:raise RuntimeError('RAID peak plus reserve unavailable')
    with (ROOT/'capacity.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')


def provenance():
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit code before GPU validation')
    paths=[p for glob in ('src/hackrl/*.py','scripts/*.py','scripts/*architecture*.sh',
                         'tests/test_spatial_architecture.py','docs/manifests/spatial_architecture_v1.json')
           for p in REPO.glob(glob) if p.is_file()]
    return dict(execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                spec=SPEC,source_file_sha256={str(p.relative_to(REPO)):sha(p) for p in paths})


def finite(tree):
    for x in jax.tree.leaves(tree):
        x=np.asarray(x)
        if x.dtype.kind in 'fc' and not np.isfinite(x).all():raise RuntimeError('Nonfinite result')


def execute(fn,r,l,label):
    with (ROOT/'update_calls.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,event='attempt',collector_slots=32768,utc=time.time()))+'\n')
        f.flush();os.fsync(f.fileno())
    start=time.monotonic();result=fn(r,l);jax.block_until_ready(result)
    seconds=time.monotonic()-start;finite(result)
    with (ROOT/'timing.jsonl').open('a') as f:
        f.write(json.dumps(dict(label=label,seconds=seconds,collector_slots_per_second=32768/seconds))+'\n')
    return result,seconds


def save(arch,name,r,l,c,p):
    capacity(arch+'_'+name);finite((r,l))
    dest=ROOT/arch/'checkpoints'/name
    payload=dict(architecture=arch,config=gc.spatial_wall_pass_gc_config_payload(c),
                 input_contract=SPEC['history'],actor_spec=SPEC['architecture'],auxiliaries=SPEC['auxiliaries'])
    if dest.exists():
        meta=json.loads((dest/'metadata.json').read_text())
        assert meta['execution_sha']==p['execution_sha'] and meta['state_sha256']==sha(dest/'state.msgpack')
        assert json.loads((dest/'config.json').read_text())==payload
        rr,ll=load_dual_checkpoint(dest,r,l);shared.parent.compare_trees((r,l),(rr,ll))
    else:
        pending=dest.with_name(name+f'.pending.{os.getpid()}')
        if pending.exists():raise RuntimeError('Preserve incomplete checkpoint')
        save_dual_checkpoint(pending,r,l,payload)
        assert (pending/'state.msgpack').stat().st_size<SPEC['storage']['checkpoint_bound_bytes']
        meta=json.loads((pending/'metadata.json').read_text())
        meta.update(execution_sha=p['execution_sha'],state_sha256=sha(pending/'state.msgpack'),
                    science_checkpoint=False,counters=shared.counters(r,l))
        write(pending/'metadata.json',meta);pending.rename(dest)
    return dest


def main():
    assert SPEC['authorization']==dict(validation=True,long_training=False)
    if os.environ.get('HACKRL_DEVICE')!='cuda' or not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Real GPU required')
    ROOT.mkdir(parents=True,exist_ok=True);p=provenance();capacity('startup')
    shared.parent.old._write_or_check(ROOT/'manifest.json',p,'architecture validation provenance')
    results=[];auxiliary_hashes=[]
    for arch in a.ARCHITECTURES:
        print('initialize '+arch,flush=True)
        c=gc.SpatialWallPassGCConfig(seed=132,num_updates=512)
        model,r,teacher,l,mb=a.initialize(c,arch)
        pp=r.train_state.params['params']
        auxiliary_hashes.append(tree_digest(dict(critic={k:v for k,v in pp.items() if k!='actor'},teacher=l.params,stats=l.batch_stats)))
        count=lambda x:sum(v.size for v in jax.tree.leaves(x))
        counts=dict(actor=count(pp['actor']),critic=count({k:v for k,v in pp.items() if k!='actor'}),teacher=count(l.params))
        assert counts['actor']==SPEC['parameter_matching']['actor_expected_from_shapes'][arch]
        print(json.dumps(dict(architecture=arch,parameter_counts=counts)),flush=True)
        fn=jax.jit(a.make_update(model,teacher,c,mb))
        (r1,l1,m1),normal_compile=execute(fn,r,l,arch+'_normal1')
        assert int(r1.train_state.step-r.train_state.step)==32
        assert int(l1.step-l.step)==128
        assert tree_digest(pp['actor'])!=tree_digest(r1.train_state.params['params']['actor'])
        cp=save(arch,'normal_before',r1,l1,c,p)
        restored,rl=load_dual_checkpoint(cp,r1,l1)
        (r2,l2,m2),normal_steady=execute(fn,r1,l1,arch+'_normal2')
        resumed,_=execute(fn,restored,rl,arch+'_normal2_restored')
        normal_difference=shared.parent.compare_trees((r2,l2,m2),resumed)
        normal_cp=save(arch,'normal_after',r2,l2,c,p)
        fork_hash=tree_digest((r2,l2));branches=[]
        cc=replace(c,goal_mode='delivery',variant='mutant',num_updates=4096)
        for probability in SPEC['fixed_episode_probabilities']:
            assert tree_digest((r2,l2))==fork_hash
            label=f'q{int(probability*100)}'
            start=a.adaptation_start(r2.replace(global_update=jnp.int32(512)),132,probability)
            assert not np.asarray(start.history.valid).any()
            shared.parent.compare_trees(start.train_state,r2.train_state)
            cp=save(arch,label+'_before',start,l2,cc,p)
            restored,rl=load_dual_checkpoint(cp,start,l2)
            af=jax.jit(a.make_update(model,teacher,cc,mb,adaptation=True,probability=probability))
            (out,leo,metrics),compile_seconds=execute(af,start,l2,arch+'_'+label)
            resumed,restored_seconds=execute(af,restored,rl,arch+'_'+label+'_restored')
            difference=shared.parent.compare_trees((out,leo,metrics),resumed)
            assert exposure(out)['collector_slots']==32768 and int(out.global_update)==513
            assert int(out.train_state.step-start.train_state.step)==int(metrics['ppo_applied_minibatches'])
            assert int(leo.step-l2.step)==int(metrics['teacher_applied_minibatches'])
            save(arch,label+'_after',out,leo,cc,p)
            before_eval=shared.parent.digest(out,leo)
            evaluations=evaluation.evaluate(model,out.train_state.params,132,history_api=a)
            assert before_eval==shared.parent.digest(out,leo)
            episodes=sum(len(v['episodes']) for mode in evaluations.values() for v in mode.values())
            assert episodes==4752
            evaluation_path=ROOT/arch/(label+'_evaluation.json')
            write(evaluation_path,dict(diagnostic_only=True,results=evaluations))
            branch=dict(fixed_episode_probability=probability,normal_checkpoint_sha256=sha(normal_cp/'state.msgpack'),
                parent_full_state_hash=fork_hash,resume_max_abs=difference,exposure=exposure(out),counters=shared.counters(out,leo),
                first_update_including_compile_seconds=compile_seconds,
                restored_update_seconds_including_possible_recompile=restored_seconds,
                diagnostic_evaluation_episodes=episodes,evaluation_state_immutable=True,evaluation_sha256=sha(evaluation_path))
            branches.append(branch);print(json.dumps(dict(architecture=arch,branch=branch)),flush=True)
        record=dict(architecture=arch,status='passed',parameter_counts=counts,normal_resume_max_abs=normal_difference,
            normal_first_update_including_compile_seconds=normal_compile,normal_steady_update_seconds=normal_steady,branches=branches)
        results.append(record);write(ROOT/arch/'validation.json',record)
        jax.clear_caches()
    assert len(set(auxiliary_hashes))==1
    actor_counts=[r['parameter_counts']['actor'] for r in results]
    assert max(actor_counts)/min(actor_counts)<SPEC['parameter_matching']['maximum_allowed_max_min_ratio']
    calls=len((ROOT/'update_calls.jsonl').read_text().splitlines())
    write(ROOT/'result.json',dict(status='passed',execution_sha=p['execution_sha'],provenance=p,architectures=results,
        identical_initial_auxiliary_hashes=auxiliary_hashes,actor_parameter_max_min_ratio=max(actor_counts)/min(actor_counts),
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),diagnostic_update_calls=28,
        diagnostic_collector_slots=917504,diagnostic_evaluation_episodes=38016,
        attempted_calls_including_retries=calls,attempted_collector_slots_upper_bound=calls*32768,
        scientific_training_started=False,long_training_authorized=False))
    capacity('complete')
    write(ROOT/'verified_complete.json',dict(execution_sha=p['execution_sha'],result_sha256=sha(ROOT/'result.json'),long_training_authorized=False))
    print('All four architecture validations passed. Main training remains closed.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if ROOT.exists():write(ROOT/'failed.json',dict(error=repr(error),long_training_authorized=False))
        raise
