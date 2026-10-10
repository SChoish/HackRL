#!/usr/bin/env python3
"""Approved 60-job queue; immutable budgets, complete-state forks and restart."""
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
import jax
import numpy as np
from hackrl import spatial_architecture as a
from hackrl import spatial_time_history_eval as evaluation
from hackrl import spatial_wall_pass_gc as gc
from hackrl.dual_leo import save_dual_checkpoint, load_dual_checkpoint
from hackrl.spatial_random_delay import exposure
import run_spatial_random_delay as shared

REPO=Path(__file__).resolve().parents[1]
SPEC=json.loads((REPO/'docs/manifests/spatial_architecture_v1.json').read_text())
ROOT=Path(SPEC['run_root'])
sha=shared.sha
write=shared.write
STOP=False


def journal(cell,record):
    """Durable attempt accounting is separate from checkpoint-restored progress."""
    with (cell/'attempts.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())


def stop_requested(signum, frame):
    global STOP
    STOP=True


def jobs():
    for seed in SPEC['seeds']:
        for arch in SPEC['architectures']:
            yield arch,seed,None
            for q in SPEC['fixed_episode_probabilities']:yield arch,seed,q


def cell_path(arch,seed,q):
    phase='pretraining' if q is None else f'fixed_{round(100*q):02d}pct'
    return ROOT/arch/f'seed_{seed}'/phase


def config(seed,q):
    c=gc.SpatialWallPassGCConfig(seed=seed,num_updates=SPEC['pretrain_updates'])
    if q is not None:c=replace(c,goal_mode='delivery',variant='mutant',num_updates=SPEC['adapt_updates'])
    return c


def checkpoint_payload(arch,c,q):
    return dict(architecture=arch,fixed_episode_probability=q,config=gc.spatial_wall_pass_gc_config_payload(c),
                history=SPEC['history'],actor_spec=SPEC['architecture'],auxiliaries=SPEC['auxiliaries'])


@lru_cache(None)
def provenance():
    assert SPEC['authorization']['long_training'] is True
    assert len(list(jobs()))==SPEC['planned_jobs']==60
    assert 4*5*(512+2*4096)*512*64==SPEC['planned_collector_slots']
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit execution code before launch')
    validation=Path(SPEC['validation_root'])
    receipt=json.loads((validation/'verified_complete.json').read_text())
    result=json.loads((validation/'result.json').read_text())
    assert receipt['execution_sha']==SPEC['validation_evidence']['execution_sha']
    assert sha(validation/'result.json')==receipt['result_sha256']==SPEC['validation_evidence']['result_sha256']
    assert result['status']=='passed' and len(result['architectures'])==4
    # The scientific model, learner, history and evaluation code must be exactly GPU-validated.
    for path,digest in result['provenance']['source_file_sha256'].items():
        if path.startswith('src/'):assert sha(REPO/path)==digest,path
    paths=[p for group in ('src/hackrl/*.py','scripts/*.py','scripts/*architecture*.sh',
                          'docs/manifests/spatial_architecture_v1.json') for p in REPO.glob(group)]
    return dict(execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        spec=SPEC,validation_result_sha256=receipt['result_sha256'],
        source_file_sha256={str(p.relative_to(REPO)):sha(p) for p in paths})


def capacity(event):
    s=SPEC['production_storage']
    present=len(list(ROOT.glob('*/seed_*/*/checkpoints/update_*/metadata.json')))
    projected=(max(0,s['planned_retained_checkpoints']+s['interruption_checkpoint_allowance']-present)+s['overlap_checkpoints'])*s['checkpoint_bound_bytes']+s['artifact_allowance_bytes']
    reserve=max(s['reserve_min_bytes'],int(np.ceil(s['reserve_fraction']*projected)))
    record=dict(event=event,utc=time.time(),chosen_filesystem='/raid/ext_csv',projected_remaining_write_bytes=projected,
        safety_reserve_bytes=reserve,existing_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        existing_validation_bytes=sum(p.stat().st_size for p in Path(SPEC['validation_root']).rglob('*') if p.is_file()),
        home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict())
    with (ROOT/'capacity.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    if record['raid']['free']<projected+reserve:raise RuntimeError('RAID peak plus reserve unavailable; safely stop without deleting checkpoints')


def finite(tree):
    for x in jax.tree.leaves(tree):
        x=np.asarray(x)
        if x.dtype.kind in 'fc' and not np.isfinite(x).all():raise RuntimeError('Nonfinite learner state or metrics')


def validate_checkpoint(path,arch,c,q):
    meta=json.loads((path/'metadata.json').read_text())
    assert meta['execution_sha']==provenance()['execution_sha']
    assert meta['state_sha256']==sha(path/'state.msgpack')
    assert json.loads((path/'config.json').read_text())==checkpoint_payload(arch,c,q)
    if (path/'evaluation_integrity.json').exists():
        assert json.loads((path/'evaluation_integrity.json').read_text())['sha256']==sha(path/'evaluation.json')
    return meta


def complete_checkpoints(cps,arch,c,q):
    result={}
    for path in cps.glob('update_*'):
        suffix=path.name.removeprefix('update_')
        if not suffix.isdigit():continue # partial pending directories are preserved, never resumed
        meta=validate_checkpoint(path,arch,c,q)
        step=int(suffix)
        assert 0<=step<=c.num_updates
        assert meta['global_update']==step+(0 if q is None else SPEC['pretrain_updates'])
        result[step]=path
    return result


def save(path,arch,c,q,r,l,source_hash,accounting):
    capacity(str(path));finite((r,l))
    if path.exists():raise RuntimeError('Refuse checkpoint overwrite')
    pending=path.with_name(path.name+'.pending.'+uuid.uuid4().hex)
    save_dual_checkpoint(pending,r,l,checkpoint_payload(arch,c,q))
    assert (pending/'state.msgpack').stat().st_size<SPEC['production_storage']['checkpoint_bound_bytes']
    meta=json.loads((pending/'metadata.json').read_text())
    meta.update(execution_sha=provenance()['execution_sha'],state_sha256=sha(pending/'state.msgpack'),
                parent_pretrain_sha256=source_hash,counters=shared.counters(r,l),
                exposure=None if q is None else exposure(r),collection_accounting=accounting)
    write(pending/'metadata.json',meta);pending.rename(path)
    write(path.parent.parent/'latest.json',dict(checkpoint=str(path),global_update=int(r.global_update)))


class Evaluator:
    """Cache compiled code only. Every invocation resets evaluation state and RNG."""
    def __init__(self,model):
        self.rollouts={s:evaluation.make_rollout(model,s=='sample',n,history_api=a)
                       for s,n in (('mode',1),('sample',8))}

    def __call__(self,params,seed):
        result={}
        for selection,repeats in (('mode',1),('sample',8)):
            result[selection]={}
            for delay in SPEC['evaluation_waits']:
                for kernel in ('fixed','mutant','mutant_block_once'):
                    trace=jax.device_get(self.rollouts[selection](params,np.int32(seed),kernel=='fixed',
                        np.int32(delay),kernel=='mutant_block_once'))
                    records=evaluation.records_from_trace(trace,repeats,delay)
                    if selection=='sample':
                        for rec in records:rec.pop('path')
                    result[selection][f'{kernel}_delay{delay}']=dict(summary=evaluation.summarize(records),episodes=records)
        return result


def run_cell(arch,seed,q):
    cell=cell_path(arch,seed,q);c=config(seed,q)
    cpdir=cell/'checkpoints';cpdir.mkdir(parents=True,exist_ok=True)
    model,r,teacher,l,mb=a.initialize(c,arch)
    source_hash=None
    if q is not None:
        source=cell_path(arch,seed,None)/'checkpoints'/f"update_{SPEC['pretrain_updates']}"
        validate_checkpoint(source,arch,config(seed,None),None)
        r,l=load_dual_checkpoint(source,r,l)
        source_hash=sha(source/'state.msgpack')
        assert int(r.global_update)==SPEC['pretrain_updates']
        r=a.adaptation_start(r,seed,q)
    origin=dict(architecture=arch,seed=seed,q=q,parent_pretrain_sha256=source_hash,
        initial_state_sha256=shared.parent.digest(r,l),initial_counters=shared.counters(r,l),
        execution_sha=provenance()['execution_sha'])
    shared.parent.old._write_or_check(cell/'origin.json',origin,'architecture job origin')
    existing=complete_checkpoints(cpdir,arch,c,q)
    done=max(existing,default=0)
    if existing:r,l=load_dual_checkpoint(existing[done],r,l)
    valid_samples=0 if not existing else json.loads((existing[done]/'metadata.json').read_text())['collection_accounting']['valid_learner_samples']
    assert int(r.global_update)==done+(0 if q is None else SPEC['pretrain_updates'])
    schedule=SPEC['evaluation_updates']['pretrain' if q is None else 'adaptation']
    evaluator=Evaluator(model)
    def accounting(u):
        if q is not None:
            result=exposure(r)
            assert result['valid_learner_samples']==valid_samples
            assert result['collector_slots']==u*c.batch_size
            return {k:result[k] for k in ('collector_slots','valid_learner_samples','forced_wait_transitions','physical_environment_transitions')}
        return dict(collector_slots=u*c.batch_size,valid_learner_samples=valid_samples,
                    forced_wait_transitions=0,physical_environment_transitions=u*c.batch_size)
    def checkpoint(u,evaluate=True):
        path=cpdir/f'update_{u}'
        if not path.exists():save(path,arch,c,q,r,l,source_hash,accounting(u))
        if evaluate:
            ep=path/'evaluation.json'
            if not ep.exists():
                capacity('evaluation_'+str(path))
                before=shared.parent.digest(r,l)
                result=evaluator(r.train_state.params,seed)
                assert before==shared.parent.digest(r,l)
                write(ep,dict(update=u,results=result,counters=shared.counters(r,l),
                              exposure=None if q is None else exposure(r)))
            if (path/'evaluation_integrity.json').exists():
                assert json.loads((path/'evaluation_integrity.json').read_text())['sha256']==sha(ep)
            else:write(path/'evaluation_integrity.json',dict(sha256=sha(ep)))
        write(cell/'latest.json',dict(checkpoint=str(path),global_update=int(r.global_update)))
    checkpoint(done,evaluate=done in schedule)
    fn=jax.jit(a.make_update(model,teacher,c,mb,adaptation=q is not None,probability=0. if q is None else q))
    for u in range(done+1,c.num_updates+1):
        if STOP:
            checkpoint(u-1,evaluate=(u-1) in schedule)
            write(ROOT/'interrupted.json',dict(architecture=arch,seed=seed,q=q,update=u-1,utc=time.time()))
            return False
        attempt=uuid.uuid4().hex
        journal(cell,dict(event='attempt',id=attempt,update=u,collector_slots=c.batch_size,utc=time.time()))
        previous_forced=0 if q is None else exposure(r)['forced_wait_transitions']
        start=time.monotonic();previous_ppo=int(r.train_state.step);previous_teacher=int(l.step)
        r,l,m=fn(r,l);jax.block_until_ready((r,l,m));seconds=time.monotonic()-start
        finite(m)
        valid_samples+=int(m['ppo_valid_transitions'])
        assert int(r.train_state.step)-previous_ppo==int(m['ppo_applied_minibatches'])
        assert int(l.step)-previous_teacher==int(m['teacher_applied_minibatches'])
        forced_delta=0 if q is None else exposure(r)['forced_wait_transitions']-previous_forced
        journal(cell,dict(event='completed',id=attempt,update=u,collector_slots=c.batch_size,
            valid_learner_samples=int(m['ppo_valid_transitions']),forced_wait_transitions=forced_delta,
            physical_environment_transitions=c.batch_size+forced_delta,seconds=seconds,utc=time.time()))
        if u%32==0 or u==done+1:
            record=dict(event='update',architecture=arch,seed=seed,q=q,update=u,utc=time.time(),seconds=seconds,
                first_call_after_start_or_resume=(u==done+1),counters=shared.counters(r,l),
                exposure=None if q is None else exposure(r),metrics={k:np.asarray(v).tolist() for k,v in m.items()})
            record['collection_accounting']=accounting(u)
            with (cell/'updates.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if u%64==0:capacity(f'{arch}_{seed}_{q}_{u}')
        if u in schedule:checkpoint(u)
    final_path=cpdir/f'update_{c.num_updates}'
    final=json.loads((final_path/'evaluation.json').read_text())['results']
    mode=final['mode']['fixed_delay0']['summary']['success_rate']
    sample=final['sample']['fixed_delay0']['summary']['success_rate']
    summary=dict(status='complete',architecture=arch,seed=seed,q=q,updates=c.num_updates,
        collector_slots=c.num_updates*c.batch_size,final_checkpoint=str(final_path),final_state_sha256=sha(final_path/'state.msgpack'),
        parent_pretrain_sha256=source_hash,counters=shared.counters(r,l),
        normal_capability_flag=(mode>=.9 and sample>=.9) if q is None else None,
        fixed_natural_mode_success=mode,fixed_natural_sample_success=sample,
        exposure=None if q is None else exposure(r),collection_accounting=accounting(c.num_updates))
    write(cell/'summary.json',summary);print(json.dumps(dict(event='job_complete',**summary)),flush=True)
    return True


def finalize():
    summaries=[]
    for arch,seed,q in jobs():
        cell=cell_path(arch,seed,q);c=config(seed,q)
        summary=json.loads((cell/'summary.json').read_text())
        assert summary['status']=='complete' and summary['updates']==c.num_updates
        assert summary['collector_slots']==c.num_updates*c.batch_size
        parent_hash=None if q is None else sha(cell_path(arch,seed,None)/'checkpoints'/f"update_{SPEC['pretrain_updates']}"/'state.msgpack')
        for u in SPEC['evaluation_updates']['pretrain' if q is None else 'adaptation']:
            path=cell/'checkpoints'/f'update_{u}'
            meta=validate_checkpoint(path,arch,c,q)
            assert meta['parent_pretrain_sha256']==parent_hash
            assert json.loads((path/'evaluation_integrity.json').read_text())['sha256']==sha(path/'evaluation.json')
        assert summary['final_state_sha256']==sha(Path(summary['final_checkpoint'])/'state.msgpack')
        assert summary['parent_pretrain_sha256']==parent_hash
        summaries.append(summary)
    assert len(summaries)==60 and sum(x['collector_slots'] for x in summaries)==SPEC['planned_collector_slots']
    write(ROOT/'result.json',dict(status='complete',execution_sha=provenance()['execution_sha'],jobs=summaries,
        planned_collector_slots=SPEC['planned_collector_slots'],completed_utc=time.time(),extension=False))
    write(ROOT/'verified_complete.json',dict(execution_sha=provenance()['execution_sha'],jobs=60,result_sha256=sha(ROOT/'result.json')))


def main():
    if os.environ.get('HACKRL_DEVICE')!='cuda' or not any(d.platform=='gpu' for d in jax.devices()):
        raise RuntimeError('Real GPU required')
    ROOT.mkdir(parents=True,exist_ok=True)
    p=provenance();capacity('startup')
    shared.parent.old._write_or_check(ROOT/'run_manifest.json',p,'architecture production provenance')
    signal.signal(signal.SIGTERM,stop_requested);signal.signal(signal.SIGINT,stop_requested)
    for arch,seed,q in jobs():
        if STOP or not run_cell(arch,seed,q):return
        jax.clear_caches()
    finalize()


if __name__=='__main__':
    try:main()
    except Exception as error:
        if ROOT.exists():write(ROOT/'failed.json',dict(error=repr(error),utc=time.time(),automatic_retry=False))
        raise
