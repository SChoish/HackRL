#!/usr/bin/env python3
"""Fixed-budget fresh-seed teacher comparison; no intermediate holdout queries."""
import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault('JAX_PLATFORMS','cuda' if os.environ.get('HACKRL_DEVICE')=='cuda' else 'cpu')
import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
import run_dual_leo_compare as base
from hackrl.dual_leo import load_dual_checkpoint
from hackrl.pack_teacher_holdout import generate_panel, deserialize_state, digest, make_evaluator

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/pack_teacher_holdout_v2.json'
SPEC=json.loads(SPEC_PATH.read_text())
ROOT=Path(SPEC['run_root'])
SEEDS=SPEC['seeds']
FILES=['docs/manifests/pack_teacher_holdout_v2.json','scripts/run_pack_teacher_holdout_v2.py',
       'scripts/validate_pack_teacher_holdout_v2.py','tests/test_pack_teacher_holdout.py',
       'scripts/run_pack_teacher_holdout_v2_gpu.sh','scripts/run_pack_teacher_holdout_v2_cpu.sh',
       'src/hackrl/pack_teacher_holdout.py','scripts/run_dual_leo_compare.py',
       'src/hackrl/pack_restore_gc.py','src/hackrl/pack_restore.py','src/hackrl/dual_leo.py',
       'src/hackrl/batch_renorm.py','src/hackrl/tick_claim.py','src/hackrl/tick_claim_gc.py',
       'src/hackrl/tick_claim_oracle.py']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')


def capacity():
    ROOT.mkdir(parents=True,exist_ok=True)
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    home=shutil.disk_usage('/home/ext_csv');raid=shutil.disk_usage('/raid/ext_csv')
    info={'unix':time.time(),'home':home._asdict(),'raid':raid._asdict(),'existing_run_bytes':used,
          'projected_additional_peak_bytes':SPEC['storage']['projected_peak_bytes'],
          'reserve_bytes':SPEC['storage']['reserve_bytes'],'chosen_filesystem':'/raid/ext_csv'}
    append(ROOT/'capacity.jsonl',info)
    if raid.free < info['projected_additional_peak_bytes']+info['reserve_bytes']:
        raise RuntimeError('RAID reserve exhausted; pause without deleting checkpoints')
    return info


def source(seed):return ROOT/f'source/seed{seed}/checkpoints/update_512'


def jobs():
    rows=[]
    for seed in SEEDS:
        common={'env':'pack','size':'S','seed':seed,'policy_hidden_size':512,'teacher_hidden_size':512,
                'learn_teacher':True,'imitate_teacher':True,'method':'dual'}
        rows.append(common|{'id':f'source-s{seed}','condition':'source','kind':'pretrain','variant':'fixed'})
        for condition,indices in SPEC['conditions'].items():
            row=common|{'id':f'{condition}-s{seed}','condition':condition,'kind':'adapt','variant':'mutant',
                        'pretrain_checkpoint':str(source(seed)),
                        'origin_arm':{'learn_teacher':True,'imitate_teacher':True}}
            if indices is not None:row['teacher_goal_indices']=indices
            rows.append(row)
    return rows


def cell(job):return ROOT/job['condition']/f"seed{job['seed']}"


def seed_audit():
    used=set();scanned=[]
    def inspect(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('seed','learner_seed') and isinstance(v,int):used.add(v)
                elif k in ('seeds','learner_seeds') and isinstance(v,list):used.update(x for x in v if isinstance(x,int))
                elif isinstance(v,(dict,list)):inspect(v)
        elif isinstance(value,list):
            for x in value:inspect(x)
    for folder,dirs,files in os.walk('/raid/ext_csv/HackRL/runs'):
        dirs[:]=[d for d in dirs if Path(folder)/d!=ROOT and d not in ('.git','updates','evaluations','curve')]
        for name in dirs:
            match=re.fullmatch(r'seed_?(\d+)',name)
            if match:used.add(int(match[1]))
        for name in files:
            if name not in ('config.json','manifest.json','run_manifest.json','jobs.json'):continue
            p=Path(folder)/name
            try:data=json.loads(p.read_text())
            except (OSError,ValueError):continue
            inspect(data);scanned.append(str(p))
    for directory in [REPO.parent.parent/'docs/manifests',*(REPO.parent.glob('*/docs/manifests'))]:
        for p in directory.glob('*.json'):
            if p.name==SPEC_PATH.name:continue
            try:inspect(json.loads(p.read_text()))
            except (ValueError,OSError):continue
            scanned.append(str(p))
    assert not set(SEEDS)&used, f'seed collision {set(SEEDS)&used}'
    return {'selected':SEEDS,'observed_seeds':sorted(used),'files_inspected':scanned,
            'selection_before_policy_query':True,'limitations':'Available mounted run history and repository manifests; inaccessible/deleted external runs cannot be certified.'}


def historical_contract():
    root=Path('/raid/ext_csv/HackRL/runs/pack_restore_delivery_only_teacher_v1')
    result=json.loads((root/'result.json').read_text())
    assert result['execution_complete'] and not result['errors'] and not result['missing_curves']
    configs=[];timings=[]
    for p in sorted(root.glob('size_s/pack/dual_teacher_delivery_only/*/seed*/checkpoints/adapt_4096')):
        cfg=json.loads((p/'config.json').read_text());arm=json.loads((p/'arm.json').read_text())
        meta=json.loads((p/'metadata.json').read_text())
        assert arm=={'learn_teacher':True,'imitate_teacher':True,'teacher_goal_indices':[11]}
        assert meta['global_update']==4608 and cfg['goal_mode']=='deliver_3'
        assert cfg['num_envs']==512 and cfg['num_steps']==64
        assert cfg['policy_hidden_size']==512 and cfg['teacher_hidden_size']==512
        planned=asdict(base._job_config(jobs()[2],goal_mode='deliver_3',variant=cfg['variant'],updates=4096))
        # Evaluation draw counts are deliberately replaced by the locked V2 panel.
        ignored={'seed','checkpoint_updates','sample_repeats_per_state','mode_repeats_per_state','evaluation_split'}
        for key,value in cfg.items():
            if key not in ignored:assert planned[key]==value,f'legacy recipe differs: {key}'
        summary=json.loads((p.parent.parent/'summary.json').read_text())
        timings.append(summary['seconds_per_update_this_process'])
        configs.append({'path':str(p),'state_sha256':sha(p/'state.msgpack'),'config_sha256':sha(p/'config.json')})
    assert len(configs)==20
    on=Path('/raid/ext_csv/HackRL/runs/pack_restore_scale_bc_v1/size_s/pack/dual/mutant/seed30/checkpoints/adapt_4096')
    on_cfg=json.loads((on/'config.json').read_text())
    for key,value in on_cfg.items():
        if key not in ignored:assert planned[key]==value,f'D-on recipe differs: {key}'
    assert json.loads((on/'arm.json').read_text())=={'learn_teacher':True,'imitate_teacher':True}
    seconds=SPEC['budget']['total_slots']/32768*statistics.median(timings)
    return {'result_sha256':sha(root/'result.json'),'finals':configs,
            'D_on_reference_fingerprint':base._checkpoint_fingerprint(on),
            'historical_seconds_per_update_median':statistics.median(timings),
            'historical_compute_seconds_projection':seconds,
            'eta_limit':'Historical device/load and eval overhead differ; not a current-machine guarantee.'}


def prepare():
    capacity()
    if (ROOT/'run_manifest.json').exists():
        verify();return
    audit=seed_audit();write(ROOT/'seed_audit.json',audit)
    panel=generate_panel();write(ROOT/'holdout_v2.json',panel)
    history=historical_contract();write(ROOT/'historical_contract.json',history)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--',*FILES],cwd=REPO,text=True)
    assert not dirty, 'commit execution sources before prepare'
    document={'experiment_id':SPEC['experiment_id'],'execution_head':head,
              'source_sha256':{f:sha(REPO/f) for f in FILES},'spec_sha256':sha(SPEC_PATH),
              'panel_sha256':sha(ROOT/'holdout_v2.json'),'seed_audit_sha256':sha(ROOT/'seed_audit.json'),
              'historical_contract_sha256':sha(ROOT/'historical_contract.json'),'jobs':jobs(),
              'created_unix':time.time(),'evaluation_has_started':False}
    write(ROOT/'run_manifest.json',document)


def verify():
    d=json.loads((ROOT/'run_manifest.json').read_text())
    assert d['jobs']==jobs()
    for f,h in d['source_sha256'].items():assert sha(REPO/f)==h,f'changed execution source: {f}'
    for f,key in [('holdout_v2.json','panel_sha256'),('seed_audit.json','seed_audit_sha256'),('historical_contract.json','historical_contract_sha256')]:
        assert sha(ROOT/f)==d[key],f'changed locked artifact: {f}'
    return d


def tree_hash(value):
    # Serialize numerical state only, including optimizers and BatchRenorm.
    return hashlib.sha256(serialization.to_bytes(value)).hexdigest()


def pair_identity(job,runner,leo):
    components={'actor_critic_adam':runner.train_state,'teacher_renorm_adam':leo,
                'environment':runner.env_state,'env_rng':runner.env_keys,'sampler_rng':runner.sampler_keys,
                'runner_rng':runner.rng,'whole_runner':runner}
    hashes={k:tree_hash(v) for k,v in components.items()}
    assert np.all(np.asarray(runner.current_goal)==11)
    record={'seed':job['seed'],'source_sha256':sha(source(job['seed'])/'state.msgpack'),'components':hashes}
    path=ROOT/f"paired_origin_seed{job['seed']}.json"
    if path.exists():assert json.loads(path.read_text())==record,'paired origin mismatch'
    else:write(path,record)
    write(cell(job)/'origin.json',record)


def save(job,runner,leo,config,number):
    capacity();root=cell(job)/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    prefix='update' if job['kind']=='pretrain' else 'adapt'
    dest=root/f'{prefix}_{number}';tmp=root/f'.{prefix}_{number}.tmp-{os.getpid()}'
    assert not dest.exists(),f'checkpoint already exists: {dest}'
    base._save_state(base.ENVS['pack'],'dual',tmp,runner,leo,config,base._arm(job))
    write(tmp/'integrity.json',{'state_sha256':sha(tmp/'state.msgpack'),'config_sha256':sha(tmp/'config.json'),
                              'checkpoint_files':base._checkpoint_fingerprint(tmp)['files'],
                              'global_update':int(runner.global_update)})
    os.replace(tmp,dest)
    # Only completed resumable predecessors inside this NEW run are pruned.
    for p in root.glob(f'{prefix}_*'):
        if p!=dest and p.is_dir():shutil.rmtree(p)
    return dest


def load_checked(path,template,leo_template):
    receipt=json.loads((path/'integrity.json').read_text())
    assert sha(path/'state.msgpack')==receipt['state_sha256']
    assert sha(path/'config.json')==receipt['config_sha256']
    for name,value in receipt.get('checkpoint_files',{}).items():assert sha(path/name)==value,name
    runner,leo=load_dual_checkpoint(path,template,leo_template)
    assert int(runner.global_update)==receipt['global_update']
    return runner,leo


def run_job(job):
    verify();capacity();dest=cell(job);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'failure.json').exists():raise RuntimeError(f"terminal failed job: {job['id']}")
    spec=base.ENVS['pack'];is_source=job['kind']=='pretrain'
    target=SPEC['source_updates'] if is_source else SPEC['adaptation_updates']
    offset=0 if is_source else SPEC['source_updates']
    cfg=base._job_config(job,goal_mode='workshop12' if is_source else 'deliver_3',variant=job['variant'],updates=target)
    network,runner=spec['initialize'](cfg)
    teacher,leo,minibatch=base._start_teacher(spec,cfg,runner)
    prefix='update' if is_source else 'adapt'
    checkpoints=sorted((dest/'checkpoints').glob(f'{prefix}_*'),key=lambda p:int(p.name.split('_')[-1]))
    if checkpoints:
        runner,leo=load_checked(checkpoints[-1],runner,leo)
    elif not is_source:
        # Fingerprint source before loading, and preserve full restoration contract.
        src_receipt=json.loads((source(job['seed'])/'integrity.json').read_text())
        assert sha(source(job['seed'])/'state.msgpack')==src_receipt['state_sha256']
        for name,value in src_receipt['checkpoint_files'].items():assert sha(source(job['seed'])/name)==value
        runner,leo=base._load_adapt_start(ROOT,job,runner,cfg)
        pair_identity(job,runner,leo)
    if not checkpoints:save(job,runner,leo,cfg,0)
    if int(runner.global_update)==offset+target and (dest/'summary.json').exists():return
    update=base._dual_update(spec,network,teacher,cfg,minibatch,base._arm(job))
    finite=jax.jit(lambda a,b:jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((a,b))])))
    start=time.monotonic();first=int(runner.global_update);seconds=[]
    while int(runner.global_update)<offset+target:
        t=time.monotonic();runner,leo,metrics=update(runner,leo)
        jax.block_until_ready(runner.train_state.step)
        host={k:float(np.asarray(v)) for k,v in metrics.items()}
        assert all(np.isfinite(v) for v in host.values()),'nonfinite metric'
        assert bool(finite(runner.train_state,leo)),'nonfinite learner state'
        if not is_source:
            assert np.all(np.asarray(runner.current_goal)==11)
            assert host['teacher_target_head_count']==(12 if job['condition']=='D-on' else 1)
        n=int(runner.global_update)-offset;elapsed=time.monotonic()-t;seconds.append(elapsed)
        append(dest/'updates.jsonl',{'global_update':int(runner.global_update),'update':n,'seconds':elapsed,**host})
        if n%64==0 or n==target:save(job,runner,leo,cfg,n)
        if n%32==0:print(json.dumps({'job':job['id'],'update':n,'target':target,'seconds':elapsed}),flush=True)
    final=dest/'checkpoints'/f'{prefix}_{target}'
    attempts=[json.loads(line) for line in (dest/'updates.jsonl').read_text().splitlines()]
    retained={row['update']:row for row in attempts}
    assert set(retained)==set(range(1,target+1))
    counter_keys=['valid_transitions','teacher_valid_samples','teacher_target_terms',
                  'ppo_applied_minibatches','teacher_applied_minibatches']
    write(dest/'summary.json',{'job':job,'complete':True,'global_update':int(runner.global_update),
          'collector_slots':target*32768,'seconds_this_process':time.monotonic()-start,
          'starting_global_update_this_process':first,'median_seconds_per_update':statistics.median(seconds) if seconds else None,
          'ppo_optimizer_step':int(runner.train_state.step),'teacher_optimizer_step':int(leo.step),
          'retained_trajectory_counters':{k:sum(row[k] for row in retained.values()) for k in counter_keys},
          'physical_attempt_counters':{k:sum(row[k] for row in attempts) for k in counter_keys},
          'completed_update_attempt_slots':len(attempts)*32768,
          'interruption_accounting':'An interrupted in-flight update can consume at most 32768 extra unlogged slots; it cannot advance the retained schedule.',
          'checkpoint_sha256':sha(final/'state.msgpack')})


def evaluate_all():
    verify()
    assert all((cell(j)/'summary.json').exists() for j in jobs())
    panel=json.loads((ROOT/'holdout_v2.json').read_text());spec=base.ENVS['pack'];evaluators={}
    for seed in SEEDS:
        for label in ['source','D-on','D-delivery']:
            job=next(j for j in jobs() if j['seed']==seed and j['condition']==label)
            path=source(seed) if label=='source' else cell(job)/'checkpoints/adapt_4096'
            cfg=spec['from_payload'](json.loads((path/'config.json').read_text()))
            network,template=spec['initialize'](cfg);_,leo_template,_=base._start_teacher(spec,cfg,template)
            runner,leo=load_checked(path,template,leo_template)
            for kernel in ['fixed','mutant']:
                for stochastic,repeats in [(False,1),(True,8)]:
                    mode='sample' if stochastic else 'mode'
                    dest=ROOT/'evaluation'/f'{label}_s{seed}_{kernel}_{mode}.json'
                    if dest.exists():
                        saved=json.loads(dest.read_text());assert saved['checkpoint_sha256']==sha(path/'state.msgpack')
                        assert saved['panel_sha256']==sha(ROOT/'holdout_v2.json') and len(saved['records'])==16*repeats
                        continue
                    states=[deserialize_state(x['state']) for x in panel['states'] for _ in range(repeats)]
                    initial=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
                    # Same draw keys across policies and kernels; no training state touched.
                    keys=jax.vmap(jax.random.PRNGKey)(jnp.asarray([SPEC['evaluation']['sample_seed_base']+seed*1000+i*8+r
                                                               for i in range(16) for r in range(repeats)],jnp.uint32))
                    key=(kernel,stochastic)
                    if key not in evaluators:evaluators[key]=make_evaluator(network,kernel,stochastic,repeats)
                    outputs=jax.device_get(evaluators[key](runner.train_state.params,initial,keys))
                    records=[]
                    for i,entry in enumerate(panel['states']):
                        for rep in range(repeats):
                            idx=i*repeats+rep
                            records.append({'map_id':entry['map_id'],'family':entry['family'],'repeat':rep,
                                            **{k:np.asarray(v[idx]).tolist() for k,v in outputs.items()}})
                    write(dest,{'seed':seed,'condition':label,'kernel':kernel,'mode':mode,
                                'checkpoint_sha256':sha(path/'state.msgpack'),'panel_sha256':sha(ROOT/'holdout_v2.json'),
                                'records':records,'training_updates':0})
                    print(f'[evaluation] {dest.name}',flush=True)
    summarize()


def summarize():
    rows=[]
    for p in sorted((ROOT/'evaluation').glob('*.json')):
        d=json.loads(p.read_text())
        for family in ['natural_reset','opportunity']:
            records=[x for x in d['records'] if x['family']==family]
            rows.append({k:d[k] for k in ['seed','condition','kernel','mode']}|{'family':family,'episodes':len(records),
                         'stage_rates':np.asarray([x['stage_executed'] for x in records]).mean(axis=0).tolist(),
                         'stage_and_completion_rates':np.asarray([np.asarray(x['stage_executed'])*x['success'] for x in records]).mean(axis=0).tolist(),
                         **{k:statistics.mean(x[k] for x in records) for k in
                            ['adoption','success','discounted_return','length','opportunity_reached','violation']}})
    assert sum(x['episodes'] for x in rows)==4320
    def query(seed,condition):
        return next(x['adoption'] for x in rows if x['seed']==seed and x['condition']==condition and x['kernel']=='mutant'
                    and x['mode']=='mode' and x['family']=='natural_reset')
    points=[{'seed':s,'delta':query(s,'D-on')-query(s,'D-delivery')} for s in SEEDS]
    values=np.array([x['delta'] for x in points]);rng=np.random.default_rng(SPEC['evaluation']['bootstrap_seed'])
    boots=rng.choice(values,(SPEC['evaluation']['bootstrap_resamples'],len(SEEDS)),replace=True).mean(axis=1)
    changes=[]
    kernel_gaps=[]
    for row in rows:
        if row['kernel']=='mutant':
            fixed=next(x for x in rows if x['kernel']=='fixed' and all(x[k]==row[k] for k in ['seed','condition','mode','family']))
            kernel_gaps.append({k:row[k] for k in ['seed','condition','mode','family']}|
                               {k+'_mutant_minus_fixed':row[k]-fixed[k] for k in ['success','discounted_return','length']})
        if row['condition']=='source':continue
        src=next(x for x in rows if x['condition']=='source' and all(x[k]==row[k] for k in ['seed','kernel','mode','family']))
        changes.append({k:row[k] for k in ['seed','condition','kernel','mode','family']}|
                       {k+'_minus_source':row[k]-src[k] for k in ['adoption','success','discounted_return','opportunity_reached']})
    result={'complete':True,'primary_not_U':True,'primary_seed_points':points,'mean':float(values.mean()),
            'range':[float(values.min()),float(values.max())],'bootstrap_95_percentile_interval':np.quantile(boots,[.025,.975]).tolist(),
            'seed_family_kernel_rows':rows,'evaluation_episodes':4320,'independent_unit':'learner seed',
            'changes_from_frozen_source':changes,'failures':[],
            'same_policy_kernel_differences':kernel_gaps,
            'interpretation':'Report additional-goal benefit as unresolved if uncertainty remains; no automatic extension.'}
    write(ROOT/'result.json',result)
    write(ROOT/'complete.json',{'result_sha256':sha(ROOT/'result.json'),'completed_unix':time.time(),
                               'final_checkpoint_sha256':{j['id']:json.loads((cell(j)/'summary.json').read_text())['checkpoint_sha256'] for j in jobs()}})


def worker():
    ROOT.mkdir(parents=True,exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        # Immutable completed artifact, not current report-code HEAD, controls terminal status.
        if (ROOT/'complete.json').exists():
            receipt=json.loads((ROOT/'complete.json').read_text());assert sha(ROOT/'result.json')==receipt['result_sha256'];return
        if (ROOT/'failure.json').exists():
            print('TERMINAL_FAILURE: inspect failure.json; no automatic retry',flush=True);return
        verify()
        if not (ROOT/'validation.json').exists():raise RuntimeError('validation gate not passed')
        validation=json.loads((ROOT/'validation.json').read_text())
        assert validation['passed'] and validation['execution_source_sha256']==verify()['source_sha256']
        for job in jobs():
            command=[sys.executable,str(Path(__file__).resolve()),'--job',job['id']]
            result=subprocess.run(command,cwd=REPO)
            if result.returncode:
                write(ROOT/'failure.json',{'job':job['id'],'returncode':result.returncode,'unix':time.time(),
                                         'reason':'Child failed; no replacement or repeated automatic launch. Inspect job failure record/log.'})
                return
        evaluate_all()


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--job');p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    try:
        if args.prepare:prepare()
        elif args.job:run_job(next(j for j in jobs() if j['id']==args.job))
        elif args.evaluate:evaluate_all()
        elif args.worker:worker()
        else:p.error('choose --prepare, --worker, --job or --evaluate')
    except Exception:
        error={'unix':time.time(),'traceback':traceback.format_exc(),'job':args.job}
        if args.job:write(cell(next(j for j in jobs() if j['id']==args.job))/'failure.json',error)
        elif args.worker:write(ROOT/'failure.json',error)
        print(error['traceback'],flush=True)
        # Host transitions to idle resource management after recorded terminal failure.
        if not args.worker:raise


if __name__=='__main__':main()
