#!/usr/bin/env python3
"""Five fixed continuations; immutable historical inputs; final-only traced eval."""
import argparse
from dataclasses import asdict, fields
import fcntl
import json
import os
from pathlib import Path
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
import run_pack_teacher_holdout_v2 as old
from hackrl.pack_teacher_holdout import deserialize_state
from hackrl.pack_teacher_fixed_trace import make_traced_evaluator

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/pack_teacher_fixed_control_v1.json'
SPEC=json.loads(SPEC_PATH.read_text())
ROOT=Path(SPEC['run_root']);OLD=Path(SPEC['old_root']);SEEDS=SPEC['seeds']
base=old.base;sha=old.sha
NEW_FILES=['docs/manifests/pack_teacher_fixed_control_v1.json','src/hackrl/pack_teacher_fixed_trace.py',
           'scripts/run_pack_teacher_fixed_control_v1.py','scripts/probe_pack_teacher_fixed_control_v1.py',
           'scripts/run_pack_teacher_fixed_control_v1_gpu.sh','scripts/run_pack_teacher_fixed_control_v1_cpu.sh',
           'tests/test_pack_teacher_fixed_control.py']


def read(path):return json.loads(Path(path).read_text())
def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write(path,value):
    old.write(path,value)
    sync_directory(Path(path).parent)


def append(path,value):
    path=Path(path)
    with path.open('a') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    sync_directory(path.parent)


def source(seed):return OLD/f'source/seed{seed}/checkpoints/update_512'
def cell(job):return ROOT/job['condition']/f"seed{job['seed']}"
def jobs():
    return [{'id':f'D-delivery-fixed-s{s}','condition':'D-delivery-fixed','kind':'adapt','env':'pack',
             'size':'S','seed':s,'policy_hidden_size':512,'teacher_hidden_size':512,'method':'dual',
             'variant':'fixed','learn_teacher':True,'imitate_teacher':True,'teacher_goal_indices':[11],
             'pretrain_checkpoint':str(source(s)), 'origin_arm':{'learn_teacher':True,'imitate_teacher':True}}
            for s in SEEDS]


def capacity():
    home=shutil.disk_usage('/home/ext_csv');raid=shutil.disk_usage('/raid/ext_csv')
    existing=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) if ROOT.exists() else 0
    info={'unix':time.time(),'home':home._asdict(),'raid':raid._asdict(),'existing_run_bytes':existing,
          'old_run_bytes':sum(p.stat().st_size for p in OLD.rglob('*') if p.is_file()),
          'projected_additional_peak_bytes':SPEC['storage']['projected_peak_bytes'],
          'reserve_bytes':max(8*1024**3,int(.2*SPEC['storage']['projected_peak_bytes'])),
          'chosen_filesystem':'/raid/ext_csv','migration':'none; new run only'}
    if raid.free<info['projected_additional_peak_bytes']+info['reserve_bytes']:
        raise RuntimeError('RAID reserve insufficient; no checkpoint write permitted')
    ROOT.mkdir(parents=True,exist_ok=True);append(ROOT/'capacity.jsonl',info)
    return info


def origin_record(job,runner,leo):
    components={'actor_critic_adam':runner.train_state,'teacher_renorm_adam':leo,
                'environment':runner.env_state,'env_rng':runner.env_keys,'sampler_rng':runner.sampler_keys,
                'runner_rng':runner.rng,'whole_runner':runner}
    assert np.all(np.asarray(runner.current_goal)==11)
    return {'seed':job['seed'],'source_sha256':sha(source(job['seed'])/'state.msgpack'),
            'components':{k:old.tree_hash(v) for k,v in components.items()}}


def prepare():
    capacity()
    if (ROOT/'run_manifest.json').exists():verify();return
    assert sha(OLD/'run_manifest.json')==SPEC['old_manifest_sha256']
    m=read(OLD/'run_manifest.json');receipt=read(OLD/'complete.json')
    assert m['execution_head']==SPEC['old_execution_sha']
    assert sha(OLD/'result.json')==receipt['result_sha256']
    inputs={str(p):sha(p) for p in sorted((OLD/'evaluation').glob('*.json'))}
    for name in ['run_manifest.json','complete.json','result.json','holdout_v2.json','validation.json','launch_verification.json']:
        inputs[str(OLD/name)]=sha(OLD/name)
    config_checks=[]
    for job in jobs():
        s=job['seed'];mutant=OLD/f'D-delivery/seed{s}/checkpoints/adapt_4096'
        for checkpoint in [source(s),mutant]:
            integrity=read(checkpoint/'integrity.json')
            for name,h in integrity['checkpoint_files'].items():assert sha(checkpoint/name)==h
            for p in checkpoint.iterdir():
                if p.is_file():inputs[str(p)]=sha(p)
        for p in [OLD/f'paired_origin_seed{s}.json',OLD/f'D-delivery/seed{s}/origin.json']:
            inputs[str(p)]=sha(p)
        assert read(OLD/f'paired_origin_seed{s}.json')==read(OLD/f'D-delivery/seed{s}/origin.json')
        cfg=asdict(base._job_config(job,goal_mode='deliver_3',variant='fixed',updates=4096))
        prior=read(mutant/'config.json')
        cfg=json.loads(json.dumps(cfg))
        changes={k:[prior.get(k),cfg.get(k)] for k in set(prior)|set(cfg) if prior.get(k)!=cfg.get(k)}
        assert changes=={'variant':['mutant','fixed']},changes
        assert base._arm(job)==read(mutant/'arm.json')
        config_checks.append({'seed':s,'differences':changes,'arm':base._arm(job)})
    files=list(dict.fromkeys(list(m['source_sha256'])+NEW_FILES+
           ['scripts/summarize_pack_restore_scale_bc.py','scripts/summarize_pack_restore_delivery_only_teacher.py']))
    for f,h in m['source_sha256'].items():assert sha(REPO/f)==h,f'changed old implementation {f}'
    dirty=subprocess.check_output(['git','status','--porcelain','--',*files],cwd=REPO,text=True)
    assert not dirty,'commit execution sources before prepare'
    times=[read(OLD/f'D-delivery/seed{s}/summary.json')['median_seconds_per_update'] for s in SEEDS]
    write(ROOT/'run_manifest.json',{'experiment_id':SPEC['experiment_id'],'execution_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
          'jobs':jobs(),'spec':SPEC,'source_sha256':{f:sha(REPO/f) for f in files},'old_input_sha256':inputs,
          'panel_sha256':sha(OLD/'holdout_v2.json'),'config_checks':config_checks,
          'historical_median_seconds_per_update':statistics.median(times),'projected_compute_seconds':20480*statistics.median(times),
          'created_unix':time.time(),'exploratory_seen_panel':True,'evaluation_has_started':False})


def verify(full=False):
    m=read(ROOT/'run_manifest.json');assert m['jobs']==jobs() and m['spec']==SPEC
    for f,h in m['source_sha256'].items():assert sha(REPO/f)==h,f
    for f,h in m['old_input_sha256'].items():
        if full or not f.endswith('state.msgpack'):assert sha(f)==h,f
    return m


def initialize(job):
    cfg=base._job_config(job,goal_mode='deliver_3',variant='fixed',updates=4096)
    network,template=base.ENVS['pack']['initialize'](cfg)
    teacher,leo_template,batch=base._start_teacher(base.ENVS['pack'],cfg,template)
    return cfg,network,teacher,template,leo_template,batch


def validate_origins():
    verify(full=True);rows=[]
    for job in jobs():
        cfg,net,teacher,template,leo_template,batch=initialize(job)
        runner,leo=base._load_adapt_start(ROOT,job,template,cfg)
        record=origin_record(job,runner,leo)
        expected=read(OLD/f"paired_origin_seed{job['seed']}.json")
        assert record==expected, f"original initial-state mismatch seed {job['seed']}; do not substitute"
        rows.append(record)
    write(ROOT/'validation.json',{'passed':True,'full_post_command_initial_state_matches':rows,
          'config_only_variant_diff':True,'new_training_updates':0,'historical_policy_queries':0,
          'backend':jax.default_backend(),'manifest_sha256':sha(ROOT/'run_manifest.json')})
    print('RESTORATION_CONTRACT_PASSED all five sources and historical full-origin hashes',flush=True)


def save(job,runner,leo,cfg,number):
    capacity();root=cell(job)/'checkpoints';root.mkdir(parents=True,exist_ok=True)
    dest=root/f'adapt_{number}';tmp=root/f'.adapt_{number}.tmp-{os.getpid()}'
    assert not dest.exists()
    base._save_state(base.ENVS['pack'],'dual',tmp,runner,leo,cfg,base._arm(job))
    write(tmp/'integrity.json',{'state_sha256':sha(tmp/'state.msgpack'),'config_sha256':sha(tmp/'config.json'),
          'checkpoint_files':base._checkpoint_fingerprint(tmp)['files'],'global_update':int(runner.global_update)})
    for p in tmp.iterdir():
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    sync_directory(tmp);os.replace(tmp,dest);sync_directory(root)
    # Only this run's superseded resumable checkpoints; original sources never touched.
    for p in root.glob('adapt_*'):
        if p!=dest and p.is_dir():shutil.rmtree(p)
    sync_directory(root)


def run_job(job):
    verify(full=True);capacity()
    validation=read(ROOT/'validation.json')
    assert validation['passed'] and validation['manifest_sha256']==sha(ROOT/'run_manifest.json')
    dest=cell(job);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'failure.json').exists(),'terminal failed branch'
    cfg,network,teacher,template,leo_template,batch=initialize(job)
    checkpoints=sorted((dest/'checkpoints').glob('adapt_*'),key=lambda p:int(p.name.split('_')[-1]))
    if checkpoints:
        runner,leo=old.load_checked(checkpoints[-1],template,leo_template)
    else:
        runner,leo=base._load_adapt_start(ROOT,job,template,cfg)
        record=origin_record(job,runner,leo)
        assert record==read(OLD/f"paired_origin_seed{job['seed']}.json")
        write(dest/'origin.json',record)
        save(job,runner,leo,cfg,0)
    current=int(runner.global_update)-512
    intent=read(dest/'update_intent.json') if (dest/'update_intent.json').exists() else None
    if intent and intent['update']>current:
        raise RuntimeError('Interrupted unsaved/in-flight update: strict budget forbids automatic replay; preserve and report')
    attempts=[json.loads(x) for x in (dest/'updates.jsonl').read_text().splitlines()] if (dest/'updates.jsonl').exists() else []
    assert [x['update'] for x in attempts]==list(range(1,current+1)), 'log/checkpoint mismatch; no replay'
    if current==4096 and (dest/'summary.json').exists():return
    update=base._dual_update(base.ENVS['pack'],network,teacher,cfg,batch,base._arm(job))
    finite=jax.jit(lambda a,b:jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((a,b))])))
    first=current;start=time.monotonic();seconds=[]
    while current<4096:
        write(dest/'update_intent.json',{'update':current+1,'collector_slots':32768,'unix':time.time()})
        t=time.monotonic();runner,leo,metrics=update(runner,leo);jax.block_until_ready(runner.train_state.step)
        host={k:float(np.asarray(v)) for k,v in metrics.items()}
        assert all(np.isfinite(v) for v in host.values()) and bool(finite(runner.train_state,leo))
        assert np.all(np.asarray(runner.current_goal)==11) and host['teacher_target_head_count']==1
        current=int(runner.global_update)-512;elapsed=time.monotonic()-t;seconds.append(elapsed)
        row={'global_update':int(runner.global_update),'update':current,'seconds':elapsed,**host};attempts.append(row)
        append(dest/'updates.jsonl',row)
        if current%64==0 or current==4096:save(job,runner,leo,cfg,current)
        if current==1 or current%32==0:print(json.dumps({'job':job['id'],'update':current,'target':4096,'seconds':elapsed}),flush=True)
    assert len(attempts)==4096 and [r['update'] for r in attempts]==list(range(1,4097))
    counters=['valid_transitions','teacher_valid_samples','teacher_target_terms','ppo_applied_minibatches','teacher_applied_minibatches']
    write(dest/'summary.json',{'job':job,'complete':True,'global_update':4608,'collector_slots':134217728,
          'completed_update_attempt_slots':len(attempts)*32768,'starting_update_this_process':first,
          'seconds_this_process':time.monotonic()-start,'median_seconds_per_update':statistics.median(seconds) if seconds else None,
          'ppo_optimizer_step':int(runner.train_state.step),'teacher_optimizer_step':int(leo.step),
          'counters':{k:sum(r[k] for r in attempts) for k in counters},
          'checkpoint_sha256':sha(dest/'checkpoints/adapt_4096/state.msgpack')})


def flatten_trace(trace):
    active=np.asarray(trace['active']);ticks,episodes=np.nonzero(active)
    result={'step_index':ticks.astype(np.int32),'episode_index':episodes.astype(np.int32)}
    for k,v in trace.items():
        if k in ('before','after'):
            for field in fields(v):result[f'{k}_{field.name}']=np.asarray(getattr(v,field.name))[active]
        elif k!='active':result[k]=np.asarray(v)[active]
    return result


def claim_evaluation(bundle, intent, identity):
    if bundle.exists():return
    if intent.exists():
        raise RuntimeError('Interrupted uncommitted evaluation: no automatic rerollout beyond episode budget')
    write(intent,identity)


def load_bundle(bundle):
    # Read every array (including ZIP CRC verification), not only the metadata.
    with np.load(bundle,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    saved=json.loads(arrays.pop('metadata_json').tobytes())
    assert all(len(v)==saved['trace_transitions'] for v in arrays.values())
    np.testing.assert_array_equal(arrays['after_tick'],arrays['before_tick']+1)
    np.testing.assert_array_equal(arrays['created'],np.maximum(arrays['total_after']-arrays['total_before'],0))
    np.testing.assert_array_equal(arrays['grain_charged'],np.minimum(arrays['balance_before']+arrays['created'],arrays['delivered']))
    assert np.all((arrays['episode_index']>=0)&(arrays['episode_index']<len(saved['records'])))
    for i,r in enumerate(saved['records']):
        mask=arrays['episode_index']==i
        assert int(mask.sum())==r['length']
        assert int(arrays['grain_charged'][mask].sum())==r['violation_grain_delivered']
    return saved


def evaluate_new():
    m=verify(full=True);capacity();panel=read(OLD/'holdout_v2.json');evaluators={}
    assert all(read(cell(j)/'summary.json')['complete'] for j in jobs())
    for job in jobs():
        cfg,network,teacher,template,leo_template,batch=initialize(job)
        path=cell(job)/'checkpoints/adapt_4096'
        runner,leo=old.load_checked(path,template,leo_template)
        for kernel in ['fixed','mutant']:
            for stochastic,repeats in [(False,1),(True,8)]:
                mode='sample' if stochastic else 'mode'
                dest=ROOT/'evaluation'/f"D-delivery-fixed_s{job['seed']}_{kernel}_{mode}.json"
                bundle=dest.with_suffix('.npz');dest.parent.mkdir(parents=True,exist_ok=True)
                key_values=[SPEC['evaluation']['sample_seed_base']+job['seed']*1000+i*8+r for i in range(16) for r in range(repeats)]
                if bundle.exists():
                    saved=load_bundle(bundle)
                else:
                    claim_evaluation(bundle,dest.with_suffix('.intent'),{'seed':job['seed'],'kernel':kernel,'mode':mode,'episodes':16*repeats})
                    states=[deserialize_state(x['state']) for x in panel['states'] for _ in range(repeats)]
                    initial=jax.tree.map(lambda *xs:jnp.stack(xs),*states)
                    keys=jax.vmap(jax.random.PRNGKey)(jnp.asarray(key_values,jnp.uint32))
                    key=(kernel,stochastic)
                    if key not in evaluators:evaluators[key]=make_traced_evaluator(network,kernel,stochastic)
                    outputs,trace=jax.device_get(evaluators[key](runner.train_state.params,initial,keys))
                    records=[]
                    for i,entry in enumerate(panel['states']):
                        for rep in range(repeats):
                            idx=i*repeats+rep
                            records.append({'map_id':entry['map_id'],'family':entry['family'],'repeat':rep,
                                            **{k:np.asarray(v[idx]).tolist() for k,v in outputs.items()}})
                    saved={'seed':job['seed'],'condition':'D-delivery-fixed','kernel':kernel,'mode':mode,
                           'checkpoint_sha256':sha(path/'state.msgpack'),'panel_sha256':m['panel_sha256'],
                           'records':records,'training_updates':0,'trace_schema':'active complete before/after states v1',
                           'sample_draw_keys':key_values,'trace_episode_order':'panel state order then repeat',
                           'trace_transitions':int(np.asarray(trace['active']).sum())}
                    arrays=flatten_trace(trace)
                    arrays['metadata_json']=np.frombuffer(json.dumps(saved,sort_keys=True).encode(),np.uint8)
                    tmp=bundle.with_suffix('.npz.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
                    os.replace(tmp,bundle);sync_directory(bundle.parent)
                    saved=load_bundle(bundle)
                assert saved['checkpoint_sha256']==sha(path/'state.msgpack') and saved['panel_sha256']==m['panel_sha256']
                assert (saved['seed'],saved['condition'],saved['kernel'],saved['mode'])==(job['seed'],'D-delivery-fixed',kernel,mode)
                assert saved['sample_draw_keys']==key_values
                assert len(saved['records'])==16*repeats
                saved.update(trace_file=str(bundle),trace_sha256=sha(bundle))
                if dest.exists():assert read(dest)==saved
                else:write(dest,saved)
                print('[evaluation]',dest.name,flush=True)
    summarize()


def summarize():
    verify(full=True);rows=[];files={}
    old_files=[p for p in sorted((OLD/'evaluation').glob('*.json')) if p.name.startswith(('source_','D-delivery_'))]
    new_files=sorted((ROOT/'evaluation').glob('*.json'))
    assert len(old_files)==40 and len(new_files)==20
    new_episodes=0
    for p in old_files+new_files:
        d=read(p);files[str(p)]=sha(p)
        if p in new_files:
            assert sha(d['trace_file'])==d['trace_sha256'];new_episodes+=len(d['records'])
        for family in ['natural_reset','opportunity']:
            rs=[r for r in d['records'] if r['family']==family]
            rows.append({k:d[k] for k in ['seed','condition','kernel','mode']}|{'family':family,'episodes':len(rs),
                        **{k:statistics.mean(r[k] for r in rs) for k in ['success','adoption','discounted_return','length','opportunity_reached','violation']}})
    assert new_episodes==1440
    def get(seed,condition,kernel,mode,family):
        return next(r for r in rows if (r['seed'],r['condition'],r['kernel'],r['mode'],r['family'])==(seed,condition,kernel,mode,family))
    contrasts=[];changes=[]
    for family in ['natural_reset','opportunity']:
        for mode in ['sample','mode']:
            points=[]
            for s in SEEDS:
                f=get(s,'D-delivery-fixed','fixed',mode,family);m=get(s,'D-delivery','fixed',mode,family)
                fm=get(s,'D-delivery-fixed','mutant',mode,family);mm=get(s,'D-delivery','mutant',mode,family)
                points.append({'seed':s,'fixed_success_fixed_minus_mutant_training':f['success']-m['success'],
                               'fixed_return_fixed_minus_mutant_training':f['discounted_return']-m['discounted_return'],
                               'D_delivery_U':mm['adoption']-fm['adoption']})
                for condition in ['D-delivery-fixed','D-delivery']:
                    for kernel in ['fixed','mutant']:
                        row=get(s,condition,kernel,mode,family);src=get(s,'source',kernel,mode,family)
                        changes.append({'seed':s,'condition':condition,'kernel':kernel,'mode':mode,'family':family,
                                        **{k+'_minus_source':row[k]-src[k] for k in ['success','adoption','discounted_return']}})
            estimates={}
            for key in points[0]:
                if key=='seed':continue
                values=np.array([p[key] for p in points]);rng=np.random.default_rng(SPEC['evaluation']['bootstrap_seed'])
                boots=rng.choice(values,(20000,5),replace=True).mean(axis=1)
                estimates[key]={'mean':float(values.mean()),'range':[float(values.min()),float(values.max())],
                                'bootstrap_95_percentile_interval':np.quantile(boots,[.025,.975]).tolist()}
            contrasts.append({'family':family,'mode':mode,'seed_points':points,'estimates':estimates})
    result={'complete':True,'exploratory_seen_panel':True,'new_evaluation_episodes':1440,'reused_evaluation_episodes':2880,
            'evaluation_episodes':4320,'primary':contrasts[0],'separate_contrasts':contrasts,
            'seed_family_kernel_rows':rows,'changes_from_source':changes,'input_evaluation_sha256':files,
            'independent_unit':'5 learner seeds','adoption_metric_unchanged':True,
            'opportunity_warning':SPEC['opportunity_reached'],'U_is_not_D_on_D_delivery_Delta_U':True,
            'failures':[],'new_collector_slots':sum(read(cell(j)/'summary.json')['collector_slots'] for j in jobs())}
    assert result['new_collector_slots']==671088640
    write(ROOT/'result.json',result)
    write(ROOT/'complete.json',{'result_sha256':sha(ROOT/'result.json'),'completed_unix':time.time(),
          'final_checkpoint_sha256':{j['id']:read(cell(j)/'summary.json')['checkpoint_sha256'] for j in jobs()},
          'evaluation_sha256':{str(p):sha(p) for p in new_files},
          'trace_sha256':{str(p):sha(p) for p in sorted((ROOT/'evaluation').glob('*.npz'))}})


def worker():
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (ROOT/'complete.json').exists():
            assert sha(ROOT/'result.json')==read(ROOT/'complete.json')['result_sha256'];return
        if (ROOT/'failure.json').exists():print('TERMINAL_FAILURE; no retry',flush=True);return
        verify(full=True)
        for job in jobs():
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',job['id']],cwd=REPO)
            if r.returncode:
                write(ROOT/'failure.json',{'reason':'Branch failed; no replacement or extension','job':job['id'],'returncode':r.returncode,'unix':time.time()});return
        evaluate_new()


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--validate',action='store_true');g.add_argument('--worker',action='store_true');g.add_argument('--job')
    args=p.parse_args()
    try:
        if args.prepare:prepare()
        elif args.validate:validate_origins()
        elif args.worker:worker()
        else:run_job(next(j for j in jobs() if j['id']==args.job))
    except Exception:
        error={'unix':time.time(),'traceback':traceback.format_exc(),'job':args.job}
        if args.job:write(cell(next(j for j in jobs() if j['id']==args.job))/'failure.json',error)
        elif args.worker:write(ROOT/'failure.json',error)
        print(error['traceback'],flush=True)
        if not args.worker:raise


if __name__=='__main__':main()
