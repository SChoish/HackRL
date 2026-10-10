#!/usr/bin/env python3
"""Verify all registered cells and build final-only S/M/L tables without JAX."""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from statistics import mean
import time

REPO=Path(__file__).resolve().parents[1]
SPEC=json.loads((REPO/'docs/manifests/spatial_delay_scale_v1.json').read_text())
ROOT=Path(SPEC['run_root'])
GROUPS={'0':(0,),'1-8':tuple(range(1,9)),'10,12':(10,12)}
METRICS=('mutant_success','mutant_wall_pass','mutant_control_length','fixed_success','mutant_return','fixed_return','return_gap','one_shot_success','one_shot_wall_pass')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise RuntimeError(message)
def write(path,text):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending')
    with temp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    temp.replace(path)

def aggregate(rows):
    seed_rows=[];groups=[]
    for size in ('S','M','L'):
        for q in SPEC['fixed_probabilities']:
            for selection in ('mode','sample'):
                for group,delays in GROUPS.items():
                    seeds=[]
                    for seed in SPEC['seeds']:
                        selected=[r for r in rows if (r['size'],r['q'],r['selection'],r['seed'])==(size,q,selection,seed) and r['delay'] in delays]
                        require(sorted(r['delay'] for r in selected)==list(delays),'Missing/duplicate seed-delay')
                        item=dict(size=size,q=q,selection=selection,seed=seed,group=group,**{k:mean(r[k] for r in selected) for k in METRICS})
                        seeds.append(item);seed_rows.append(item)
                    groups.append(dict(size=size,q=q,selection=selection,group=group,n_learners=5,**{k:mean(r[k] for r in seeds) for k in METRICS}))
    return seed_rows,groups

def main():
    provenance=read(ROOT/'run_manifest.json');code=provenance['execution_sha']
    sroot=Path(SPEC['s_root']);sreceipt=read(sroot/'verified_complete.json')
    require(sreceipt['execution_sha']==SPEC['s_execution_sha'],'Wrong S execution')
    require(sreceipt['result_sha256']==sha(sroot/'result.json'),'S result changed')
    records=[];inputs={};capabilities=[];accounting=[];episodes=0;slots=0
    for size in ('S','M','L'):
        for seed in SPEC['seeds']:
            if size!='S':
                pre=ROOT/size/f'pretraining/dual_seed_{seed}'
                summary=read(pre/'summary.json')
                require(summary['status']=='complete' and summary['collector_slots']==16777216,'Pretrain unfinished')
                slots+=summary['collector_slots'];capabilities.append(summary)
                for u in SPEC['pretrain_checkpoints']:
                    cp=pre/f'checkpoints/update_{u}';m=read(cp/'metadata.json');ep=cp/'evaluation.json';e=read(ep)
                    require(m['state_sha256']==sha(cp/'state.msgpack') and m['execution_sha']==code,'Pretrain checkpoint integrity')
                    cfg=read(cp/'config.json');require(cfg['policy_hidden_size']==SPEC['policy_widths'][size] and cfg['teacher_hidden_size']==512,'Pretrain width')
                    require(read(cp/'evaluation_integrity.json')['sha256']==sha(ep),'Pretrain evaluation integrity')
                    require(e['training_update']==u,'Wrong pretrain evaluation update')
                    for kernel in ('fixed','mutant'):
                        for mode,n in [('mode',16),('sample',1024)]:
                            require(len(e[kernel][mode]['episode_records'])==n,'Pretrain episode count');episodes+=n
                source_hash=sha(pre/'checkpoints/update_512/state.msgpack')
            origins=[]
            for q in SPEC['fixed_probabilities']:
                cell=(sroot if size=='S' else ROOT/size/'adaptation')/f'fixed_{round(q*100):02d}pct_seed_{seed}'
                summary=read(cell/'summary.json');origin=read(cell/'origin.json');origins.append(origin)
                require(summary['status']=='complete' and summary['updates']==4096 and summary['collector_slots']==134217728,'Adaptation unfinished')
                accounting.append(dict(size=size,q=q,seed=seed,exposure=summary['exposure'],counters=summary['counters'],initial_counters=origin['initial_counters']))
                if size!='S':
                    slots+=summary['collector_slots'];require(origin['source_sha256']==source_hash,'Wrong branch source')
                    for u in SPEC['adapt_checkpoints']:
                        cp=cell/f'checkpoints/adapt_{u}';m=read(cp/'metadata.json');ep=cp/'evaluation.json';e=read(ep)
                        require(m['state_sha256']==sha(cp/'state.msgpack') and m['execution_sha']==code,'Adaptation checkpoint integrity')
                        cfg=read(cp/'config.json');require(cfg['policy_hidden_size']==SPEC['policy_widths'][size] and cfg['teacher_hidden_size']==512,'Adaptation width')
                        require(read(cp/'evaluation_integrity.json')['sha256']==sha(ep),'Adaptation evaluation integrity')
                        require(e['update']==u,'Wrong adaptation evaluation update')
                        for mode,n in [('mode',16),('sample',128)]:
                            require(len(e['results'][mode])==33,'Missing evaluation conditions')
                            for v in e['results'][mode].values():
                                require(len(v['episodes'])==n,'Adaptation episode count');episodes+=n
                ep=cell/'checkpoints/adapt_4096/evaluation.json';inputs[str(ep)]=sha(ep)
                require(summary['final_evaluation_sha256']==inputs[str(ep)],'Final evaluation summary integrity')
                ev=read(ep)['results']
                for selection in ('mode','sample'):
                    for delay in (*range(9),10,12):
                        m=ev[selection][f'mutant_delay{delay}']['summary'];f=ev[selection][f'fixed_delay{delay}']['summary'];b=ev[selection][f'mutant_block_once_delay{delay}']['summary']
                        records.append(dict(size=size,q=q,seed=seed,selection=selection,delay=delay,
                            mutant_success=m['success_rate'],mutant_wall_pass=m['wall_pass_rate'],mutant_control_length=m['mean_control_length'],fixed_success=f['success_rate'],
                            mutant_return=m['mean_discounted_return'],fixed_return=f['mean_discounted_return'],return_gap=m['mean_discounted_return']-f['mean_discounted_return'],one_shot_success=b['success_rate'],one_shot_wall_pass=b['wall_pass_rate'],
                            one_shot_triggered_episodes=b['triggered_episodes'],one_shot_untriggered_episodes=b['untriggered_episodes'],one_shot_retries_given_block=b['mean_retries_given_block']))
            require(origins[0]['commanded_full_state_sha256']==origins[1]['commanded_full_state_sha256'],'Branch start mismatch')
    require(slots==2852126720 and episodes==864320,'Final budget mismatch')
    seed_groups,groups=aggregate(records)
    result=dict(status='complete',jobs_completed=30,new_collector_slots=slots,new_evaluation_episodes=episodes,execution_sha=code,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),input_sha256=inputs,normal_capability=capabilities,accounting=accounting,
        by_seed_delay=records,by_seed_group=seed_groups,groups=groups,
        interpretation='All five learner seeds, equal learner weights after equal delay weights; final4096 only. Normal failures retained. Control lengths include failures: inspect success alongside length. Previously observed10/12 evaluation points, not a new holdout. Teacher architecture fixed, learned teacher and data may differ.')
    write(ROOT/'result.json',json.dumps(result,indent=2,allow_nan=False)+'\n')
    for filename,data in [('seed_delay.csv',records),('seed_group.csv',seed_groups)]:
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(data[0]));w.writeheader();w.writerows(data);write(ROOT/filename,stream.getvalue())
    lines=['# Spatial random-delay size comparison','',result['interpretation'],'',
           '| Size | Fixed fraction | Delay | Mutant success | Wall pass | Control ticks | Fixed success | Return gap |',
           '|---|---:|---|---:|---:|---:|---:|---:|']
    for g in groups:
        if g['selection']=='mode':lines.append(f"| {g['size']} | {g['q']:.0%} | {g['group']} | {g['mutant_success']:.1%} | {g['mutant_wall_pass']:.1%} | {g['mutant_control_length']:.3f} | {g['fixed_success']:.1%} | {g['return_gap']:+.6f} |")
    write(ROOT/'summary.md','\n'.join(lines)+'\n')
    write(ROOT/'verified_complete.json',json.dumps(dict(execution_sha=code,result_sha256=sha(ROOT/'result.json')))+'\n')
    print(json.dumps(dict(event='verified_complete',jobs=30,collector_slots=slots,episodes=episodes)),flush=True)

if __name__=='__main__':main()
