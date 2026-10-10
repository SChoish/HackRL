#!/usr/bin/env python3
"""Run the preregistered 15,120 frozen episodes, resume only finished policy cells."""
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
from hackrl import spatial_wall_pass as env
from hackrl import spatial_wall_pass_gc as gc
from hackrl import spatial_recovery_probe as probe

REPO = Path(__file__).resolve().parents[1]
SPEC = json.loads((REPO/'docs/manifests/spatial_recovery_probe_v1.json').read_text())
ROOT = Path(SPEC['run_root'])
MIX = Path(SPEC['mixture_root'])
BASE = Path(SPEC['replication_root'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    temporary.replace(path)


def checkpoint(family, seed):
    if family == 'mixed_10pct':
        return MIX/f'fixed_10pct_seed_{seed}/checkpoints/adapt_4096'
    branch = 'mutant_adapt' if family == 'mutant_only' else 'fixed_continue'
    return BASE/f'adaptation/{branch}_seed_{seed}/checkpoints/adapt_4096'


def baseline(family, seed):
    if family == 'mixed_10pct':
        return json.loads((checkpoint(family,seed)/'evaluation.json').read_text())
    branch = 'mutant_adapt' if family == 'mutant_only' else 'fixed_continue'
    return json.loads((MIX/f'endpoints/{branch}_seed_{seed}.json').read_text())


def capacity(event):
    projected = SPEC['storage']['projected_artifacts_bytes']
    reserve = max(8*1024**3, int(np.ceil(.2*projected)))
    info = dict(event=event,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        chosen_filesystem='/raid/ext_csv',projected_remaining_write_bytes=projected,
        safety_reserve_bytes=reserve,existing_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        existing_source_run_bytes=sum(p.stat().st_size for p in MIX.rglob('*') if p.is_file()))
    if info['raid']['free'] < projected+reserve:
        raise RuntimeError('Insufficient RAID peak plus reserve; no evaluation launched')
    with (ROOT/'capacity.jsonl').open('a') as f: f.write(json.dumps(info)+'\n')


def provenance():
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise RuntimeError('Commit evaluation code before execution')
    original=json.loads((MIX/'run_manifest.json').read_text())
    for path in ('src/hackrl/spatial_wall_pass.py','src/hackrl/spatial_wall_pass_gc.py'):
        assert sha(REPO/path)==original['source_file_sha256'][path]
    for family in SPEC['policies']:
        for seed in SPEC['seeds']:
            cp=checkpoint(family,seed)
            digest=sha(cp/'state.msgpack')
            metadata=json.loads((cp/'metadata.json').read_text())
            assert metadata['global_update']==4608
            expected=metadata['state_sha256'] if family=='mixed_10pct' else baseline(family,seed)['source_state_sha256']
            assert digest==expected
    return dict(spec=SPEC,execution_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_checkpoints={f'{family}_{seed}':sha(checkpoint(family,seed)/'state.msgpack')
                            for family in SPEC['policies'] for seed in SPEC['seeds']},
        source_configs={f'{family}_{seed}':sha(checkpoint(family,seed)/'config.json')
                        for family in SPEC['policies'] for seed in SPEC['seeds']})


def save_trace(path, trace):
    # Map entries are exactly 0/1; uint8 storage is lossless. Other inputs stay float32.
    if not np.all((trace['map_channels']==0)|(trace['map_channels']==1)):
        raise RuntimeError('Map compression would be lossy')
    ti, ei = np.where(trace['active'])
    data = {k:v[ti,ei] for k,v in trace.items()}
    data['map_channels'] = data['map_channels'].astype(np.uint8)
    data['step_index'] = ti; data['episode_index'] = ei
    temporary = path.with_name(path.name+'.pending.npz')
    np.savez_compressed(temporary, **data)
    temporary.replace(path)


def observation_diagnostics(trace, condition, repeats, anchors):
    """Anchors are first actual visits to the identical non-time input (no synthetic states)."""
    pairs=[]; blocked=[]; full_repeats=0
    for i in range(16*repeats):
        local={}; seen_full=set()
        for t in range(int(trace['active'][:,i].sum())):
            if trace['forced'][t,i]: continue
            maps=trace['map_channels'][t,i]; numeric=trace['numeric_features'][t,i]
            goal=trace['goal_one_hot'][t,i]; probs=trace['probabilities'][t,i]
            non_time=np.delete(numeric,probe.TIME_INDEX)
            key=hashlib.sha256(maps.tobytes()+non_time.tobytes()+goal.tobytes()).hexdigest()
            full=hashlib.sha256(maps.tobytes()+numeric.tobytes()+goal.tobytes()).hexdigest()
            full_repeats+=int(full in seen_full);seen_full.add(full)
            point=dict(condition=condition,episode=i,tick=int(trace['tick'][t,i]),
                       probabilities=probs.tolist(),argmax=int(trace['mode_action'][t,i]),
                       position=trace['position'][t,i].tolist(),direction=int(trace['direction'][t,i]))
            for kind,reference in [('within_episode',local.get(key)),
                ('cross_delay',anchors.get((condition.split('_')[0],i,key)) if 'delay' in condition and not condition.endswith('0') else None)]:
                if reference is None or reference['tick']==point['tick']: continue
                # Map/goal/numeric equality is exact; also require same physical position/direction.
                assert reference['position']==point['position'] and reference['direction']==point['direction']
                pairs.append(dict(kind=kind,non_time_input_sha256=key,anchor=reference,current=point,
                    changed_argmax=reference['argmax']!=point['argmax'],
                    max_probability_difference=float(np.max(np.abs(np.array(reference['probabilities'])-probs)))))
            local.setdefault(key,point)
            if condition in ('fixed_delay0','mutant_delay0'):
                anchors.setdefault((condition.split('_')[0],i,key),point)
            if trace['blocked'][t,i] and t+1<64 and trace['active'][t+1,i]:
                diff=np.flatnonzero(numeric!=trace['numeric_features'][t+1,i]).tolist()
                maps_same=bool(np.array_equal(maps,trace['map_channels'][t+1,i]))
                goal_same=bool(np.array_equal(goal,trace['goal_one_hot'][t+1,i]))
                blocked.append(dict(episode=i,step=t+1,changed_numeric_features=[env.MODEL_FEATURE_NAMES[x] for x in diff],
                    map_unchanged=maps_same,goal_unchanged=goal_same,
                    only_time_changed=maps_same and goal_same and diff==[probe.TIME_INDEX],
                    next_argmax=int(trace['mode_action'][t+1,i]),argmax=int(trace['mode_action'][t,i])))
    counts={kind:dict(eligible_pairs=sum(p['kind']==kind for p in pairs),
                     changed_argmax_pairs=sum(p['kind']==kind and p['changed_argmax'] for p in pairs),
                     episodes=sorted({p['current']['episode'] for p in pairs if p['kind']==kind}),
                     layouts=sorted({p['current']['episode']//repeats for p in pairs if p['kind']==kind}))
            for kind in ('within_episode','cross_delay')}
    return dict(time_pairs=pairs,time_pair_counts=counts,
                time_pair_comparison_available=bool(pairs),
                blocked_input_changes=blocked,exact_full_input_repetitions=full_repeats)


def parity(records, expected):
    assert len(records)==len(expected)
    fields=('layout_index','repeat','success','length','wall_pass','wall_pass_count','beneficial_use',
            'first_wall_pass_step','dash_used','dash_count','clear_dash','clear_dash_count')
    for a,b in zip(records,expected):
        for field in fields:
            if a[field]!=b[field]: raise RuntimeError(f'Baseline parity failed: {field}: {a[field]} vs {b[field]}')
        assert abs(a['discounted_return']-b['discounted_return'])<1e-12
        if 'path' in b:
            assert len(a['path'])==len(b['path'])
            for x,y in zip(a['path'],b['path']):
                assert (x['action'],x['from_position'],x['to_position'])==(y['action'],y['from'],y['to'])


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if not any(d.platform=='gpu' for d in jax.devices()): raise RuntimeError('GPU entrypoint required')
    p=provenance(); path=ROOT/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==p
    else: write(path,p)
    capacity('start')
    network=gc.SpatialWallPassGCActorCritic(hidden_size=512)
    rollouts={selection:probe.make_rollout(network,selection=='sample',repeats)
              for selection,repeats in [('mode',1),('sample',8)]}
    for family in SPEC['policies']:
        for seed in SPEC['seeds']:
            name=f'{family}_{seed}'; folder=ROOT/name; folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():
                prior=json.loads((folder/'result.json').read_text())
                assert prior['source_state_sha256']==p['source_checkpoints'][name]
                assert prior['episodes']==1008
                assert prior['execution_sha']==p['execution_sha']
                assert all(sha(folder/file)==digest for file,digest in prior['artifacts_sha256'].items())
                continue
            capacity(name)
            source=checkpoint(family,seed)/'state.msgpack'; raw=serialization.msgpack_restore(source.read_bytes())
            assert int(raw['gc']['global_update'])==4608
            params=jax.tree.map(jnp.asarray,raw['gc']['train_state']['params']); del raw
            before=hashlib.sha256(serialization.to_bytes(params)).hexdigest()
            expected=baseline(family,seed); results={};diagnostics={}
            for selection,repeats in [('mode',1),('sample',8)]:
                anchors={};results[selection]={};diagnostics[selection]={}
                for condition in SPEC['conditions']:
                    fixed=condition.startswith('fixed');once=condition=='mutant_block_once'
                    delay=0 if once else int(condition[-1])
                    trace=jax.device_get(rollouts[selection](params,jnp.int32(seed),jnp.bool_(fixed),jnp.int32(delay),jnp.bool_(once)))
                    records=probe.records_from_trace(trace,repeats,delay)
                    if condition in ('fixed_delay0','mutant_delay0'):
                        parity(records,expected[condition.split('_')[0]][selection]['episode_records'])
                    if once: assert all(r['intervention_trigger_count']<=1 for r in records)
                    save_trace(folder/f'{condition}_{selection}.npz',trace)
                    diagnostics[selection][condition]=observation_diagnostics(trace,condition,repeats,anchors)
                    results[selection][condition]=dict(summary=probe.summarize(records),episodes=records,
                        original_episode_parity=True if condition in ('fixed_delay0','mutant_delay0') else None)
                    print(json.dumps(dict(policy=name,selection=selection,condition=condition,
                        success=results[selection][condition]['summary']['success_rate'])),flush=True)
            assert before==hashlib.sha256(serialization.to_bytes(params)).hexdigest()
            assert sha(source)==p['source_checkpoints'][name]
            write(folder/'diagnostics.json',diagnostics)
            write(folder/'result.json',dict(policy=family,seed=seed,episodes=1008,
                execution_sha=p['execution_sha'],
                artifacts_sha256={file.name:sha(file) for file in [*folder.glob('*.npz'),folder/'diagnostics.json']},
                source_state_sha256=p['source_checkpoints'][name],parameters_unchanged=True,
                natural_baseline_parity=True,results=results))
    cells={f'{f}_{s}':json.loads((ROOT/f'{f}_{s}'/'result.json').read_text()) for f in SPEC['policies'] for s in SPEC['seeds']}
    assert sum(c['episodes'] for c in cells.values())==15120
    assert provenance()==p
    capacity('complete')
    write(ROOT/'result.json',dict(status='complete',completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        execution_sha=p['execution_sha'],policies_completed=15,mode_episodes=1680,sample_episodes=13440,
        training_updates=0,optimizer_updates=0,cells=cells))


if __name__=='__main__': main()
