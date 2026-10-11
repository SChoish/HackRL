#!/usr/bin/env python3
"""Lock the existing32 maps on CPU, then run only the authorized frozen matrix."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from hackrl import spatial_architecture as a
from hackrl import spatial_layout_panel as panel_api
from hackrl import spatial_layout_eval as evaluate
from hackrl import spatial_wall_pass as env
from hackrl.spatial_recovery_probe import records_from_trace, summarize
from run_spatial_history_probe import sha, tree_hash, write, provenance as source_provenance, count

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/spatial_layout_transfer_v2.json'
SPEC=json.loads(SPEC_PATH.read_text())
ROOT=Path(SPEC['root'])
ORIGINAL=Path(SPEC['original_panel'])


def journal(event,**values):
    with (ROOT/'operations.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=event,utc=time.time(),**values))+'\n');f.flush();os.fsync(f.fileno())


def capacity():
    s=SPEC['storage'];projected=s['projected_artifact_bytes'];reserve=max(s['minimum_reserve_bytes'],int(.2*projected))
    record=dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        existing_run_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        source_checkpoint_bytes=sum(p.stat().st_size for p in Path(SPEC['source_root']).rglob('state.msgpack')),
        chosen_filesystem='/raid/ext_csv',projected_remaining_write_bytes=projected,safety_reserve_bytes=reserve,
        checkpoint_writes=0)
    assert record['raid']['free']>=projected+reserve
    journal('capacity',**record)


def execution_sha():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()


def lock_panel():
    assert jax.default_backend()=='cpu'
    assert not ROOT.exists(), 'New version must not overwrite an existing run'
    ROOT.mkdir(parents=True)
    capacity()
    old_integrity=json.loads((ORIGINAL/'integrity.json').read_text())
    for name,digest in old_integrity['output_sha256'].items():assert sha(ORIGINAL/name)==digest,name
    assert sha(ORIGINAL/'partial_candidates.json')==SPEC['original_partial_sha256']
    maps=json.loads((ORIGINAL/'partial_candidates.json').read_text())
    assert len(maps)==32
    excluded=json.loads((ORIGINAL/'excluded_historical_maps.json').read_text())
    historical={x['canonical'] for x in excluded};historical_geometry={x['geometry'] for x in excluded}
    seen=set();geometries=set();composition={}
    for m in maps:
        f=m['fixture'];digest=panel_api.canonical_hash(f);geo=panel_api.canonical_hash(f,include_direction=False)
        assert digest==m['canonical_sha256'] and geo==m['geometry_sha256']
        assert digest not in historical|seen and geo not in historical_geometry|geometries
        seen.add(digest);geometries.add(geo)
        assert panel_api.solve(f)==m['oracle']
        assert panel_api.classify(m['oracle'])[0]==tuple(m['cell'])
        cell='/'.join(m['cell'])
        composition.setdefault(cell,[]).append(m)
    assert set(composition)=={'short/large_gain','short/small_gain','short/harmful','medium/large_gain'}
    report={}
    for cell,rows in composition.items():
        assert len(rows)==8 and Counter(r['fixture']['family'] for r in rows)==dict.fromkeys(panel_api.FAMILIES,2)
        report[cell]=dict(maps=8,family_counts=dict(Counter(r['fixture']['family'] for r in rows)),
            map_ids=[r['map_id'] for r in rows],L0=[r['oracle']['L0'] for r in rows],
            Lcross=[r['oracle']['Lcross'] for r in rows],A=[r['oracle']['A'] for r in rows],
            mean_L0=float(np.mean([r['oracle']['L0'] for r in rows])),
            group='primary24' if cell.startswith('short/') else 'extension8')
    write(ROOT/'design_manifest.json',dict(spec=SPEC,execution_sha=execution_sha(),spec_sha256=sha(SPEC_PATH),
        original_generation_sha=old_integrity['execution_sha'],original_integrity_sha256=sha(ORIGINAL/'integrity.json'),
        original_partial_sha256=sha(ORIGINAL/'partial_candidates.json'),locked_before_checkpoint_access=True))
    write(ROOT/'maps_locked.json',dict(maps=maps,composition=report,selection='All32 existing candidates unchanged, no new maps'))
    write(ROOT/'oracle_paths.json',{m['map_id']:m['oracle'] for m in maps})
    write(ROOT/'panel_integrity.json',dict(execution_sha=execution_sha(),checkpoint_reads_before_lock=0,
        files={name:sha(ROOT/name) for name in ('design_manifest.json','maps_locked.json','oracle_paths.json')},
        original_partial_sha256=SPEC['original_partial_sha256']))
    journal('panel_locked',maps=32,composition=report)
    print(json.dumps(report,indent=2))


def validate_lock():
    lock=json.loads((ROOT/'panel_integrity.json').read_text())
    assert lock['execution_sha']==execution_sha()
    for name,digest in lock['files'].items():assert sha(ROOT/name)==digest
    assert sha(ORIGINAL/'partial_candidates.json')==lock['original_partial_sha256']==SPEC['original_partial_sha256']
    assert json.loads((ROOT/'design_manifest.json').read_text())['spec_sha256']==sha(SPEC_PATH)
    return lock,json.loads((ROOT/'maps_locked.json').read_text())['maps']


def load_policy(source):
    cp=Path(source['path'])
    for name,digest in source['hashes'].items():assert sha(cp/name)==digest
    raw=serialization.msgpack_restore((cp/'state.msgpack').read_bytes())
    assert int(raw['gc']['global_update'])==4608
    for actual,name in ((raw['gc']['train_state']['step'],'ppo_optimizer_steps'),(raw['leo']['step'],'teacher_optimizer_steps')):
        assert int(actual)==source['counters'][name]
    return raw,jax.tree.map(jnp.asarray,raw['gc']['train_state']['params'])


def smoke(arch,source,functions,evidence):
    dest=ROOT/f'smoke_{arch}.json'
    receipt=ROOT/f'smoke_{arch}_integrity.json'
    if receipt.exists():
        assert json.loads(receipt.read_text())['sha256']==sha(dest)
        saved=json.loads(dest.read_text())
        assert saved['execution_sha']==evidence['execution_sha'] and saved['source']==source
        return saved
    raw,params=load_policy(source);before=tree_hash(raw)
    initial=jax.vmap(lambda i:env.make_spatial_wall_pass_state(i,split='validation'))(jnp.arange(16))
    expected=json.loads((Path(source['path'])/'evaluation.json').read_text())['results']
    n=0;ticks=0
    for selection in ('mode','sample') if arch in ('current_mlp','history_mlp') else ('mode',):
        repeats=8 if selection=='sample' else 1
        for delay in SPEC['delays']:
            for kernel in ('fixed','mutant'):
                tr=jax.device_get(functions[selection](params,jnp.int32(132),kernel=='fixed',jnp.int32(delay),initial,jnp.arange(16)))
                rows=records_from_trace(tr,repeats,delay)
                if selection=='sample':
                    for r in rows:r.pop('path')
                assert rows==expected[selection][f'{kernel}_delay{delay}']['episodes'],(arch,selection,delay,kernel)
                assert summarize(rows)==expected[selection][f'{kernel}_delay{delay}']['summary']
                n+=len(rows);ticks+=sum(r['length'] for r in rows)
    assert tree_hash(raw)==before and tree_hash(params)==tree_hash(raw['gc']['train_state']['params'])
    record=dict(execution_sha=evidence['execution_sha'],source=source,episodes=n,active_ticks=ticks,parity=True)
    write(dest,record);write(receipt,dict(sha256=sha(dest)))
    journal('smoke_complete',architecture=arch,episodes=n)
    return record


def run_policy(arch,seed,q,source,functions,states,maps,evidence):
    key=f'{arch}_{seed}_{q}';dest=ROOT/key;dest.mkdir(exist_ok=True)
    receipt=dest/'verified_complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        assert saved['execution_sha']==evidence['execution_sha'] and saved['source']==source
        assert saved['panel_hash']==evidence['panel_hash'] and saved['result_sha256']==sha(dest/'result.json')
        return json.loads((dest/'result.json').read_text())
    capacity();journal('policy_attempt',policy=key)
    raw,params=load_policy(source);before=tree_hash(raw);param_before=tree_hash(params)
    episodes=[]
    selections=('mode','sample') if q==10 and arch in ('current_mlp','history_mlp') else ('mode',)
    for selection in selections:
        repeats=8 if selection=='sample' else 1
        for delay in SPEC['delays']:
            for kernel in ('fixed','mutant'):
                for offset in (0,16):
                    journal('rollout_attempt',policy=key,selection=selection,delay=delay,kernel=kernel,offset=offset,episodes=16*repeats)
                    initial=jax.tree.map(lambda x:x[offset:offset+16],states)
                    tr=jax.device_get(functions[selection](params,jnp.int32(seed),kernel=='fixed',jnp.int32(delay),initial,jnp.arange(offset,offset+16)))
                    rows=evaluate.decorate(tr,maps[offset:offset+16],repeats,delay,kernel,selection)
                    episodes.extend(rows)
                    journal('rollout_complete',policy=key,episodes=len(rows),active_ticks=sum(r['length'] for r in rows))
    assert tree_hash(raw)==before and tree_hash(params)==param_before
    for name,digest in source['hashes'].items():assert sha(Path(source['path'])/name)==digest
    p=raw['gc']['train_state']['params']['params']
    result=dict(architecture=arch,seed=seed,fixed_percent=q,source=source,episodes=episodes,
        optimizer_updates=0,normalization_updates=0,frozen_raw_sha256=before,
        parameters=dict(actor=count(p['actor']),critic=count({k:v for k,v in p.items() if k!='actor'}),teacher=count(raw['leo']['params'])),
        accounting={sel:dict(episodes=sum(r['selection']==sel for r in episodes),
            active_ticks=sum(r['length'] for r in episodes if r['selection']==sel),
            forced_wait_ticks=sum(r['delay'] for r in episodes if r['selection']==sel)) for sel in selections})
    write(dest/'result.json',result)
    write(receipt,dict(execution_sha=evidence['execution_sha'],source=source,panel_hash=evidence['panel_hash'],result_sha256=sha(dest/'result.json')))
    journal('policy_complete',policy=key,episodes=len(episodes));print('Completed '+key,flush=True)
    return result


def run():
    assert jax.default_backend()=='gpu'
    lock,maps=validate_lock();capacity()
    # This is the first checkpoint access: all32 maps/config/oracles are locked.
    provenance=source_provenance()
    evidence=dict(execution_sha=execution_sha(),panel_hash=lock['files']['maps_locked.json'],sources=provenance['sources'])
    if (ROOT/'source_integrity.json').exists():assert json.loads((ROOT/'source_integrity.json').read_text())==evidence
    else:write(ROOT/'source_integrity.json',evidence)
    states=jax.tree.map(lambda *x:jnp.stack(x),*[panel_api.kernel_state(m['fixture']) for m in maps])
    functions={}
    smokes=[]
    for arch in SPEC['architectures']:
        model=a.ArchitectureActorCritic(arch)
        functions[arch]={s:evaluate.make_rollout(model,s=='sample',8 if s=='sample' else 1) for s in ('mode','sample')}
        smokes.append(smoke(arch,evidence['sources'][f'{arch}_132_10'],functions[arch],evidence))
    assert sum(s['episodes'] for s in smokes)==1280
    write(ROOT/'smoke_complete.json',dict(episodes=1280,active_ticks=sum(s['active_ticks'] for s in smokes),parity=True))
    results=[]
    for arch in SPEC['architectures']:
        for seed in SPEC['seeds']:
            for q in SPEC['fixed_percent']:
                results.append(run_policy(arch,seed,q,evidence['sources'][f'{arch}_{seed}_{q}'],functions[arch],states,maps,evidence))
    counts={s:sum(r['accounting'].get(s,{}).get('episodes',0) for r in results) for s in ('mode','sample')}
    assert counts==dict(mode=5120,sample=10240)
    ticks=sum(c['active_ticks'] for r in results for c in r['accounting'].values())
    assert ticks<=983040
    # Portable episode table, with mode trajectories kept in per-policy results.
    with (ROOT/'episode_results.jsonl.pending').open('w') as f:
        for r in results:
            for e in r['episodes']:
                f.write(json.dumps({k:r[k] for k in ('architecture','seed','fixed_percent')}|
                    {k:v for k,v in e.items() if k!='path'})+'\n')
        f.flush();os.fsync(f.fileno())
    (ROOT/'episode_results.jsonl.pending').replace(ROOT/'episode_results.jsonl')
    aggregate=dict(status='complete',execution_sha=evidence['execution_sha'],panel_hash=evidence['panel_hash'],
        counts=counts,active_ticks=ticks,smoke_episodes=1280,optimizer_updates=0,normalization_updates=0,
        seed_rows=[row for r in results for row in evaluate.seed_aggregates(r)],
        primary_comparison=evaluate.primary_bootstrap(results,maps),
        result_files={f"{r['architecture']}_{r['seed']}_{r['fixed_percent']}":sha(ROOT/f"{r['architecture']}_{r['seed']}_{r['fixed_percent']}"/'result.json') for r in results})
    write(ROOT/'aggregate_result.json',aggregate)
    validate_lock()
    for source in evidence['sources'].values():
        for name,digest in source['hashes'].items():assert sha(Path(source['path'])/name)==digest
    write(ROOT/'verified_complete.json',dict(execution_sha=evidence['execution_sha'],episodes=15360,
        aggregate_sha256=sha(ROOT/'aggregate_result.json'),episode_table_sha256=sha(ROOT/'episode_results.jsonl'),completed_utc=time.time()))
    journal('verified_complete',episodes=15360)
    print('VERIFIED COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lock-panel',action='store_true');args=parser.parse_args()
    if args.lock_panel:lock_panel()
    else:
        try:run()
        except Exception:
            write(ROOT/'failed.json',dict(utc=time.time(),error=traceback.format_exc()))
            raise
