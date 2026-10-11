#!/usr/bin/env python3
"""Bounded CPU-only panel feasibility run. Never imports/loads a policy."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import jax
import numpy as np
import scipy
from hackrl import spatial_layout_panel as p
from hackrl import spatial_wall_pass as env

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/spatial_layout_panel_v1.json'
SPEC=json.loads(SPEC_PATH.read_text())
ROOT=Path(SPEC['output_root'])


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(name,value):
    path=ROOT/name
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False)
        f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def capacity():
    record=dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        existing_run_bytes=sum(f.stat().st_size for f in ROOT.rglob('*') if f.is_file()),
        additional_write_allowance=SPEC['storage']['artifact_allowance_bytes'],
        reserve_bytes=SPEC['storage']['reserve_bytes'],chosen_filesystem='/raid/ext_csv',checkpoint_writes=0)
    assert record['raid']['free']>record['additional_write_allowance']+record['reserve_bytes']
    with (ROOT/'capacity.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=time.time(),**record))+'\n')
    return record


def exclusions():
    entries=[]
    for split in ('train','validation','test'):
        for i in range(16):
            f=p.from_state(env.make_spatial_wall_pass_state(i,split=split))
            entries.append(dict(source=f'original/{split}/{i}',fixture=f,
                canonical=p.canonical_hash(f),geometry=p.canonical_hash(f,include_direction=False)))
    old=json.loads((REPO/'docs/manifests/spatial_wall_pass_frozen_transfer_v1.json').read_text())
    for f in old['fixtures']:
        layout=p.fixture(f['walls'],f['camp'],f['start'],f['item'])
        entries.append(dict(source='historical_transfer/'+f['id'],fixture=layout,
            canonical=p.canonical_hash(layout),geometry=p.canonical_hash(layout,include_direction=False)))
    return entries


def main():
    assert jax.default_backend()=='cpu'
    assert SPEC['authorization']==dict(panel_generation=True,policy_evaluation=False,training=False)
    if ROOT.exists() and any(ROOT.iterdir()):
        raise RuntimeError('Preserve existing panel attempt; refuse overwrite or automatic retuning')
    ROOT.mkdir(parents=True,exist_ok=True)
    started=time.time();started_utc=datetime.now(timezone.utc).isoformat()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    paths=['src/hackrl/spatial_layout_panel.py','src/hackrl/spatial_wall_pass.py',
           'scripts/build_spatial_layout_panel.py','docs/manifests/spatial_layout_panel_v1.json',
           'docs/manifests/spatial_wall_pass_frozen_transfer_v1.json']
    before={name:sha(REPO/name) for name in paths}
    old_kernel=subprocess.check_output(['git','show',f"{SPEC['source_kernel_sha']}:src/hackrl/spatial_wall_pass.py"],cwd=REPO)
    assert hashlib.sha256(old_kernel).hexdigest()==before['src/hackrl/spatial_wall_pass.py']
    write('design_manifest.json',dict(spec=SPEC,execution_sha=commit,input_sha256=before,
          command=[sys.executable,*sys.argv],started_utc=started_utc,
          software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,jax=jax.__version__),
          storage=capacity(),policy_checkpoint_reads=0,policy_episodes=0,optimizer_updates=0))
    old=exclusions()
    write('excluded_historical_maps.json',old)
    excluded={r['canonical'] for r in old};excluded_geometry={r['geometry'] for r in old}
    benchmark=p.from_state(env.make_spatial_wall_pass_state(0))
    known=p.solve(benchmark)
    assert (known['L0'],known['Lcross'])==(14,8)
    write('reference_verification.json',dict(oracle=known,verification=p.verify(benchmark,known)))
    rng=np.random.default_rng(SPEC['generator_seed'])
    quotas={(length,value,family):[] for length in SPEC['length_cells'] for value in SPEC['value_cells'] for family in SPEC['geometry_families']}
    seen=set();seen_geometry=set();retained=[];rejected=Counter();hist=Counter();extrema={}
    log=ROOT/'candidates.jsonl'
    with log.open('x') as output:
        for index in range(SPEC['candidate_cap']):
            layout=p.generate_candidate(rng,index)
            digest=p.canonical_hash(layout);geometry=p.canonical_hash(layout,include_direction=False)
            record=dict(candidate_index=index,family=layout['family'],canonical=digest,geometry=geometry)
            reason=None
            if digest in excluded or geometry in excluded_geometry:
                reason='historical_equivalent'
            elif digest in seen or geometry in seen_geometry:
                reason='duplicate_candidate'
            else:
                seen.add(digest);seen_geometry.add(geometry)
                answer=p.solve(layout)
                record.update({k:answer[k] for k in ('L0','Lcross','A')})
                cell,reason=p.classify(answer)
                stats=extrema.setdefault(layout['family'],dict(min_L0=None,max_L0=None,min_A=None,max_A=None,solved=0))
                stats['solved']+=1
                for key in ('L0','A'):
                    value=answer[key]
                    if value is not None:
                        for side,fn in (('min',min),('max',max)):
                            name=side+'_'+key;stats[name]=value if stats[name] is None else fn(stats[name],value)
                if cell:
                    quota=(*cell,layout['family'])
                    hist['/'.join(quota)]+=1
                    record['cell']=list(cell)
                    if len(quotas[quota])>=2:
                        reason='quota_full'
                    else:
                        verification=p.verify(layout,answer)
                        selected=dict(map_id=f'candidate_{index:05d}',candidate_index=index,
                            cell=list(cell),canonical_sha256=digest,geometry_sha256=geometry,
                            fixture=layout,oracle=answer,verification=verification)
                        retained.append(selected);quotas[quota].append(selected['map_id'])
            record['disposition']=reason or 'retained'
            rejected[record['disposition']]+=1
            output.write(json.dumps(record,sort_keys=True)+'\n')
            if (index+1)%1000==0 or len(retained)==72:
                output.flush();os.fsync(output.fileno());capacity()
                progress=dict(candidates=index+1,retained=len(retained),elapsed_seconds=time.time()-started,
                    quotas={'/'.join(k):len(v) for k,v in quotas.items()},rejections=dict(rejected),extrema=extrema)
                write('progress.json',progress)
                write('partial_candidates.json',retained)
                print(json.dumps(progress),flush=True)
            if len(retained)==72:
                break
        output.flush();os.fsync(output.fileno())
    complete=len(retained)==72
    missing={'/'.join(k):2-len(v) for k,v in quotas.items() if len(v)<2}
    result=dict(status='panel_locked' if complete else 'incomplete_panel_gate_closed',
        candidates_examined=index+1,selected_maps=len(retained),target_maps=72,missing_subcells=missing,
        quota_counts={'/'.join(k):len(v) for k,v in quotas.items()},eligible_candidate_counts=dict(hist),
        rejections=dict(rejected),family_extrema=extrema,policy_evaluation_authorized=False,
        policy_checkpoint_reads=0,policy_episodes=0,optimizer_updates=0,
        non_claim='Missing cells are not a proof of impossibility beyond the fixed tested generator.')
    write('partial_candidates.json',retained)
    write('oracle_paths.json',{m['map_id']:m['oracle'] for m in retained})
    if complete:
        write('maps_locked.json',dict(spec_sha256=sha(SPEC_PATH),seed=SPEC['generator_seed'],maps=retained))
    write('feasibility_result.json',result)
    after={name:sha(REPO/name) for name in paths}
    assert before==after
    outputs={x.name:sha(x) for x in ROOT.iterdir() if x.is_file() and not x.name.endswith('.pending')}
    write('integrity.json',dict(execution_sha=commit,input_before=before,input_after=after,output_sha256=outputs,
          kernel_sha=SPEC['source_kernel_sha'],policy_checkpoint_reads=0,policy_episodes=0,optimizer_updates=0,
          normalization_updates=0,elapsed_seconds=time.time()-started))
    # Portable v1 computation schema; own integrity record supplies additional
    # before/after input hashes. This is finite evidence, never a theorem proof.
    evidence=dict(schema_version=1,claim_id='spatial_layout_panel_v1_feasibility',
        repository=dict(commit=commit,dirty=False),command=' '.join([sys.executable,*sys.argv]),
        environment=dict(software=[f'Python {platform.python_version()}',f'NumPy {np.__version__}',
            f'SciPy {scipy.__version__}',f'JAX {jax.__version__}'],hardware=platform.machine()+' CPU'),
        mathematics=dict(assertion_tested='Fill all36 registered family/length/value subcells twice under bounded generation',
            coefficient_domain='Exact integer graph transitions and unit-cost BFS; float64 discount reporting only',
            conventions=SPEC['oracle'],inputs=paths,bounds=dict(candidates=50000,seed=20261012,horizon=64,paths_max=48),
            non_claims=SPEC['non_claims']),randomness=dict(used=True,generator='NumPy default_rng PCG64',seed=20261012),
        run=dict(started_at=started_utc,runtime_seconds=time.time()-started,exit_status=0),
        outputs=[dict(path=name,sha256=digest) for name,digest in outputs.items()],
        checks=['Known14/8 benchmark','All23 kernel actions on every retained oracle state',
                'Oracle path kernel replay at0/12 waits','D4/translation and historical deduplication',
                'Source hashes unchanged; no policies loaded'],result=result['status'],
        residual_risks=['Finite generator only','Original48-cell support retained; no larger floor tested',
                        'Legacy v1 manifest complemented by local integrity input hashes'])
    write('computation_manifest.json',evidence)
    write('generation_complete.json',dict(status=result['status'],execution_sha=commit,
          result_sha256=sha(ROOT/'feasibility_result.json'),policy_gate_open=False))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
