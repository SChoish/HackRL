#!/usr/bin/env python3
"""Bounded policy-blind generation; never silently relax split constraints."""
from collections import Counter
from datetime import datetime, timezone
import hashlib,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
import jax,numpy as np,scipy
from hackrl import spatial_layout_panel as p
from hackrl import spatial_multimap_panel as m

REPO=Path(__file__).resolve().parents[1]
SPEC_PATH=REPO/'docs/manifests/spatial_multimap_v1.json'
SPEC=json.loads(SPEC_PATH.read_text());ROOT=Path(SPEC['panel_root'])
PRIOR=Path('/raid/ext_csv/HackRL/runs/spatial_layout_transfer_v2/maps_locked.json')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,data):
    path=ROOT/name;tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('w') as f:json.dump(data,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def main():
    assert jax.default_backend()=='cpu'
    assert not ROOT.exists(),'Refuse overwrite/reselection'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    ROOT.mkdir(parents=True)
    start=time.time();utc=datetime.now(timezone.utc).isoformat()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    paths=[SPEC_PATH,REPO/'src/hackrl/spatial_multimap_panel.py',REPO/'src/hackrl/spatial_layout_panel.py',
           REPO/'src/hackrl/spatial_wall_pass.py',Path(__file__).resolve(),PRIOR,
           REPO/'docs/manifests/spatial_wall_pass_frozen_transfer_v1.json']
    hashes={str(f):sha(f) for f in paths}
    disk=dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
              projected_write_bytes=2**30,reserve_bytes=8*2**30,checkpoint_writes=0)
    assert disk['raid']['free']>disk['projected_write_bytes']+disk['reserve_bytes']
    write('design_manifest.json',dict(spec=SPEC,execution_sha=commit,input_sha256=hashes,storage=disk,
                                     policy_reads=0,policy_episodes=0,optimizer_updates=0))
    old=m.historical(REPO,PRIOR);excluded={m.wall_group(f) for f in old}
    excluded_geometry={p.canonical_hash(f,include_direction=False) for f in old}
    write('historical_exclusions.json',dict(record_count=len(old),distinct_wall_groups=len(excluded),
        records=[dict(index=i,wall_group=m.wall_group(f),geometry=p.canonical_hash(f,include_direction=False)) for i,f in enumerate(old)],
        record_order='original train/validation/test each16; prior6 fixture manifest order; prior32 locked order',
        wall_groups=sorted(excluded),geometry_hashes=sorted(excluded_geometry)))
    cfg=SPEC['generation'];quota=cfg['per_cell'];counts=Counter();rejected=Counter();seen=set();maps=[]
    rng=np.random.default_rng(cfg['seed'])
    with (ROOT/'candidates.jsonl').open('x') as log:
        for i in range(cfg['cap']):
            f=p.generate_candidate(rng,i);group=m.wall_group(f);split=m.split_for(group)
            geometry=p.canonical_hash(f,include_direction=False)
            rec=dict(candidate=i,wall_group=group,split=split,geometry=geometry);reason=None
            if group in excluded or geometry in excluded_geometry:reason='historical'
            elif geometry in seen:reason='duplicate'
            else:
                seen.add(geometry);answer=p.solve(f);cell,error=p.classify(answer)
                rec.update({k:answer[k] for k in ('L0','Lcross','A')})
                if cell is None or cell[0]!='short':reason=error or 'not_short'
                elif counts[split,cell[1]]>=quota[split]:reason='quota_full'
                else:
                    verification=p.verify(f,answer)
                    maps.append(dict(map_id=f'multimap_{i:05d}',split=split,value_cell=cell[1],
                        wall_group_sha256=group,geometry_sha256=geometry,fixture=f,oracle=answer,verification=verification))
                    counts[split,cell[1]]+=1
            rec['disposition']=reason or 'retained';rejected[rec['disposition']]+=1
            log.write(json.dumps(rec)+'\n')
            if (i+1)%1000==0 or len(maps)==90:
                log.flush();os.fsync(log.fileno())
                progress=dict(candidates=i+1,selected=len(maps),counts={'/'.join(k):v for k,v in counts.items()},elapsed_seconds=time.time()-start)
                write('progress.json',progress);write('partial_candidates.json',maps);print(json.dumps(progress),flush=True)
            if len(maps)==90:break
    complete=len(maps)==90
    groups=m.validate_partition(maps,quota,excluded) if complete else None
    composition={}
    for split in quota:
        for cell in cfg['value_cells']:
            xs=[x for x in maps if x['split']==split and x['value_cell']==cell]
            composition[split+'/'+cell]=dict(count=len(xs),families=dict(Counter(x['fixture']['family'] for x in xs)),
                L0=[x['oracle']['L0'] for x in xs],A=[x['oracle']['A'] for x in xs])
    result=dict(status='panel_locked' if complete else 'generation_incomplete_gate_closed',candidates=i+1,
                selected=len(maps),group_counts=groups,composition=composition,rejections=dict(rejected),
                policy_reads=0,policy_episodes=0,optimizer_updates=0,elapsed_seconds=time.time()-start)
    write('partial_candidates.json',maps);write('feasibility_result.json',result)
    if complete:write('maps_locked.json',dict(maps=maps,composition=composition))
    assert {str(f):sha(f) for f in paths}==hashes
    outputs={f.name:sha(f) for f in ROOT.iterdir() if f.is_file()}
    write('integrity.json',dict(execution_sha=commit,input_before=hashes,input_after=hashes,output_sha256=outputs))
    write('computation_manifest.json',dict(schema_version=1,claim_id='spatial_multimap_v1_panel_feasibility',
        repository=dict(commit=commit,dirty=False),command=' '.join([sys.executable,*sys.argv]),
        environment=dict(software=[platform.python_version(),'NumPy '+np.__version__,'SciPy '+scipy.__version__,'JAX '+jax.__version__],hardware=platform.machine()+' CPU'),
        mathematics=dict(assertion_tested='Construct balanced36/18/36 short-route maps with wall-group-disjoint splits',
            coefficient_domain='exact integer transitions/unit-cost BFS',conventions=json.dumps(cfg,sort_keys=True),
            inputs=list(hashes),bounds=dict(candidates=cfg['cap'],seed=cfg['seed'],horizon=64),
            non_claims=['Not learnability or universal generator feasibility']),
        randomness=dict(used=True,generator='NumPy PCG64',seed=cfg['seed']),
        run=dict(started_at=utc,runtime_seconds=time.time()-start,exit_status=0),
        outputs=[dict(path=k,sha256=v) for k,v in outputs.items()],
        checks=['D4 wall-group and historical exclusion','All-action kernel/oracle comparison for every selected map','Oracle path replay0/12','Input hashes unchanged'],
        result=result['status'],residual_risks=['Finite generator','Legacyv1 plus explicit before/after input hashes']))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
