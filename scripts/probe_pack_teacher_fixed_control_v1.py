#!/usr/bin/env python3
"""Read-only terminal probe; no learner imports and no queue relaunch on finish."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def status(root,manifest_sha256):
    root=Path(root)
    assert sha(root/'run_manifest.json')==manifest_sha256,'wrong manifest'
    m=read(root/'run_manifest.json');assert m['experiment_id']=='pack_teacher_fixed_control_v1'
    complete=root/'complete.json';failed=root/'failure.json'
    assert not (complete.exists() and failed.exists()),'conflicting terminal records'
    if failed.exists():
        f=read(failed);assert 'traceback' in f or 'reason' in f
        return {'terminal':True,'outcome':'failed','science_complete':False,'receipt_sha256':sha(failed)}
    if not complete.exists():return {'terminal':False,'outcome':'pending'}
    c=read(complete);r=read(root/'result.json')
    assert sha(root/'result.json')==c['result_sha256']
    assert r['complete'] and r['new_evaluation_episodes']==1440 and r['new_collector_slots']==671088640
    assert len(m['jobs'])==5 and set(c['final_checkpoint_sha256'])=={j['id'] for j in m['jobs']}
    for j in m['jobs']:
        cell=root/j['condition']/f"seed{j['seed']}";checkpoint=cell/'checkpoints/adapt_4096'
        s=read(cell/'summary.json');assert s['complete'] and s['global_update']==4608 and s['collector_slots']==134217728
        assert sha(checkpoint/'state.msgpack')==s['checkpoint_sha256']==c['final_checkpoint_sha256'][j['id']]
        files=read(checkpoint/'integrity.json')['checkpoint_files']
        assert set(files)=={'state.msgpack','config.json','metadata.json','arm.json'}
        for name,h in files.items():assert sha(checkpoint/name)==h
    assert len(c['evaluation_sha256'])==len(c['trace_sha256'])==20
    for section in ['evaluation_sha256','trace_sha256']:
        for p,h in c[section].items():
            assert Path(p).resolve().parent==(root/'evaluation').resolve()
            assert sha(p)==h
    return {'terminal':True,'outcome':'complete','science_complete':True,'verified_final_checkpoints':5,
            'new_evaluation_episodes':1440,'receipt_sha256':sha(complete)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--manifest-sha256',required=True);a=p.parse_args()
    try:r=status(a.root,a.manifest_sha256)
    except (AssertionError,OSError,ValueError,KeyError,TypeError) as e:
        print(json.dumps({'terminal':False,'error':str(e)}),flush=True);return 2
    print(json.dumps(r),flush=True);return 0 if r['terminal'] else 10


if __name__=='__main__':raise SystemExit(main())
