#!/usr/bin/env python3
"""No accelerator imports: verify full checkpoint and evaluation completion."""
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
spec=json.loads((REPO/'docs/manifests/spatial_random_delay_v1.json').read_text())
root=Path(spec['run_root'])
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='complete' and result['jobs_completed']==10
    assert result['collector_slots']==1342177280
    episodes=0
    for seed in spec['seeds']:
        for q in spec['fixed_probabilities']:
            cell=root/f'fixed_{round(q*100):02d}pct_seed_{seed}'
            summary=json.loads((cell/'summary.json').read_text())
            assert summary['status']=='complete' and summary['updates']==4096
            assert summary['exposure']['collector_slots']==134217728
            for u in spec['evaluation']['updates']:
                cp=cell/'checkpoints'/f'adapt_{u}'
                m=json.loads((cp/'metadata.json').read_text())
                assert m['state_sha256']==sha(cp/'state.msgpack')
                assert m['execution_sha']==result['execution_sha']
                assert m['seed']==seed and m['q']==q
                assert json.loads((cp/'evaluation_integrity.json').read_text())['sha256']==sha(cp/'evaluation.json')
                e=json.loads((cp/'evaluation.json').read_text())
                assert e['update']==u
                for mode,n in [('mode',16),('sample',128)]:
                    assert len(e['results'][mode])==33
                    for rec in e['results'][mode].values():
                        assert len(rec['episodes'])==n
                        episodes+=n
    assert episodes==380160
    for seed in spec['seeds']:
        for family in ('mutant_only','mixed_10pct'):
            e=json.loads((root/'baselines'/f'{family}_{seed}.json').read_text())
            assert e['natural_parity'] and e['execution_sha']==result['execution_sha']
            for mode,n in [('mode',16),('sample',128)]:
                assert len(e['results'][mode])==33
                for rec in e['results'][mode].values():
                    assert len(rec['episodes'])==n
                    episodes+=n
    assert episodes==427680
    print('Verified ten jobs, 1,342,177,280 collection slots, 427,680 evaluation episodes.')
if __name__=='__main__': check()
