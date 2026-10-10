"""Stdlib-only completion gate: code 3 means not yet complete, not success."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/raid/ext_csv/HackRL/runs/spatial_recovery_probe_v1')


def verify(root=ROOT):
    if not (root/'result.json').exists(): return False
    result=json.loads((root/'result.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    assert result['status']=='complete' and result['policies_completed']==15
    assert result['mode_episodes']==1680 and result['sample_episodes']==13440
    assert result['training_updates']==result['optimizer_updates']==0
    assert result['execution_sha']==manifest['execution_sha']
    expected={f'{f}_{s}' for f in manifest['spec']['policies'] for s in manifest['spec']['seeds']}
    assert set(result['cells'])==expected
    for name in expected:
        folder=root/name;cell=json.loads((folder/'result.json').read_text())
        assert cell==result['cells'][name] and cell['episodes']==1008
        assert cell['parameters_unchanged'] and cell['natural_baseline_parity']
        for selection in ('mode','sample'):
            for condition in ('fixed_delay0','mutant_delay0'):
                assert cell['results'][selection][condition]['original_episode_parity'] is True
        assert cell['source_state_sha256']==manifest['source_checkpoints'][name]
        assert len(cell['artifacts_sha256'])==15
        for file,digest in cell['artifacts_sha256'].items():
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest
    return True


if __name__=='__main__':
    raise SystemExit(0 if verify() else 3)
