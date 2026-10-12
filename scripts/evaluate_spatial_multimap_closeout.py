#!/usr/bin/env python3
"""One frozen train-map evaluation; no optimizer or checkpoint-writing calls."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import jax
import numpy as np

import run_spatial_multimap as original
from hackrl.dual_leo import load_dual_checkpoint
from run_spatial_history_probe import sha, tree_hash, write

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / 'docs/manifests/spatial_multimap_closeout_v1.json'
SPEC = json.loads(SPEC_PATH.read_text())
SOURCE = Path(SPEC['source_root'])
ROOT = Path(SPEC['output_root'])


def aggregate(results, maps):
    """Assert equal map weights and retain per-map scores, not only pooled rates."""
    output = {}
    for selector, repeats in (('mode', 1), ('sample', 8)):
        assert set(results[selector]) == {'fixed_delay0'}
        episodes = results[selector]['fixed_delay0']['episodes']
        assert len(episodes) == len(maps) * repeats
        expected = {(m['map_id'], repeat) for m in maps for repeat in range(repeats)}
        assert {(e['map_id'], e['repeat']) for e in episodes} == expected
        by_map = []
        for m in maps:
            rows = [e for e in episodes if e['map_id'] == m['map_id']]
            assert len(rows) == repeats
            by_map.append(dict(map_id=m['map_id'], value_cell=m['value_cell'],
                family=m['fixture']['family'], normal_length=m['oracle']['L0'],
                episodes=repeats, successes=sum(bool(e['success']) for e in rows),
                success_rate=float(np.mean([e['success'] for e in rows])),
                discounted_return=float(np.mean([e['discounted_return'] for e in rows]))))
        rate = float(np.mean([x['success_rate'] for x in by_map]))
        assert abs(rate - results[selector]['fixed_delay0']['summary']['success_rate']) < 1e-12
        strata = {}
        for field in ('value_cell', 'family'):
            strata[field] = {v:dict(maps=len([x for x in by_map if x[field] == v]),
                success_rate=float(np.mean([x['success_rate'] for x in by_map if x[field] == v])))
                for v in sorted({x[field] for x in by_map})}
        output[selector] = dict(success_rate=rate, episodes=len(episodes),
            successes=sum(x['successes'] for x in by_map), per_map=by_map, strata=strata)
    return output


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'verified_complete.json').exists():
        receipt = json.loads((ROOT / 'verified_complete.json').read_text())
        assert receipt['aggregate_sha256'] == sha(ROOT / 'aggregate_result.json')
        return
    assert jax.default_backend() == 'gpu'
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip()
    execution_sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    old = json.loads((SOURCE / 'run_manifest.json').read_text())
    assert old['execution_sha'] == SPEC['training_sha']
    for name, expected in old['files'].items():
        assert sha(REPO / name) == expected, name
    assert sha(original.PANEL / 'maps_locked.json') == old['panel_sha256']
    gate = json.loads((SOURCE / 'development_gate.json').read_text())
    assert gate['passed'] is False and gate['test_episodes'] == 0
    assert not (SOURCE / 'main').exists() and not list(SOURCE.rglob('final_test.json'))
    source_hashes = {str(p):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    capacity = dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),
        raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
        source_bytes=sum(p.stat().st_size for p in SOURCE.rglob('*') if p.is_file()),
        existing_output_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
        projected_additional_bytes=256*1024**2, reserve_bytes=8*1024**3,
        chosen_filesystem='/raid/ext_csv', checkpoint_writes=0)
    assert capacity['raid']['free'] > capacity['projected_additional_bytes'] + capacity['reserve_bytes']
    manifest = dict(spec=SPEC, execution_sha=execution_sha, script_sha256=sha(__file__),
        spec_sha256=sha(SPEC_PATH), source_hashes=source_hashes)
    if (ROOT / 'run_manifest.json').exists():
        assert json.loads((ROOT / 'run_manifest.json').read_text()) == manifest
    else:
        write(ROOT / 'run_manifest.json', manifest)
    write(ROOT / 'capacity.json', capacity)
    train_maps, train_states = original.maps_for('train')
    # Only metadata of development maps is needed: its policy evaluations already exist.
    dev_maps = [m for m in json.loads((original.PANEL / 'maps_locked.json').read_text())['maps'] if m['split'] == 'development']
    assert len(train_maps) == 36 and len(dev_maps) == 18
    rows = []
    new_episodes = 0
    for seed in SPEC['seeds']:
        cp = SOURCE / 'development' / f'seed_{seed}' / 'pretraining/checkpoints/update_512'
        c = original.config(seed)
        meta = original.check(cp, c, None, old)
        assert meta['global_update'] == 512
        assert sha(cp / 'evaluation.json') == json.loads((cp / 'evaluation_integrity.json').read_text())['sha256']
        dest = ROOT / f'seed_{seed}.json'
        receipt = ROOT / f'seed_{seed}.integrity.json'
        if dest.exists():
            assert receipt.exists(), 'Unreceipted result: preserve for manual review'
            assert json.loads(receipt.read_text())['sha256'] == sha(dest)
            result = json.loads(dest.read_text())
            assert result['checkpoint_sha256'] == meta['state_sha256'] and result['execution_sha'] == execution_sha
        else:
            _, model, r, _, teacher, _ = original.template(seed, None, train_states)
            r, teacher = load_dual_checkpoint(cp, r, teacher)
            before = tree_hash((r, teacher))
            train = original.Evaluator(model)(r, seed, train_maps, train_states, False)
            after = tree_hash((r, teacher))
            assert before == after
            result = dict(seed=seed, execution_sha=execution_sha, checkpoint_sha256=meta['state_sha256'],
                frozen_state_sha256_before=before, frozen_state_sha256_after=after,
                optimizer_updates=0, test_episodes=0, train_results=train)
            write(dest, result)
            write(receipt, dict(sha256=sha(dest)))
            jax.clear_caches()
        train = aggregate(result['train_results'], train_maps)
        development = aggregate(json.loads((cp / 'evaluation.json').read_text())['results'], dev_maps)
        new_episodes += sum(v['episodes'] for v in train.values())
        row = dict(seed=seed, train=train, development=development,
            train_minus_development={s:train[s]['success_rate']-development[s]['success_rate'] for s in train})
        rows.append(row)
        print(json.dumps(dict(seed=seed,train={s:v['success_rate'] for s,v in train.items()},
            development={s:v['success_rate'] for s,v in development.items()})), flush=True)
    assert new_episodes == SPEC['budget']['new_total_episodes']
    assert source_hashes == {str(p):sha(p) for p in SOURCE.rglob('*') if p.is_file()}
    assert not (SOURCE / 'main').exists() and not list(SOURCE.rglob('final_test.json'))
    write(ROOT / 'aggregate_result.json', dict(seed_rows=rows, new_evaluation_episodes=new_episodes,
        development_episodes_reused=486, optimizer_updates=0, checkpoint_writes=0, test_episodes=0,
        source_unchanged=True, training_sha=SPEC['training_sha'], execution_sha=execution_sha,
        interpretation='Descriptive closure only; no causal failure attribution, no new training authorized.'))
    write(ROOT / 'verified_complete.json', dict(execution_sha=execution_sha,
        aggregate_sha256=sha(ROOT / 'aggregate_result.json'), completed_utc=time.time(),
        seed_result_sha256={s:sha(ROOT / f'seed_{s}.json') for s in SPEC['seeds']}))
    print('FROZEN TRAIN-MAP CLOSEOUT COMPLETE; MAIN20 REMAIN CLOSED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / 'failed.json', dict(error=traceback.format_exc(), utc=time.time()))
        raise
