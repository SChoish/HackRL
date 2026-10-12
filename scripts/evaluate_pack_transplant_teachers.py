#!/usr/bin/env python3
"""Frozen PACK S donor evaluation only. This entry point cannot launch BC."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import jax
import numpy as np

from evaluate_dual_teacher_greedy import TeacherGreedyPolicy, _sha256, _write_json
from hackrl.dual_leo import init_dual_leo_teacher, load_dual_checkpoint
from hackrl.pack_restore_gc import evaluate_pack_restore_gc_frozen, _FROZEN_EVAL_CACHE
from run_dual_leo_compare import ENVS

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "docs/manifests/pack_teacher_transplant_v1.json"
SPEC = json.loads(SPEC_PATH.read_text())
ROOT = Path(SPEC["teacher_eval_root"])


def tree_hash(tree):
    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        value = np.asarray(leaf)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def jobs():
    return [(seed, origin, Path(SPEC["source_root"]) /
             SPEC["checkpoint_relative"].format(seed=seed, origin=origin))
            for seed in SPEC["seeds"] for origin in SPEC["teacher_origins"]]


def capacity():
    data = dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),
                raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
                source_bytes=sum(p.stat().st_size for _, _, cp in jobs()
                                 for p in cp.iterdir() if p.is_file()),
                existing_run_bytes=sum(p.stat().st_size for p in ROOT.parent.rglob('*')
                                       if p.is_file()),
                projected_additional_bytes=256*1024**2, reserve_bytes=8*1024**3,
                chosen_filesystem='/raid/ext_csv', checkpoint_writes=0,
                measured_unix=time.time())
    assert data['raid']['free'] > data['projected_additional_bytes'] + data['reserve_bytes']
    return data


def summarize(rows):
    output = []
    for origin in SPEC['teacher_origins']:
        selected = [r for r in rows if r['origin'] == origin]
        for kernel in SPEC['evaluation']['kernels']:
            metrics = [r['kernels'][kernel]['natural_reset'] for r in selected]
            output.append(dict(origin=origin, kernel=kernel, learner_seeds=len(selected),
                **{k:float(np.mean([m[k] for m in metrics])) for k in
                   ('success_rate','violation_rate','violation_delivery_rate',
                    'mean_discounted_return','mean_length')}))
    return output


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'verified_complete.json').exists():
        receipt = json.loads((ROOT/'verified_complete.json').read_text())
        assert receipt['result_sha256'] == _sha256(ROOT/'result.json')
        return
    assert jax.default_backend() == 'gpu'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    execution_sha = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    hashes = {str(cp / name): _sha256(cp / name) for _,_,cp in jobs()
              for name in ('state.msgpack','config.json','metadata.json')}
    manifest = dict(execution_sha=execution_sha, spec_sha256=_sha256(SPEC_PATH),
                    spec=SPEC, source_hashes=hashes)
    manifest_path = ROOT/'run_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        _write_json(manifest_path, manifest)
    _write_json(ROOT/'capacity_start.json', capacity())
    spec = ENVS['pack']
    rows = []
    for seed, origin, cp in jobs():
        out = ROOT/f'seed{seed}_{origin}.json'
        receipt = out.with_suffix('.integrity.json')
        if out.exists():
            assert receipt.exists(), 'Unreceipted output preserved for manual review'
            assert json.loads(receipt.read_text())['sha256'] == _sha256(out)
            row = json.loads(out.read_text())
            assert row['execution_sha'] == execution_sha
            assert row['checkpoint_sha256'] == hashes[str(cp/'state.msgpack')]
        else:
            config = spec['from_payload'](json.loads((cp/'config.json').read_text()))
            meta = json.loads((cp/'metadata.json').read_text())
            assert meta['global_update'] == SPEC['teacher_global_update']
            assert config.seed == seed and config.variant == origin
            assert config.policy_hidden_size == config.teacher_hidden_size == 512
            _, runner = spec['initialize'](config)
            inputs = spec['inputs'](runner.env_state, runner.current_goal)
            teacher_net, teacher, _ = init_dual_leo_teacher(
                config, inputs[0], inputs[1], spec['goals'], spec['actions'])
            runner, teacher = load_dual_checkpoint(cp, runner, teacher)
            before = tree_hash((runner, teacher))
            actor = TeacherGreedyPolicy(teacher_net)
            params = dict(params=teacher.params, batch_stats=teacher.batch_stats)
            kernels = {}
            for kernel in SPEC['evaluation']['kernels']:
                kernels[kernel] = evaluate_pack_restore_gc_frozen(actor, params,
                    variant=kernel, split=SPEC['evaluation']['split'], stochastic=False,
                    repeats_per_state=1, seed_base=SPEC['evaluation']['seed_base'],
                    learner_seed=seed, source_growth_period=config.source_growth_period,
                    natural_only=True, record_episodes=True)
                assert kernels[kernel]['overall']['episodes'] == 32
                assert len(kernels[kernel]['episode_records']) == 32
                assert all(e['family'] == 'natural_reset' for e in kernels[kernel]['episode_records'])
            after = tree_hash((runner, teacher))
            assert before == after
            row = dict(seed=seed, origin=origin, execution_sha=execution_sha,
                checkpoint=str(cp), checkpoint_sha256=hashes[str(cp/'state.msgpack')],
                frozen_state_sha256_before=before, frozen_state_sha256_after=after,
                kernels=kernels, optimizer_updates=0,
                same_teacher_mutant_minus_fixed={k:kernels['mutant']['natural_reset'][k]-
                    kernels['fixed']['natural_reset'][k] for k in
                    ('success_rate','mean_discounted_return','mean_length','violation_delivery_rate')})
            _write_json(out,row)
            _write_json(receipt,dict(sha256=_sha256(out)))
            _FROZEN_EVAL_CACHE.clear()
            jax.clear_caches()
        rows.append(row)
        _write_json(ROOT/'capacity_latest.json', capacity())
        print(json.dumps(dict(completed_teachers=len(rows),seed=seed,origin=origin,
            kernels={k:v['natural_reset'] for k,v in row['kernels'].items()})),flush=True)
    episodes = sum(r['kernels'][k]['overall']['episodes'] for r in rows
                   for k in SPEC['evaluation']['kernels'])
    assert episodes == SPEC['evaluation']['total_episodes']
    assert hashes == {path:_sha256(path) for path in hashes}
    _write_json(ROOT/'result.json',dict(rows=rows,aggregate=summarize(rows),
        evaluation_episodes=episodes,optimizer_updates=0,checkpoint_writes=0,
        source_hashes_unchanged=True,bc_started=False,execution_sha=execution_sha))
    _write_json(ROOT/'verified_complete.json',dict(result_sha256=_sha256(ROOT/'result.json'),
        execution_sha=execution_sha,completed_unix=time.time(),evaluation_episodes=episodes))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True)
        _write_json(ROOT/'failed.json',dict(traceback=traceback.format_exc(),unix=time.time()))
        raise
