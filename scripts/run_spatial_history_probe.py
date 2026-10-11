#!/usr/bin/env python3
"""Bounded, restartable evaluation of all40 immutable final policies."""
import hashlib
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
from hackrl import spatial_history_probe as probe
from hackrl import spatial_time_history_eval as evaluation

REPO = Path(__file__).resolve().parents[1]
SPEC = json.loads((REPO / 'docs/manifests/spatial_history_probe_v1.json').read_text())
ROOT = Path(SPEC['run_root'])
SOURCE = Path(SPEC['source_root'])


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tree_hash(tree):
    return hashlib.sha256(serialization.to_bytes(jax.device_get(tree))).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.pending')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def journal(record):
    with (ROOT / 'attempts.jsonl').open('a') as f:
        f.write(json.dumps(dict(utc=time.time(), **record)) + '\n')
        f.flush(); os.fsync(f.fileno())


def capacity():
    projected = SPEC['storage']['artifact_allowance_bytes']
    reserve = max(SPEC['storage']['reserve_min_bytes'], int(.2 * projected))
    record = dict(home=shutil.disk_usage('/home/ext_csv')._asdict(),
                  raid=shutil.disk_usage('/raid/ext_csv')._asdict(),
                  existing_source_bytes=sum(p.stat().st_size for p in SOURCE.rglob('*') if p.is_file()),
                  existing_probe_bytes=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
                  projected_peak_additional_bytes=projected, reserve_bytes=reserve,
                  chosen_filesystem='/raid/ext_csv', checkpoint_writes=0)
    journal(dict(event='capacity', **record))
    assert record['raid']['free'] > projected + reserve
    return record


def source_path(arch, seed, percent):
    return SOURCE / arch / f'seed_{seed}' / f'fixed_{percent:02d}pct' / 'checkpoints' / 'update_4096'


def provenance():
    assert not subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip()
    execution_sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    receipt = json.loads((SOURCE / 'verified_complete.json').read_text())
    assert receipt['execution_sha'] == SPEC['source_sha']
    assert receipt['jobs'] == 60 and receipt['result_sha256'] == sha(SOURCE / 'result.json')
    # Every old scientific Python file is byte-for-byte unchanged, not just the actor.
    for name in subprocess.check_output(['git','ls-tree','-r','--name-only', SPEC['source_sha'],
                                          'src/hackrl'], cwd=REPO, text=True).splitlines():
        expected = subprocess.check_output(['git','show',f"{SPEC['source_sha']}:{name}"], cwd=REPO)
        assert hashlib.sha256(expected).hexdigest() == sha(REPO / name), name
    sources = {}
    for arch in SPEC['architectures']:
        for seed in SPEC['seeds']:
            parents = []
            for q in SPEC['fixed_percent']:
                p = source_path(arch, seed, q)
                hashes = {name: sha(p / name) for name in ('state.msgpack','metadata.json','config.json','evaluation.json')}
                meta = json.loads((p / 'metadata.json').read_text())
                cfg = json.loads((p / 'config.json').read_text())
                assert meta['state_sha256'] == hashes['state.msgpack']
                assert meta['execution_sha'] == SPEC['source_sha'] and meta['global_update'] == 4608
                assert cfg['architecture'] == arch and cfg['fixed_episode_probability'] == q / 100
                assert cfg['config']['seed'] == seed
                assert json.loads((p / 'evaluation_integrity.json').read_text())['sha256'] == hashes['evaluation.json']
                parent = SOURCE / arch / f'seed_{seed}' / 'pretraining/checkpoints/update_512/state.msgpack'
                assert meta['parent_pretrain_sha256'] == sha(parent)
                parents.append(meta['parent_pretrain_sha256'])
                sources[f'{arch}_{seed}_{q}'] = dict(path=str(p), hashes=hashes,
                    parent_sha256=parents[-1], counters=meta['counters'])
            assert parents[0] == parents[1]
    return dict(execution_sha=execution_sha, spec=SPEC, source_receipt=receipt, sources=sources,
                source_contract='All existing src/hackrl files equal training SHA; new probe is separate.')


def count(tree):
    return sum(np.asarray(x).size for x in jax.tree.leaves(tree))


def run_policy(arch, seed, percent, evidence, panel, rows, functions):
    key = f'{arch}_{seed}_{percent}'
    dest = ROOT / key
    dest.mkdir(exist_ok=True)
    source = evidence['sources'][key]
    receipt_path = dest / 'verified_complete.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['execution_sha'] == evidence['execution_sha']
        assert receipt['source_hashes'] == source['hashes']
        assert receipt['result_sha256'] == sha(dest / 'result.json')
        return json.loads((dest / 'result.json').read_text())
    capacity()
    journal(dict(event='policy_attempt', policy=key))
    cp = Path(source['path'])
    raw = serialization.msgpack_restore((cp / 'state.msgpack').read_bytes())
    before = tree_hash(raw)
    assert int(raw['gc']['global_update']) == 4608
    assert int(raw['gc']['train_state']['step']) == source['counters']['ppo_optimizer_steps']
    assert int(raw['leo']['step']) == source['counters']['teacher_optimizer_steps']
    actual_bn = {str(path): int(value) for path, value in
                 jax.tree_util.tree_leaves_with_path(raw['leo']['batch_stats'])
                 if str(path[-1]).find('steps') >= 0}
    assert actual_bn == source['counters']['batch_renorm_counters']
    params = jax.tree.map(jnp.asarray, raw['gc']['train_state']['params'])
    params_before = tree_hash(params)
    expected = json.loads((cp / 'evaluation.json').read_text())['results']['mode']
    natural, query, continuation = functions
    A = {}
    for delay in SPEC['delays']:
        for kernel in ('fixed','mutant'):
            name = f'{kernel}_delay{delay}'
            journal(dict(event='A_attempt', policy=key, condition=name, max_episodes=16))
            trace = jax.device_get(natural(params, np.int32(seed), kernel == 'fixed', np.int32(delay), False))
            recs = evaluation.records_from_trace(trace, 1, delay)
            assert recs == expected[name]['episodes'], f'Natural path parity failed: {key}/{name}'
            assert evaluation.summarize(recs) == expected[name]['summary']
            A[name] = dict(summary=evaluation.summarize(recs),
                          episodes=probe.enrich_natural(recs, trace))
            journal(dict(event='A_complete', policy=key, condition=name,
                         episodes=len(recs), active_ticks=sum(r['length'] for r in recs)))
    prefix_probs = np.asarray(query(params, panel['prefix'], panel['prefix_history']))
    blocked_probs = np.asarray(query(params, panel['blocked'], panel['blocked_history']))
    wait_probs = np.asarray(query(params, panel['waited'], panel['waited_history']))
    for value in (prefix_probs, blocked_probs, wait_probs):
        assert np.isfinite(value).all() and np.allclose(value.sum(-1), 1)
    valid = jnp.asarray([r['included'] for r in rows])
    states = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), panel['blocked'], panel['waited'])
    histories = jax.tree.map(lambda x, y: jnp.concatenate((x, y)), panel['blocked_history'], panel['waited_history'])
    journal(dict(event='B_attempt', policy=key, max_continuations=2 * int(valid.sum())))
    traces = jax.device_get(continuation(params, states, histories, jnp.concatenate((valid, valid))))
    journal(dict(event='B_complete', policy=key, continuations=2 * int(valid.sum()),
                 active_ticks=int(traces['active'].sum())))
    n = len(rows)
    blocked_records = probe.continuation_records({k: v[:, :n] for k, v in traces.items()}, rows, 'blocked')
    wait_records = probe.continuation_records({k: v[:, n:] for k, v in traces.items()}, rows, 'wait')
    B = []
    for rb, rw in zip(blocked_records, wait_records):
        i = rb['panel_index']
        assert i == rw['panel_index']
        d = rows[i]['branch_actions'][0]
        if arch == 'current_mlp':
            assert np.array_equal(blocked_probs[i], wait_probs[i]), 'Current-only negative control failed'
            assert rb['path'] == rw['path'], 'Current-only continuation negative control failed'
        B.append(dict(panel_index=i, layout=rows[i]['layout'], delay=rows[i]['delay'],
            tv=float(.5 * np.abs(blocked_probs[i] - wait_probs[i]).sum()),
            mode_changed=bool(blocked_probs[i].argmax() != wait_probs[i].argmax()),
            prefix_dash_probability=float(prefix_probs[i, d]), prefix_wait_probability=float(prefix_probs[i, 0]),
            blocked_dash_probability=float(blocked_probs[i, d]), wait_dash_probability=float(wait_probs[i, d]),
            blocked_probabilities=blocked_probs[i].tolist(), wait_probabilities=wait_probs[i].tolist(),
            path_changed=rb['path'] != rw['path'], success_changed=rb['success'] != rw['success'],
            blocked=rb, wait=rw))
    p = raw['gc']['train_state']['params']['params']
    counts = dict(actor=count(p['actor']), critic=count({k: v for k, v in p.items() if k != 'actor'}),
                  teacher=count(raw['leo']['params']))
    training_spec = json.loads((REPO / 'docs/manifests/spatial_architecture_v1.json').read_text())
    assert counts['actor'] == training_spec['parameter_matching']['actor_expected_from_shapes'][arch]
    assert counts['critic'] == training_spec['validation_evidence']['critic_parameters']
    assert counts['teacher'] == training_spec['validation_evidence']['teacher_parameters']
    assert tree_hash(raw) == before and tree_hash(params) == params_before
    assert {name: sha(cp / name) for name in source['hashes']} == source['hashes']
    result = dict(architecture=arch, seed=seed, fixed_percent=percent, source=source,
                  parameter_counts=counts, optimizer_counters=source['counters'],
                  frozen_raw_tree_sha256=before, new_optimizer_updates=0, normalization_updates=0,
                  A_parity=True, A=A, B=B, excluded=[r for r in rows if not r['included']],
                  accounting=dict(A_episodes=sum(len(v['episodes']) for v in A.values()),
                    B_continuations=2 * len(B),
                    A_active_ticks=sum(r['length'] for v in A.values() for r in v['episodes']),
                    A_forced_wait_ticks=2 * SPEC['layouts'] * sum(SPEC['delays']),
                    B_continuation_ticks=sum(x[b]['continuation_length'] for x in B for b in ('blocked','wait'))))
    write(dest / 'result.json', result)
    write(receipt_path, dict(execution_sha=evidence['execution_sha'], source_hashes=source['hashes'],
                            result_sha256=sha(dest / 'result.json')))
    journal(dict(event='policy_complete', policy=key, accounting=result['accounting']))
    print(json.dumps(dict(completed=key, accounting=result['accounting'])), flush=True)
    return result


def mean(rows, key):
    return float(np.mean([r[key] for r in rows])) if rows else None


def aggregate(results):
    """One row per learner and delay (plus within-learner all-delay summary)."""
    out = []
    for result in results:
        for delay in [*SPEC['delays'], 'all']:
            select = SPEC['delays'] if delay == 'all' else [delay]
            row = {k: result[k] for k in ('architecture','seed','fixed_percent')}
            row['delay'] = delay
            row['A'] = {}
            for kernel in ('fixed','mutant'):
                records = [r for d in select for r in result['A'][f'{kernel}_delay{d}']['episodes']]
                row['A'][kernel] = evaluation.summarize(records)
                row['A'][kernel]['path_classes'] = {name: sum(r['path_class'] == name for r in records)
                    for name in ('blocked_then_success','blocked_then_failure','no_block_success','no_block_failure')}
            pairs = [p for p in result['B'] if p['delay'] in select]
            row['B'] = dict(pairs=len(pairs), **{key: mean(pairs, key) for key in
                ('tv','mode_changed','path_changed','success_changed','prefix_dash_probability',
                 'prefix_wait_probability','blocked_dash_probability','wait_dash_probability')})
            for branch in ('blocked','wait'):
                branch_rows = [p[branch] for p in pairs]
                row['B'][branch] = {key: mean(branch_rows, key) for key in
                    ('success','same_wall_retries','continuation_length','discounted_return','remaining_discounted_return')}
            out.append(row)
    return out


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', 'Production probe requires the registered GPU queue'
    evidence = provenance()
    capacity()
    manifest = ROOT / 'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == evidence
    else:
        write(manifest, evidence)
    panel = jax.jit(lambda: probe.make_panel(tuple(SPEC['delays']), SPEC['layouts']))()
    rows = probe.validate_panel(panel)
    panel_hash = tree_hash(panel)
    panel_record = dict(rows=rows, full_panel_sha256=panel_hash,
        generation_forced_wait_ticks=SPEC['layouts'] * sum(SPEC['delays']),
        generation_controlled_prefix_ticks=len(rows), generation_branch_ticks=2 * len(rows),
        exact_state_rule='All dataclass leaves AND all current input arrays match exactly')
    if (ROOT / 'panel.json').exists():
        assert json.loads((ROOT / 'panel.json').read_text()) == panel_record
    else:
        write(ROOT / 'panel.json', panel_record)
    # Save reproducible exact arrays before querying any trained policy.
    panel_bytes = serialization.to_bytes(jax.device_get(panel))
    panel_path = ROOT / 'panel.msgpack'
    if panel_path.exists():
        assert panel_path.read_bytes() == panel_bytes
    else:
        with panel_path.open('xb') as f:
            f.write(panel_bytes); f.flush(); os.fsync(f.fileno())
    journal(dict(event='panel_frozen', sha256=panel_hash, included=sum(r['included'] for r in rows)))
    results = []
    for arch in SPEC['architectures']:
        model = a.ArchitectureActorCritic(arch)
        functions = (evaluation.make_rollout(model, False, 1, history_api=a),
                     probe.make_queries(model), probe.make_continuation(model))
        for seed in SPEC['seeds']:
            for percent in SPEC['fixed_percent']:
                results.append(run_policy(arch, seed, percent, evidence, panel, rows, functions))
        jax.clear_caches()
    assert len(results) == 40
    accounting = {key: sum(r['accounting'][key] for r in results) for key in results[0]['accounting']}
    assert accounting['A_episodes'] == 5120
    assert accounting['B_continuations'] <= 5120
    assert accounting['A_active_ticks'] + accounting['B_continuation_ticks'] <= 655360
    assert tree_hash(panel) == panel_hash
    write(ROOT / 'result.json', dict(execution_sha=evidence['execution_sha'], status='complete',
        source_sha=SPEC['source_sha'], policies=40, new_optimizer_updates=0, normalization_updates=0,
        accounting=accounting, panel=panel_record, seed_rows=aggregate(results),
        policy_results={f"{r['architecture']}_{r['seed']}_{r['fixed_percent']}":
            sha(ROOT / f"{r['architecture']}_{r['seed']}_{r['fixed_percent']}" / 'result.json') for r in results}))
    write(ROOT / 'verified_complete.json', dict(execution_sha=evidence['execution_sha'], policies=40,
          result_sha256=sha(ROOT / 'result.json'), completed_utc=time.time()))
    print('VERIFIED COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        ROOT.mkdir(parents=True, exist_ok=True)
        write(ROOT / 'failed.json', dict(utc=time.time(), error=traceback.format_exc()))
        raise
