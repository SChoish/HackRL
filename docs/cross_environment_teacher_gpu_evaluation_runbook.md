# Common teacher matrix GPU evaluation runbook

This run evaluates the frozen 180-policy inventory as 1,440 full-trace tasks on the H200. It does not reuse any of the 206 receipts from the superseded CPU attempt.

## Preconditions

- Evaluation sources and the trigger-bearing CPU/CUDA validation receipt are committed and clean.
- The 180 checkpoints pass the completeness check.
- JAX reports only the CUDA GPU backend for scientific shards.
- The RAID capacity preflight preserves the larger of 8 GiB or 20% of projected remaining writes.

## Execution

Prepare the immutable manifest once:

```bash
PYTHONPATH=src:scripts python scripts/run_cross_environment_teacher_evaluation_gpu.py \
  --prepare --authorize-common-evaluation
```

Run the operational wrapper. It starts four policy-disjoint GPU shards concurrently and finalizes only after all four pass:

```bash
bash ops/cross_environment_teacher_gpu_watch/run_gpu.sh
```

Each shard owns 45 policies and 360 tasks, holds a non-blocking shard lock, and writes a separate ledger. The external server watcher uses the same wrapper and a read-only completion probe. The CPU fallback only waits; it never performs scientific evaluation.

Finalization re-hashes all artifacts, checks every trace header and independent replay, requires exactly 88,200 episodes, and enforces the 11,289,600-step upper bound.

## Interruption contract

Completed task receipts are verified and reused. A started task with no passing receipt is not automatically re-evaluated. Preserve or quarantine its partial artifacts and record explicit retry authorization before another attempt. The old CPU evaluation and all validation smokes remain separate from the scientific GPU root.
