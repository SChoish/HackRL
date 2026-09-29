# TICK-CLAIM completion log

Understood as: finish the scientific-review gates in order—land-ready kernel artifacts, a goal-conditioned learner with distinct goal and world termination semantics, complete deterministic resume and frozen-evaluation checks with an actual parameter count, then run the fixed/mutant × seed 0–2 calibration matrix at 1,048,576 transitions per cell.

Quality gates:

1. **Complete.** The reviewed kernel regressions, generated artifacts, and attestations agree. The accounting and materialized-start fixes landed in `61b3a72` and the reviewed kernel manifest in `ffce206`.
2. **Complete.** Goal completion cuts reward bootstrap/GAE without resetting the world; only world termination resets it. Boundary and simultaneous-terminal probes pass.
3. **Complete.** A checkpoint round trip preserves the complete learner state and next update, frozen evaluation leaves it byte-identical, and the measured GC policy parameter count is 2,640,181. The frozen protocol is `8a50968`; the synchronized implementation revision is `50c985c` and its gate attestation is `f16828c`.
4. **Complete.** All six fixed/mutant x seed 0-2 cells finished 1,048,576 transitions each. The aggregate validates all configs, final checkpoints, budgets, evaluations, implementation ancestry, and source hashes with no errors.

Final evidence:

- Frozen run manifest: `docs/manifests/tick_claim_gc_pilot_v1.json`
- Pre-run validation: `docs/manifests/tick_claim_gc_v1_validation.json`
- Resolved implementation and hashes: `docs/manifests/tick_claim_gc_v1_resolved.json`
- Six-cell aggregate: `docs/manifests/tick_claim_gc_pilot_v1_results.json`
- Fresh-process reproducibility check: `docs/manifests/tick_claim_gc_pilot_v1_reproducibility.json`
- Scientific interpretation: `docs/tick_claim_gc_calibration_results_ko.md`
- Raw configs, updates, complete checkpoints, evaluations, and working-tree attestations: `runs/tick_claim_gc_pilot_v1_synced/`
- Final repository suite: 126 passed, 1 third-party deprecation warning, in 350.12 seconds; the timing-only synchronization change then passed the 9 focused GC tests and a two-update pilot.

## Cycle 1 memo: learner vertical slice

- Preserve: the GC runner owns model/optimizer state, per-worker world and oracle state, environment and sampler RNG, current command, seen goals, command age, world-level violation provenance, counters, and schedule position as one serialized PyTree.
- Preserve: `goal_done` is a pseudo-termination for GAE while `world_done` alone selects a reset state. In fixed `deliver_3` calibration, successful workers retain the world and mask learner transitions until the next world reset because the monotone goal cannot become false again.
- Anti-pattern eliminated: a parameter-only or update-zero checkpoint can appear reproducible without proving optimizer continuation. The gate saves after one optimizer update and compares the complete next update, metrics, policy, RNG, sampler, environment, and schedule state.
- Anti-pattern eliminated: “delivery after any violation” overstates exploit contribution. The runner tracks excess-created grain as physical provenance and decrements it only when that amount is delivered.
- Gate result: the full 512×64 shape completed two fully synchronized JIT updates in validation; the measured steady CPU validation throughput was 6,168.87 transitions/second. Frozen evaluation was byte-identical and the complete checkpoint produced an identical next update after restore.

## Cycle 2 memo: frozen six-cell calibration

- Preserve: freeze the manifest and implementation hashes before results, then accept a cell only when its final checkpoint records update 32 and 1,048,576 environment steps and its evaluation leaves learner bytes unchanged.
- Preserve: use learner seed as the aggregate unit. Evaluation episodes are repeated observations within a run, not independent learner replicates.
- Observed: fixed training produced no oracle violations. Mutant training produced 70, 48, and 25 violation events for seeds 0, 1, and 2; violation-created grain delivered was 8, 2, and 1.
- Observed: stochastic frozen evaluation exposed a mutant violation in every seed and delivered violation-created grain for seeds 0 and 1. Greedy mode solved neither variant and exposed no violations in any seed.
- Fresh-process check: a complete fixed seed-0 repeat produced byte-identical update metrics and full checkpoint state. The unsynchronized preliminary queue was superseded in full because its policy output did not reproduce; its causal mismatch was not established.
- Reproducibility boundary: the canonical paired matrix uses the recorded CPU runtime. One complete fixed seed-0 cell was repeated from a fresh process with byte-identical update metrics and checkpoint state; cross-seed, cross-variant, cross-runtime, and GPU bitwise determinism are not claimed.
- Interpretation: the instrumentation distinguishes opportunity, creation, repetition, delivery contribution, and ordinary success. This calibration demonstrates that the wiring can observe the synthetic defect, not that three seeds establish reliable exploit adoption or general bug-discovery ability.
- Stop condition: the registered six-cell calibration is complete. Do not expand width, algorithm, or budget solely from these descriptive results.
