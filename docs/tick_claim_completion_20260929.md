# TICK-CLAIM completion log

Understood as: finish the scientific-review gates in order—land-ready kernel artifacts, a goal-conditioned learner with distinct goal and world termination semantics, complete deterministic resume and frozen-evaluation checks with an actual parameter count, then run the fixed/mutant × seed 0–2 calibration matrix at 1,048,576 transitions per cell.

Quality gates:

1. The reviewed kernel regressions, generated artifacts, and attestations agree.
2. Goal completion cuts reward bootstrap/GAE without resetting the world; only world termination resets it.
3. A checkpoint round trip preserves the complete learner state and next update, frozen evaluation leaves it unchanged, and the GC policy parameter count is recorded.
4. Only after gates 1–3 pass, all six calibration cells finish with configs, checkpoints, metrics, and an aggregate report.

## Cycle 1 memo: learner vertical slice

- Preserve: the GC runner owns model/optimizer state, per-worker world and oracle state, environment and sampler RNG, current command, seen goals, command age, world-level violation provenance, counters, and schedule position as one serialized PyTree.
- Preserve: `goal_done` is a pseudo-termination for GAE while `world_done` alone selects a reset state. In fixed `deliver_3` calibration, successful workers retain the world and mask learner transitions until the next world reset because the monotone goal cannot become false again.
- Anti-pattern eliminated: a parameter-only or update-zero checkpoint can appear reproducible without proving optimizer continuation. The gate saves after one optimizer update and compares the complete next update, metrics, policy, RNG, sampler, environment, and schedule state.
- Anti-pattern eliminated: “delivery after any violation” overstates exploit contribution. The runner tracks excess-created grain as physical provenance and decrements it only when that amount is delivered.
- Next gate: run the full 512×64 shape through JIT, record measured throughput and 2,640,181 parameters, validate frozen evaluation byte-for-byte, then freeze the six-cell run manifest before launching it.
