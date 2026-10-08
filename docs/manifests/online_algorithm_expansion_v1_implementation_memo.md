# Online algorithm implementation cycle memo

This cycle implemented reusable GC-PQN, LEO, Dual LEO(PQN), and GC-SD-SAC cores plus common TICK-CLAIM/PACK-RESTORE wiring. It did not run an environment smoke, training job, queue, or checkpoint write; the executions reported below are unit tests only.

## Portable lessons

- Online PQN and LEO must not silently inherit Double DQN replay or target networks. Their frozen rollout is consumed once and then discarded.
- A replay optimizer step is compute, not a new physical transition. SD-SAC increments its environment counter only when the fixture runner collects a transition.
- Raw Q values are not categorical-policy logits for a stochastic evaluation. The registered Q view is epsilon-zero greedy; any epsilon behavior view needs its own frozen evaluation contract.
- Goal completion terminates the commanded PQN target, while LEO retains a distinct terminal mask for every goal. World termination masks every head.
- The Dual method in this phase is two TD learners combined at action selection with a fixed 0.3 LEO weight. It has no PPO policy or behavior-cloning loss.
- SD-SAC needs behavior entropy in replay. Recomputing it later would erase the published old-policy entropy penalty.

## Quality gates carried forward

- Hand-compute categorical expectations, terminal masks, Q clipping, entropy penalty, and temperature-gradient direction before any environment run.
- Clone learner, optimizer, normalization, environment, RNG, replay contents, cursor, and sampler RNG at the fixed/mutant fork.
- Reject empty or invalid-only replay updates instead of reporting them as applied gradients.
- Keep the candidate registry empty until a reviewed amendment freezes every hyperparameter. Passing implementation tests is not development authority.
- Before any checkpoint-producing smoke or training run, apply the repository capacity policy and choose RAID when the peak forecast threatens the reserve.

## QA result

The focused implementation suite passed 18 tests. A broader related suite passed 45 tests and reproduced two exact-equality failures in the unchanged legacy Dual test path. Those failures are recorded but were not repaired in this cycle because they are outside the new implementation scope. Because no clean parent-SHA checkout was tested, this record does not claim that the failures predate the current worktree.
