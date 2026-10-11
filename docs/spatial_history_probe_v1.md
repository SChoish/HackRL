# Frozen action-history probe

The [registered panel and budget](manifests/spatial_history_probe_v1.json) are the execution contract. No new training, source-checkpoint writes, sample rollout, or input changes are permitted. All40 final4096 policies are included, including Transformer seed132. Input contracts below describe training SHA441a9ed, not a new implementation.

| Component | Frozen contract |
| --- | --- |
| Current MLP actor | Current local map,13 numeric features including remaining time, and current goal. No action/history/validity input. |
| History MLP actor | Eight tokens: seven previous pre-action observations with selected actions, then current observation with zero unknown action. Flattened token encodings and two704-wide layers. |
| GRU actor | Same eight tokens,576-wide GRU, zero initial hidden on every forward; no hidden state preceding the window. |
| Transformer actor | Same eight tokens, six blocks,128 width,four heads,FF512; learned positions, causal valid-token attention, zero dropout. |
| Padding and waits | Left-zero padding with explicit invalid mask. Forced waits advance physical time but never enter history. First controlled observation has empty history; world reset clears history. |
| Time and goals | All actors retain time in every available observation. Same current commanded goal is given to every token. No hidden kernel or collision flag is provided. |
| Critic | Separate current-only CNN32/Dense512/four512/value head. No actor parameter sharing. |
| Teacher | Current-only CNN16/four512 DualLeoQ with input BatchRenorm and LayerNorm,12 goal heads. Same structure and update rule across actors, not the same learned weights/data. |
| Optimization | Existing PPO/teacher schedules in the [training manifest](manifests/spatial_architecture_v1.json). Probe reads actual final counters and parameter counts for every checkpoint; no optimizer is created or applied. |

Actor counts from the completed validation are1,476,663 /1,505,527 /1,503,479 /1,465,271 in table order. Critic2,103,585 and teacher1,460,494 are common. The evaluator checks actual checkpoint counts. This is a parameter-matched actor-family comparison, not matched compute or identical hidden widths.

A replays the original natural episodes and verifies complete path/summary parity against saved evaluations. A block means an attempted dash through an intermediate wall that would succeed in mutant but is rejected in fixed. Conditional recovery requires this event followed by delivery; no-block successes remain separate. Same-wall retries mean the same position and dash direction after the first block, not arbitrary direction changes.

B uses the legal transformed RIGHT approach, then blocked DASH_RIGHT versus WAIT in fixed. All physical/provenance state leaves and current inputs must be identical. Histories differ through the actual chosen action; no state corrections are allowed. Both continuations stop at the original episode tick64. The full panel is serialized before trained-policy queries. A current-only MLP must produce identical distributions and complete paths in both branches.

The panel is exactly16 layouts times delays0,8,10,12, shared by every policy. There are no replacements for rejected pairs. Equality covers the entire environment dataclass, not the deliberately different history buffers. Initial forced waits are outside history; the experimental WAIT action is inside history just like the experimental dash. Both branches therefore have two completed tokens: the common approach and the branch action. If no pair matches, only A is interpretable. In this kernel provenance means wall-passage/item-delivery fields, not a hidden previous-action flag.

A's reference is each final checkpoint's integrity-verified evaluation.json, which contains all requested paths, including failed episodes. Missing reference evidence and mismatched replay both halt execution, with distinct logged assertions/errors; the evaluator does not silently substitute a fresh reference. A failed current-MLP control also halts the remaining run, not just its interpretation.

Absolute return discounts from episode start; remaining return discounts from the first continuation action. Control length excludes forced waits but includes B's prefix and branch action. Lengths of failed episodes are not successful route lengths. The matched intervention separates actual history inputs, not action identity from the semantic meaning of failure. Its forced actions can be unlikely under the policy, so their prefix probabilities are reported.

Steps are one-based completed physical ticks. B's common prefix ends at delay+1; the forced branch ends at delay+2; the first continuation action ends at delay+3. Movement latency counts ticks after the block/branch until the first nonzero displacement, or null if none. Retry counts include all later same-position/same-direction dashes, including after revisiting the position. B's count excludes the forced branch itself; on its WAIT side it means subsequent attempts at the designated panel wall, not repeats of WAIT. The655,360-tick ceiling is conservative; B's actual available time is64 minus delay+2.

Results are aggregated inside each learner first; all five learner seeds and individual delay results remain available. The16 layouts are transforms of one fixture, and delays10/12 were already examined. Neither is a new holdout. A natural recovery and a B history effect are separate observations; their conjunction does not establish bug inference or a learned belief state.

Artifacts, logs and receipts go to the RAID path in the manifest. The official server queue runs the bounded evaluator with a lock, integrity-checked per-policy resume, a failure gate and the existing identity-checked GPU keepalive handoff. A verified completion receipt closes the queue. No training or architecture expansion follows automatically.
