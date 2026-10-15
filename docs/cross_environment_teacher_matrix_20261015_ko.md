# 공통 Teacher Matrix v1 구현 정본

## 상태

TICK-CLAIM, PACK-RESTORE, CRAFT-REMAIN을 동일한 teacher 조건 비교로 고정했다. 기존 결과와 대형 checkpoint는 원래 위치에서 해시로 참조하며, TICK `D-task`·`D-frozen` 20 jobs와 CRAFT `D-task`·`D-frozen`·`D-off` 30 jobs만 신규 학습 대상이다. PACK 신규 학습은 0이다.

근거 인덱스, 정확한 50-job builder, 재개 가능한 runner, 공통 full-trace evaluator와 시작 패널 구현은 완료했다. 본학습 50 jobs와 후속 공통 과학 평가는 명시적으로 승인됐다. 통합 smoke는 독립 구현 검증으로 보존하고 과학 성능 결과로 사용하지 않으며, 본학습은 별도 RAID root에서 각 job의 4,096 updates를 모두 실행한다.

| 환경 | Source | D-on | D-task | D-frozen | D-off | 신규 학습 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TICK-CLAIM, seed 20–24 | 재사용 | 재사용 | 10 | 10 | 재사용 | 20 jobs |
| PACK-RESTORE, seed 30–39 | 재사용 | 재사용 | 재사용 (`D-delivery`) | 재사용 | 재사용 | 0 jobs |
| CRAFT-REMAIN, seed 20–24 | 재사용 | 재사용 | 10 | 10 | 10 | 30 jobs |

## 근거 고정

[evidence manifest](manifests/cross_environment_teacher_matrix_v1_evidence.json)는 6개 결과군, 130개 source/final checkpoint, 123개 config·최종 curve·result artifact를 연결한다. checkpoint tree에는 새 파일을 쓰지 않고 `state.msgpack`, `config.json`, `metadata.json`, 선택적 `arm.json`의 합성 SHA-256만 기록했다.

작업 시작 시 수정·미추적 목록을 evidence manifest에 보존했다. 선택 커밋에는 매트릭스가 참조하는 CRAFT 명세·결과와 Dual 결과, 새 구현·검증·문서만 포함한다. 다른 사용자 파일은 reset, stash, 삭제, stage하지 않는다.

과거 코드의 동일성에는 다음 경계를 둔다.

- PACK 결과는 각 RAID run manifest의 `execution_code_sha`와 당시 scientific source hashes를 정본으로 삼는다.
- CRAFT의 runner와 CRAFT GC 코드는 `fd201788...`과 일치하지만, 현재 `dual_leo.py`는 당시 파일과 다르다.
- TICK D-on은 dirty `fbaef1d...` 작업 트리에서 시작했다는 과거 기록이 있으므로 `9155a0d...`는 가장 가까운 사후 freeze일 뿐 정확한 실행 checkout을 증명하지 않는다.
- 과거 curve는 원래 지표 정의를 유지한다. 새 exhaustive legal opportunity로 소급 변경하지 않는다.

## 비교 근거 등급

- 신규 동일 코드 비교: TICK의 D-task–D-frozen, CRAFT의 D-task–D-frozen–D-off와 각 fixed–mutant 분기.
- 동등성이 확인된 재사용 비교: 같은 과거 run 내부의 fixed–mutant, PACK `pack_restore_scale_bc_v1` 내부 D-on–D-off.
- 탐색적 비교: 신규 조건과 과거 D-on/D-off의 비교 및 서로 다른 PACK run을 가로지르는 비교. 이 항목은 단일 요인 개입으로 표시하지 않는다.

TICK·CRAFT의 독립 source manifest 부재는 checkpoint 합성 hash, config·curve hash와 보존된 실행 commit 상태를 대체 근거로 남긴다.

## 공통 runner

작업 목록의 유일한 기준은 [matrix manifest](manifests/cross_environment_teacher_matrix_v1.json)의 `new_learning_jobs`다. `src/hackrl/cross_environment_teacher_matrix.py`가 이를 정확히 50 jobs로 변환하고, `scripts/run_cross_environment_teacher_matrix.py`가 한 job씩 명시적으로 실행한다. queue·watcher mode는 없다.

- source는 환경과 learner seed가 같은 full-Dual `update_512` checkpoint다.
- 목표는 각 환경 명세의 `goal_id`를 해당 환경 `GOAL_IDS`에서 독립적으로 조회하고 기록 index와 대조한다.
- `D-task`는 한 task head만 teacher TD target에 포함하며 target term 수를 검사한다.
- `D-frozen`은 teacher parameters, Adam state, BatchRenorm statistics, optimizer step의 전후 digest가 모두 같은지 검사한다.
- `D-off`는 teacher update를 유지하면서 policy BC coefficient가 실제로 0인지 검사한다.
- 기존 D-on의 update-0 평가는 새 rollout 없이 원본 SHA-256과 provenance sidecar를 붙여 재사용한다.
- smoke의 `adapt_1`은 별도 smoke root에만 보존한다. 본학습은 새 main root에서 source checkpoint부터 4,096 updates를 전부 수행한다.

검증과 작업 목록 확인:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/validate_cross_environment_teacher_matrix.py
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py --validate
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py --list-jobs
```

본학습은 고정 목록 순서대로 다음 한-job 명령을 반복한다.

```bash
HACKRL_DEVICE=cuda PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py \
  --run-job JOB_ID --authorize-main-training
```

## 공통 trace evaluator

[panel manifest](manifests/cross_environment_teacher_trace_panel_v1.json)은 TICK·PACK의 validation 16 layouts × 2 phases와 CRAFT의 두 phases를 family별 순서·state hash와 함께 고정한다. fixed·mutant, mode·sample, 자연·기회 시작은 경로상 별도 파일로 저장한다.

각 gzip JSONL은 완전 상태, public observation, action과 logits, timestep, 종료 이유, 전후 물리 자원, 독립 oracle의 실제 결함 발동·활용, exhaustive legal trigger action 목록, 기존 좁은 `opportunity_reached`를 분리 기록한다. oracle 계산은 actor action 선택 뒤에만 실행하며 actor·teacher 입력이나 학습 보상에 들어가지 않는다. 저장 trace는 transition kernel과 policy·teacher를 호출하지 않는 별도 post-hoc replay로 자원 보존량·결함 발동·최종 활용을 다시 검사한다. 재사용 checkpoint와 신규 checkpoint도 `reused_result`·`new_matrix_result`로 분리한다.

전체 공통 평가의 사전 상한은 88,200 episodes, 11,289,600 steps다. 50 jobs 완료 후 `scripts/run_cross_environment_teacher_evaluation.py --run-all --authorize-common-evaluation`로 실행한다. 완전한 task receipt와 artifact hash가 있는 셀만 재사용하며, 기록 유실·불완전 셀은 자동 재평가하지 않는다.

## 예산 경계

본학습 50개의 collection은 6,710,886,400 slots다. 완료된 독립 smoke는 163,840 slots이며, 계획 총량은 6,711,050,240 slots다. Smoke는 본학습 update를 줄이지 않는다. 실패·재개 시도는 RAID run root의 `collection_ledger.jsonl`에 별도 행과 보수적 charge로 기록한다.

Smoke 저장 위치는 `/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1`, 본학습 저장 위치는 `/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1_main`이다. checkpoint writer 직전에 `/home/ext_csv`와 `/raid/ext_csv`의 여유 공간, 기존 run 크기, final retention, predecessor-successor overlap, 평가 artifact, 로그, 안전 reserve를 다시 계산한다.

## 정본

- 매트릭스: `docs/manifests/cross_environment_teacher_matrix_v1.json`
- 근거: `docs/manifests/cross_environment_teacher_matrix_v1_evidence.json`
- 평가 패널: `docs/manifests/cross_environment_teacher_trace_panel_v1.json`
- validator: `scripts/validate_cross_environment_teacher_matrix.py`
- runner: `scripts/run_cross_environment_teacher_matrix.py`
- trace evaluator: `scripts/eval_cross_environment_teacher_traces.py`
- 평가 orchestrator: `scripts/run_cross_environment_teacher_evaluation.py`

## 통합 smoke 완료

구현 freeze `27bf886...`과 독립 replay 보강 `838af48...`로 독립 smoke 다섯 job을 실행해 5/5 통과했다. 163,840 slots는 본학습 6,710,886,400 slots와 분리해 기록한다. TICK·CRAFT의 D-task scaling, D-frozen 전체 teacher 상태 동결, CRAFT D-off의 실제 BC 0을 확인했다. 비과학적 trace smoke도 두 환경의 fixed·mutant에서 통과했고, 저장 trace-only 독립 replay는 4/4 통과했다. 50-job 본학습과 공통 과학 평가는 이번 승인 commit 직후 고정 순서로 실행한다.

상세 수치와 실행 명령은 [구현 완료 보고](cross_environment_teacher_matrix_v1_implementation_20261015_ko.md)에 있다.
