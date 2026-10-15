# 공통 Teacher Matrix v1 구현 완료 보고

## 완료 상태

구현 freeze commit은 `27bf886484ae74a3544682ac10ca591a608fc99e`다. 이 commit으로 근거 인덱스, 50-job builder, 단일-job runner, full-trace evaluator, 평가 패널과 contract tests를 고정했고 학습 contract smoke를 실행했다. 저장 trace의 독립 post-hoc replay 보강은 `838af485530f8c04d9f53336f8ade156a9dfd783`에 고정한 뒤 trace smoke를 다시 검증했다. **50-job 본학습과 후속 공통 과학 평가는 명시적으로 승인됐으며, 새 실행 commit 고정 직후 시작한다.**

Smoke receipt는 `docs/manifests/cross_environment_teacher_matrix_v1_smoke.json`, 원본은 `/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1`에 있다.

## 실행 명령

모든 명령의 기준 디렉터리는 `/home/ext_csv/HackRL`이다. 첫 본학습 job이 시작될 때의 `HEAD`, scientific source SHA-256, matrix digest, 고정 job ID 목록과 열 개 source checkpoint fingerprint는 main root의 `run_manifest.json`에 기록된다. `--list-job-ids`가 내보내는 순서를 그대로 사용하며, 같은 job 명령을 다시 실행하면 마지막 저장 `adapt_N`에서 재개한다. `adapt_4096` 확인과 `job_receipts/JOB_ID.json`의 `passed` 기록이 job 완료 조건이다.

검증과 작업 확인:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/validate_cross_environment_teacher_matrix.py
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py --validate
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py --list-job-ids
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python scripts/run_cross_environment_teacher_matrix.py --list-jobs
```

이번에 실행한 smoke:

```bash
HACKRL_DEVICE=cpu JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/run_cross_environment_teacher_matrix.py --smoke
```

본학습은 matrix가 출력하는 고정 job ID 순서대로 한 job씩 명시한다.

```bash
HACKRL_DEVICE=cuda PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/run_cross_environment_teacher_matrix.py \
  --run-job JOB_ID --authorize-main-training
```

공통 trace 한 셀의 명시적 실행 형식:

```bash
HACKRL_DEVICE=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/eval_cross_environment_teacher_traces.py \
  --environment ENV --checkpoint CHECKPOINT --checkpoint-origin reused_result \
  --learner-seed SEED --variant fixed --view mode --family natural_reset
```

저장 trace의 독립 replay만 다시 실행하는 형식:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/eval_cross_environment_teacher_traces.py --verify-trace TRACE.jsonl.gz
```

50 jobs 완료 후 공통 평가:

```bash
HACKRL_DEVICE=cpu PYTHONPATH=src:scripts /home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/run_cross_environment_teacher_evaluation.py \
  --run-all --authorize-common-evaluation
```

## 재사용 및 신규 작업

| 환경 | 재사용 | 신규 학습 |
| --- | --- | ---: |
| TICK | source, D-on, D-off | D-task 10, D-frozen 10 |
| PACK | source, D-on, D-task(`D-delivery`), D-frozen, D-off | 0 |
| CRAFT | source, D-on | D-task 10, D-frozen 10, D-off 10 |

총 신규 작업은 50 jobs다. 각 job은 4,096 updates × 512 env × 64 steps = 134,217,728 slots다. 작업 목록은 JSON matrix 외의 fallback이나 glob에서 추가하지 않는다.

기존 D-on의 update-0 평가는 동일 source actor의 결과이므로 새 rollout하지 않고 원본 curve hash와 `adapt_0.provenance.json`을 붙여 재사용한다. 최종 공통 trace에서는 기존 checkpoint를 `reused_result`, 새 50-job checkpoint를 `new_matrix_result`로 분리한다.

## 통합 smoke 결과

학습 smoke의 `adapt_1`은 기존 smoke root에 독립 구현 검증으로 보존한다. 본학습은 별도 main root에서 같은 source checkpoint로부터 각 job의 4,096 updates를 전부 실행한다.

| 환경/조건/커널 | 검사 결과 | 시간 |
| --- | --- | ---: |
| TICK D-task/fixed | task head 1, target terms 38,318 = valid samples, teacher minibatches 128 | 37.39 s |
| TICK D-frozen/mutant | teacher params·Adam·BatchRenorm·step 전부 동일, applied minibatches 0 | 19.46 s |
| CRAFT D-task/fixed | task head 1, target terms 52,102 = valid samples, teacher minibatches 128 | 26.14 s |
| CRAFT D-frozen/mutant | teacher params·Adam·BatchRenorm·step 전부 동일, applied minibatches 0 | 18.76 s |
| CRAFT D-off/mutant | teacher 갱신 128 minibatches, imitation 0, BC coefficient 0 | 23.21 s |

모든 저장 config가 지정 fixed·mutant kernel 및 arm과 일치했다. 다섯 smoke 총 시간은 124.97초였다. 과학 성능 결과는 아니다.

Trace evaluator는 TICK·CRAFT × fixed·mutant를 각 1 episode, 최대 4 steps로만 실행했다. 총 4 episodes·16 steps에서 header, 완전 state/observation, action/logits, exhaustive legal opportunity, 기존 좁은 opportunity, 독립 resource oracle, terminal record가 gzip JSONL에 기록됐다. 저장된 trace만 읽는 post-hoc replay도 4/4 통과했다. 이 replay는 transition kernel과 policy·teacher를 호출하지 않고 전후 상태와 행동에서 자원 보존량·결함 발동·최종 활용을 다시 계산한다. 역시 과학 성능 결과가 아니다.

## 예산

| 항목 | slots |
| --- | ---: |
| 본학습 50 jobs | 6,710,886,400 |
| 완료된 독립 smoke | 163,840 |
| **계획 총량** | **6,711,050,240** |
| 현재 본학습 잔여량 | **6,710,886,400** |
| 실패 실행 | 0 |
| 재개 추가 수집 | 0 |

Smoke ledger는 기존 RAID root에 보존한다. 본학습은 새 root의 별도 `collection_ledger.jsonl`에 fresh·resume·실패별 실제 update 행 수를 기록한다. Smoke 때문에 본학습 update를 줄이지 않는다.

## 저장공간과 예상 시간

Smoke 직전 `/home/ext_csv`는 90% 사용 중이어서 checkpoint 대상에서 제외했다. RAID는 7,488,342,851,584 bytes가 비어 있었다. 보수적 forecast는 다음과 같다.

- 최종 checkpoint 50개: 2,467,008,550 bytes.
- predecessor-successor overlap 상한: 2,467,008,550 bytes.
- 평가 artifact allowance: 25,769,803,776 bytes.
- logs/curves allowance: 8,589,934,592 bytes.
- 예상 잔여 쓰기 합계: 39,293,755,468 bytes.
- 별도 안전 reserve: 8,589,934,592 bytes.

Smoke tree는 493,327,473 bytes로 그대로 보존한다. 본학습 checkpoint는 새 `/raid/ext_csv/HackRL/runs/cross_environment_teacher_matrix_v1_main`에 기록한다.

기존 GPU 실행의 steady update는 TICK 0.09486초/update, CRAFT 0.08785초/update였다. 본학습 전체 순수 update 시간은 단일 worker 환산 약 5.16시간이다. 기존 science-curve 평가를 포함한 full-job 평균(TICK 535.02초, CRAFT 508.68초)으로는 50 jobs 순차 실행 약 7.21시간이다. 실제 wall-clock은 동시성, compilation, 현재 GPU 경합에 따라 달라진다.

공통 평가 runner는 완전한 task receipt와 artifact hash가 있는 결과만 재사용하며, 기록이 유실되거나 불완전하면 자동 재평가하지 않고 중단한다.

Trace smoke의 gzip 평균은 701.06 bytes/step이었다. 단순 비례 추정은 전체 step 상한에서 약 7.91 GB이며, 상태 분포에 따른 압축률 변화가 있어 capacity plan에서는 24 GiB를 배정했다.

## 공통 평가 상한

| 환경 | 정책 수 | episode 상한 | step 상한 |
| --- | ---: | ---: | ---: |
| TICK | 45 | 28,800 | 3,686,400 |
| PACK | 90 | 57,600 | 7,372,800 |
| CRAFT | 45 | 1,800 | 230,400 |
| 합계 | 180 | **88,200** | **11,289,600** |

fixed·mutant, mode·sample, 자연·기회 시작은 합치지 않고 별도 저장한다. 전체 합법적 opportunity와 과거 좁은 `opportunity_reached`도 별도 필드다.

## 과거 코드와의 미해결 차이

- TICK D-on은 dirty `fbaef1d...` 작업 트리에서 시작했다는 과거 기록이 있다. `9155a0d...`는 가장 가까운 사후 freeze이며 exact execution checkout 증명은 아니다.
- CRAFT 당시 `scripts/run_craft_remain_compare.py`와 `src/hackrl/craft_remain_gc.py`는 `fd201788...`과 일치하지만 현재 `src/hackrl/dual_leo.py`는 당시 hash와 다르다.
- TICK·CRAFT 과거 run에는 PACK처럼 모든 scientific source hash를 담은 독립 run manifest가 없다.
- 기존 curve는 완전 trace가 아니며 새 opportunity 정의로 소급 수정하지 않는다.

상세 path와 SHA-256은 `docs/manifests/cross_environment_teacher_matrix_v1_evidence.json`에 있다.
