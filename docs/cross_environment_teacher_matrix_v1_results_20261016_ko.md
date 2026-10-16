# 공통 Teacher Matrix v1 결과

이 문서는 [결과 JSON](manifests/cross_environment_teacher_matrix_v1_results.json)의 기계 생성 요약이다. 전체 seed별 수치와 artifact 경로는 JSON을 정본으로 삼는다. 실험 계약과 비교 근거는 [matrix 명세](manifests/cross_environment_teacher_matrix_v1.json)와 [evidence manifest](manifests/cross_environment_teacher_matrix_v1_evidence.json)에 있다.

## 실행 무결성

- 정책 180개, task 1,440개, episode 88,200개, step 1,916,510개를 완료했다.
- 독립 trace replay는 1,440/1,440개가 통과했다.
- 평가 commit은 `3750fa457a02a339bb94e2b4250ddd81de5f92b2`, 학습 실행 commit은 `6931d290c2f17b7f56bc55427a04cddf20931d0e`다.
- 1,440개 task는 180개 정책 × 정상/결함 kernel × mode/sample × 자연/기회 시작의 2×2×2 완전 조합이다. TICK·PACK은 seed별 mode 32회와 sample 128회, CRAFT는 mode 2회와 sample 8회를 각 시작 family에서 실행했다.
- replay는 저장된 전후 상태에 별도 자원 보존식을 적용하며 transition kernel·actor·teacher를 호출하지 않는다. 코드 경로는 독립적이지만 같은 프로젝트와 설계자가 만든 검증기이므로 외부에서 작성된 ground truth는 아니다.
- 아래 평균은 seed 동일 가중이며 kernel·mode/sample·시작 family를 합치지 않았다. 결측 cell은 없고 유의성·동등성 검정은 수행하지 않았다.

## 조건과 지표

- TICK-CLAIM, PACK-RESTORE, CRAFT-REMAIN은 각각 reservation-settlement, source-growth, crafting-input 자원 보존 결함 fixture다. `fixed`는 정상 전이, `mutant`는 결함 전이다.
- `Source`는 적응 전 정상 사전학습 checkpoint다. `D-on`은 전체 teacher head 갱신과 정책 모방을 유지하고, `D-task`는 해당 과제 head만 갱신하며 모방을 유지한다. `D-frozen`은 teacher 전체 상태를 동결한 채 모방하고, `D-off`는 teacher 갱신을 유지하되 정책 BC 계수를 0으로 둔다.
- `mode`는 greedy 행동 1회, `sample`은 확률적 행동 4회 반복이다. 자연 시작은 `natural_reset`, 합법적 기회 시작은 TICK·PACK의 `common_setup`과 CRAFT의 `path_check` 고정 패널이다.
- 성공률은 최종 과제 완료 비율이다. 활용률은 TICK·PACK에서는 결함으로 생긴 자원이 실제 납품된 episode, CRAFT에서는 결함 발동 뒤 보존 자원을 회수하고 초과 납품까지 한 episode의 비율이다.
- `U_{s,c}`는 seed `s`와 조건 `c`에서 mutant 학습 정책의 활용률에서 fixed 계속학습 정책의 활용률을 뺀 값이다. 조건 대비는 `ΔU_s = U_{s,left} − U_{s,right}`이고 표에는 seed 동일가중 `mean(ΔU_s)`를 보고한다. `양수 seed`는 반올림 전 `ΔU_s>0`인 seed 수다.

## 원고용 핵심 결과

> TICK·CRAFT에서는 모방을 유지한 task teacher 갱신 대 동결의 `ΔU`가 양수였고, PACK에서는 전체 teacher 갱신 중 모방 유지 대 중단의 `ΔU`가 양수였다. 두 비교는 서로 다른 개입이며 learner seed별 결과가 달랐다. 활용 채택의 증가는 정상 수행 보존을 보장하지 않았다.

아래 주 효과는 직접 활용률 차이가 아니라 seed별 `ΔU_s = U_{s,left} − U_{s,right}`의 평균 `mean(ΔU_s)`다. `U_{s,c}`는 같은 seed·조건에서 mutant 학습 정책과 fixed 계속학습 정책을 모두 mutant kernel에 평가한 활용률 차이다. `ΔΔJ`도 같은 차이-중-차이 구조의 할인 return이며 환경 사이의 절대 크기를 직접 비교하지 않는다. 정상 성능은 각 조건의 최종 mutant 학습 정책을 fixed kernel·mode·자연 시작에서 평가한 좌/우 값이다.

| 환경 | 개입: 좌 − 우 | 자연 ΔU | 기회 ΔU | 양수 seed 자연/기회 | 자연 ΔΔJ | 기회 ΔΔJ | 정상 성공률 좌/우 | 정상 return 좌/우 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TICK-CLAIM | D-task − D-frozen | +11.56pp | +25.16pp | 2/5 · 2/5 | +0.00092 | +0.00431 | 100.00% / 100.00% | 0.85420 / 0.85730 |
| CRAFT-REMAIN | D-task − D-frozen | +60.00pp | +60.00pp | 3/5 · 3/5 | +0.03314 | +0.03315 | 100.00% / 100.00% | 0.79858 / 0.79981 |
| PACK-RESTORE | D-on − D-off | +59.30pp | +59.06pp | 7/10 · 7/10 | +0.01413 | +0.01152 | 98.75% / 99.38% | 0.89098 / 0.90082 |

예를 들어 PACK 자연 시작의 `ΔU`는 `(64.61−0.31)−(5.00−0.00)=59.30pp`다. 최종 mutant 학습 정책끼리의 직접 활용률 차이 `64.61−5.00=59.61pp`와 다르다.

PACK `D-on − D-off`는 같은 historical run manifest·실행 코드·source checkpoint·seed·branch config를 공유하고 적응 중 모방만 달랐다는 해시·config 감사에 근거한다. 다른 run을 가로지르는 PACK 비교의 과거 코드 차이는 여전히 미해결이다.

원고 그림은 [PNG](figures/cross_environment_teacher_matrix_v1_seed_effects.png)와 [PDF](figures/cross_environment_teacher_matrix_v1_seed_effects.pdf)로 제공한다. [그림 데이터 CSV](figures/cross_environment_teacher_matrix_v1_seed_effects.csv)는 모든 learner-seed 점을 보존한다.

**그림 캡션.** 환경별 사전 지정 비교의 learner-seed 분포. TICK·CRAFT는 `D-task − D-frozen`, PACK은 별개의 `D-on − D-off` 개입이다. (a) `ΔU`는 mutant kernel의 sample 평가에서 조건별 mutant-vs-fixed 학습 효과를 다시 뺀 값이며 자연 시작과 합법적 기회 시작을 분리한다. 직접 활용률은 이 패널에 표시하지 않는다. (b) 두 조건의 최종 mutant 학습 정책을 fixed kernel의 mode·자연 시작에서 평가한 성공률이다. (c) (a)와 같은 차이-중-차이의 할인 return이다. 옅은 선은 같은 learner seed를 잇고, seed별 수평 jitter는 값이 겹쳐도 각 점을 보이게 한다. 검은 마름모는 seed 동일가중 평균이다. 88,200개 평가 episode는 각 정책 추정의 정밀도를 높이지만 독립 learner 수를 늘리지 않는다.

**Manuscript text.** In TICK and CRAFT, the prespecified `D-task − D-frozen` difference-in-differences in exploit adoption was positive on average: +11.56/+25.16 pp and +60.00/+60.00 pp for natural/opportunity starts, respectively. In PACK, the distinct prespecified `D-on − D-off` contrast was +59.30/+59.06 pp. These descriptive contrasts were concentrated in subsets of learner seeds, and no significance or equivalence test was performed. `D-on − D-task` comparisons were exploratory because of unresolved historical code differences and do not identify an effect of full-goal updating. Increased exploit adoption did not guarantee normal-task preservation.

## Source와 mutant 적응 정책의 공통 패널 결과

아래 활용률과 자연 성공률은 mutant kernel의 sample 평가다. 정상 성공률은 같은 정책을 fixed kernel의 mode·자연 시작에서 평가한 값이다. `D-*` 행은 mutant 환경에서 학습한 분기를 사용한다.

| 환경 | 정책 | 자연 활용률 | 기회 활용률 | 자연 성공률 | 정상 kernel mode 성공률 |
| --- | --- | ---: | ---: | ---: | ---: |
| TICK-CLAIM | Source | 1.09% | 1.25% | 100.00% | 98.75% |
| TICK-CLAIM | D-on | 100.00% | 100.00% | 100.00% | 100.00% |
| TICK-CLAIM | D-task | 100.00% | 100.00% | 100.00% | 100.00% |
| TICK-CLAIM | D-frozen | 88.44% | 74.84% | 100.00% | 100.00% |
| TICK-CLAIM | D-off | 49.53% | 37.50% | 100.00% | 100.00% |
| PACK-RESTORE | Source | 0.00% | 0.00% | 100.00% | 98.75% |
| PACK-RESTORE | D-on | 64.61% | 64.14% | 100.00% | 98.75% |
| PACK-RESTORE | D-task | 43.36% | 35.86% | 100.00% | 100.00% |
| PACK-RESTORE | D-frozen | 9.38% | 10.00% | 100.00% | 100.00% |
| PACK-RESTORE | D-off | 5.00% | 5.00% | 99.45% | 99.38% |
| CRAFT-REMAIN | Source | 0.00% | 0.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-on | 40.00% | 40.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-task | 60.00% | 60.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-frozen | 0.00% | 0.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-off | 0.00% | 0.00% | 80.00% | 80.00% |

## Mutant 경험의 활용 채택 효과

화살표는 fixed 계속학습 분기에서 mutant 학습 분기로 바뀐 활용률이고, `U`는 그 paired 차이의 seed 평균이다. 모두 mutant kernel의 sample 평가다.

| 환경 | 조건 | 자연 fixed→mutant | 자연 U | 양수 seed | 기회 fixed→mutant | 기회 U | 양수 seed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TICK-CLAIM | D-on | 0.00%→100.00% | +100.00pp | 5/5 | 0.00%→100.00% | +100.00pp | 5/5 |
| TICK-CLAIM | D-task | 0.00%→100.00% | +100.00pp | 5/5 | 0.00%→100.00% | +100.00pp | 5/5 |
| TICK-CLAIM | D-frozen | 0.00%→88.44% | +88.44pp | 5/5 | 0.00%→74.84% | +74.84pp | 4/5 |
| TICK-CLAIM | D-off | 0.00%→49.53% | +49.53pp | 5/5 | 0.00%→37.50% | +37.50pp | 3/5 |
| PACK-RESTORE | D-on | 0.31%→64.61% | +64.30pp | 7/10 | 0.08%→64.14% | +64.06pp | 7/10 |
| PACK-RESTORE | D-task | 0.00%→43.36% | +43.36pp | 5/10 | 0.00%→35.86% | +35.86pp | 5/10 |
| PACK-RESTORE | D-frozen | 0.00%→9.38% | +9.38pp | 1/10 | 0.00%→10.00% | +10.00pp | 1/10 |
| PACK-RESTORE | D-off | 0.00%→5.00% | +5.00pp | 1/10 | 0.00%→5.00% | +5.00pp | 1/10 |
| CRAFT-REMAIN | D-on | 0.00%→40.00% | +40.00pp | 2/5 | 0.00%→40.00% | +40.00pp | 2/5 |
| CRAFT-REMAIN | D-task | 0.00%→60.00% | +60.00pp | 3/5 | 0.00%→60.00% | +60.00pp | 3/5 |
| CRAFT-REMAIN | D-frozen | 0.00%→0.00% | +0.00pp | 0/5 | 0.00%→0.00% | +0.00pp | 0/5 |
| CRAFT-REMAIN | D-off | 0.00%→0.00% | +0.00pp | 0/5 | 0.00%→0.00% | +0.00pp | 0/5 |

## 사전 지정 비교

`ΔU`는 왼쪽 조건의 paired 활용 채택 효과에서 오른쪽 조건의 효과를 뺀 값이다.

| 비교 | 자연 ΔU | 기회 ΔU | 근거 상태 |
| --- | ---: | ---: | --- |
| TICK-CLAIM D-task − D-frozen | +11.56pp | +25.16pp | `new_same_code_and_initialization` |
| CRAFT-REMAIN D-task − D-frozen | +60.00pp | +60.00pp | `new_same_code_and_initialization` |
| PACK-RESTORE D-on − D-off | +59.30pp | +59.06pp | `historical_equivalence_verified` |
| TICK-CLAIM D-on − D-task | +0.00pp | +0.00pp | `exploratory_due_to_unresolved_code_difference` |
| CRAFT-REMAIN D-on − D-task | -20.00pp | -20.00pp | `exploratory_due_to_unresolved_code_difference` |
| PACK-RESTORE D-on − D-task | +20.94pp | +28.20pp | `exploratory_due_to_unresolved_code_difference` |

주요 matched 비교의 자연 시작 seed별 ΔU는 다음과 같다. 값은 percentage point다.

| 비교 | seed별 자연 ΔU |
| --- | --- |
| TICK-CLAIM D-task − D-frozen | 20:+12.50pp, 21:+45.31pp, 22:+0.00pp, 23:+0.00pp, 24:+0.00pp |
| CRAFT-REMAIN D-task − D-frozen | 20:+100.00pp, 21:+100.00pp, 22:+0.00pp, 23:+100.00pp, 24:+0.00pp |
| PACK-RESTORE D-on − D-off | 30:+99.22pp, 31:+0.00pp, 32:+96.09pp, 33:-1.56pp, 34:+99.22pp, 35:+100.00pp, 36:+50.00pp, 37:+0.00pp, 38:+100.00pp, 39:+50.00pp |

## 정상 수행 보존

아래 값은 fixed kernel·mode·자연 시작 성공률이다. Source는 적응 전 기준선이고, 두 학습 열은 각각 fixed 계속학습과 mutant 적응의 최종 정책이다.

| 환경 | 조건 | Source | fixed 학습 | mutant 학습 |
| --- | --- | ---: | ---: | ---: |
| TICK-CLAIM | D-on | 98.75% | 100.00% | 100.00% |
| TICK-CLAIM | D-task | 98.75% | 100.00% | 100.00% |
| TICK-CLAIM | D-frozen | 98.75% | 100.00% | 100.00% |
| TICK-CLAIM | D-off | 98.75% | 85.00% | 100.00% |
| PACK-RESTORE | D-on | 98.75% | 100.00% | 98.75% |
| PACK-RESTORE | D-task | 98.75% | 100.00% | 100.00% |
| PACK-RESTORE | D-frozen | 98.75% | 99.38% | 100.00% |
| PACK-RESTORE | D-off | 98.75% | 98.75% | 99.38% |
| CRAFT-REMAIN | D-on | 100.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-task | 100.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-frozen | 100.00% | 100.00% | 100.00% |
| CRAFT-REMAIN | D-off | 100.00% | 70.00% | 80.00% |

## 해석

- 신규 동일 코드·초기화 비교에서 task-goal teacher를 갱신한 `D-task`의 활용 채택 효과는 `D-frozen`보다 TICK 자연 +11.56pp, 기회 +25.16pp 높았고, CRAFT에서는 자연 +60.00pp, 기회 +60.00pp 높았다. 방향은 두 환경에서 같았지만 양의 차이는 각각 2/5와 3/5 seed에 집중됐다.
- 동등성이 확인된 PACK `D-on − D-off` 비교에서 활용 채택 효과 차이는 자연 +59.30pp, 기회 +59.06pp였다. 자연 시작 seed별로 7개 양수, 2개 0, 1개 음수였으므로 모든 seed에 일관된 효과는 아니다.
- 전체 목표 갱신 `D-on`과 task-only `D-task`의 탐색적 ΔU는 TICK +0.00pp, CRAFT -20.00pp, PACK +20.94pp였다. 과거 코드 차이가 남아 있어 전체 목표 갱신의 단일 요인 효과로 해석하지 않는다.
- 정상 수행은 조건과 학습 분기에 따라 달랐다. TICK D-off fixed 학습은 85.00%, CRAFT D-off는 fixed 70.00%·mutant 80.00%였다. 활용 채택과 정상 수행 보존은 별도 결과다.
- Source의 sample 활용은 TICK 자연 1.09%·기회 1.25%, PACK 자연 0.00%·기회 0.00%, CRAFT 자연 0.00%·기회 0.00%였다. 최종 활용률과 mutant 경험으로 생긴 paired 증가분 `U`를 구분해야 한다.

## 주장 경계

- TICK·CRAFT의 신규 조건과 historical D-on/D-off 간 비교, 그리고 서로 다른 PACK run을 가로지르는 비교는 탐색적이다.
- 결과 범위는 이 세 fixture와 고정 패널이다. 같은 방향의 환경 재현은 관측됐지만 teacher가 어떤 내부 기전으로 행동을 바꿨는지는 이 평가만으로 규명하지 않는다.
- seed 수는 TICK·CRAFT 5개, PACK 10개이며 seed SD와 전체 per-seed 값은 결과 JSON에 보존했다. 별도 신뢰구간이나 유의성·동등성 검정은 하지 않았다.
- 과거의 좁은 opportunity 지표는 소급 변경하지 않았으며, 본 문서는 새 고정 패널의 exhaustive legal opportunity family 결과를 별도로 보고한다.
