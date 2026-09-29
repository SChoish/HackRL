# TICK-CLAIM GC 6셀 보정 실험 결과

2026-09-29 · fixed/mutant × learner seed 0–2 · 셀당 1,048,576 transitions

## 결론

계획한 6셀 배선 보정 실험은 완료됐다. 여섯 셀 모두 정확한 예산, 파라미터 2,640,181개, update 32의 전체 체크포인트, 동결 평가 상태 불변성, 구현 SHA 계보와 source hash 검사를 통과했다. 총 학습량은 6,291,456 transitions이며 집계 오류는 0개다.

mutant에서는 세 seed 모두 학습 중 독립 oracle 위반과 위반 곡물 납품이 발생했다. 동결 stochastic 평가에서도 세 seed 모두 위반을 보였고 seed 0과 1은 위반 곡물을 납품했다. 따라서 결함 기회, 중복 생성, 반복 사용, 납품 기여를 구분하는 계측 배선은 작동한다.

그러나 안정적인 활용 정책이나 mutant의 성능 우위는 확인되지 않았다. 최종 greedy 정책은 fixed와 mutant 모두 성공률과 위반률이 0이었다. stochastic 성공률의 seed 평균은 fixed 14.71%, mutant 13.41%였고 seed별 편차가 컸다. 이 결과는 일반적인 버그 발견 능력이나 학습 우위를 주장하지 않는다.

## 실행 및 무결성

| 항목 | 결과 |
| --- | ---: |
| 셀 | 6 / 6 통과 |
| 셀당 transitions | 1,048,576 |
| 전체 transitions | 6,291,456 |
| 셀당 updates | 32 |
| workers × rollout | 512 × 64 |
| 실제 파라미터 수 | 2,640,181 |
| final checkpoint 검증 | 6 / 6 통과 |
| 동결 평가 불변성 | 6 / 6 통과 |
| source hash 일치 | 통과 |
| 전체 저장소 테스트 | 126 passed, 1 warning |

CPU 동기화 학습의 셀별 steady-state 처리량은 약 5,374–5,429 transitions/s였고, pre-run full-shape 검증은 6,168.87 transitions/s를 기록했다. 처리량은 기능 게이트이며 variant 성능 비교 지표로 사용하지 않는다. 동기화 전 예비 queue는 optimizer 완료 전 시간을 잰 데다 최종 fresh-process 경로에서 정책 결과가 재현되지 않아 전체를 제외했다.

## 학습 중 사건

| variant | seed | goal 성공 | 기회 노출 | 예약 생성 | 위반 | 반복 위반 | 위반 곡물 납품 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 0 | 309 | 1,276 | 9,407 | 0 | 0 | 0 |
| fixed | 1 | 6 | 202 | 1,590 | 0 | 0 | 0 |
| fixed | 2 | 157 | 736 | 5,752 | 0 | 0 | 0 |
| mutant | 0 | 312 | 1,285 | 10,980 | 70 | 2 | 8 |
| mutant | 1 | 134 | 1,145 | 6,885 | 48 | 1 | 2 |
| mutant | 2 | 68 | 726 | 5,374 | 25 | 3 | 1 |

fixed의 위반 0은 음성 대조군과 일치한다. mutant는 모든 seed에서 중복 지급을 만들고 반복했으며 일부 초과 생성분을 납품했다. 이는 성공률과 별개로 결함 발견·사용 경로가 학습 중 존재했음을 뜻한다.

## 동결 평가

평가는 validation의 natural reset 32개와 실제 fixed prefix를 실행한 common setup 32개를 사용했다. greedy mode는 상태당 1회, stochastic sample은 상태당 4회로 각각 64개와 256개 episode다. 모든 평가는 학습 상태를 변경하지 않았다.

| 지표, stochastic 전체 | fixed seed 0/1/2 | mutant seed 0/1/2 | seed 평균 fixed / mutant |
| --- | --- | --- | --- |
| 성공률 | 23.83% / 0.00% / 20.31% | 22.27% / 16.41% / 1.56% | 14.71% / 13.41% |
| 위반률 | 0 / 0 / 0 | 1.56% / 1.56% / 1.17% | 0 / 1.43% |
| 반복 위반률 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| 위반 곡물 납품률 | 0 / 0 / 0 | 1.17% / 0.78% / 0 | 0 / 0.65% |
| 평균 길이 | 116.89 / 126.00 / 117.80 | 117.70 / 120.65 / 125.61 | 120.23 / 121.32 |

paired mutant-minus-fixed 성공률은 seed별 -1.56, +16.41, -18.75 percentage points, 평균 -1.30 points였다. 성공률만으로는 결함 채택을 판정할 수 없다는 검토의 우려가 그대로 나타났다. 위반과 위반 곡물 납품은 관측됐지만 greedy 정책에는 남지 않았고 stochastic 재현율도 낮았다.

## 재현성과 해석 범위

최종 구현에서 fresh Python process로 fixed seed 0 전체 셀을 독립 재실행했다. 두 실행의 32-update metrics와 31,891,723-byte 전체 checkpoint가 각각 byte-identical했다. 이 검사는 한 CPU/software runtime의 한 셀에 한정되며 다른 seed, variant, 하드웨어의 결정성을 주장하지 않는다.

- 이 실험은 사전 등록된 배선 보정이며 성능 임계값이나 유의성 검정을 두지 않았다.
- 세 learner seed는 안정적인 채택률이나 모델 규모 효과를 추정하기에 부족하다.
- 48개 layout은 하나의 workshop template에 대한 D4, 위치, 벽 변형이지 독립 메커니즘 48개가 아니다.
- oracle은 정책 입력과 보상에서 분리됐지만 합성 fixture와 작성자 맥락을 공유한다.
- fixed와 mutant 모두 정상 전략으로 목표를 달성할 수 있다. 위반률, 반복 위반, 위반 곡물 납품을 성공률과 따로 해석해야 한다.
- 이번 결과만으로 알고리즘, 폭, 장기 예산 스윕으로 확대하지 않는다.

## 재현 자료

- 동결 프로토콜: `docs/manifests/tick_claim_gc_pilot_v1.json`
- 사전 검증: `docs/manifests/tick_claim_gc_v1_validation.json`
- 구현 및 hash attestation: `docs/manifests/tick_claim_gc_v1_resolved.json`
- 기계 판독 집계와 셀별 artifact hash: `docs/manifests/tick_claim_gc_pilot_v1_results.json`
- fresh-process 재현 검사: `docs/manifests/tick_claim_gc_pilot_v1_reproducibility.json`
- 최종 raw run: `runs/tick_claim_gc_pilot_v1_synced/{fixed,mutant}_seed{0,1,2}/`
- 실행기·검증기·집계기: `scripts/run_tick_claim_gc.py`, `scripts/validate_tick_claim_gc.py`, `scripts/summarize_tick_claim_gc_calibration.py`
