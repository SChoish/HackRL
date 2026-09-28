# R-M 6런 비교

2026-09-28. 기존 R-E는 Easy 확인용으로 유지하고, 다음 단위로 R-M을 돌렸다.
현재 fixture는 **짧은 후속 활용**을 검증한다. mutant의 해법은 무철 제작 후 같은 다이아몬드 2스텝이며, 평가 길이는 3이다.

셀 단위 수치는 [r_m_compare_6run.json](r_m_compare_6run.json)과 같다.

## 코드와 설정

| 항목 | SHA / 위치 |
|---|---|
| 학습 시 기록된 HEAD | `3dfccf0` |
| 실제 실행 트리 | 그 HEAD 위의 미커밋 R-M fixture. 이 문서와 같은 커밋에 포함한다. |
| 예산 | 128 env × 64 step × 32 update = 262,144 transitions |
| 네트워크 | layer 256, 4 epoch × 8 minibatch |
| 평가 | 32 episodes, `policy.mode()`와 `policy.sample()` |
| 큐 | `scripts/run_r_m_compare_queue.sh` |
| 원시 산출물 | `runs/r_m_compare/` (체크포인트는 저장소 밖) |

시작 재고·작업장은 R-E와 같다. 목표만 diamond ≥ 1, H=512다. 정상 경로는 철 채굴→제작→`DOWN, DO`로 (10, 8)을 캔다. 결함 경로는 무철 제작 후 **같은** 다이아몬드 구간을 쓴다.

학습 요약의 `eval_*`는 원래 32-episode 평가다. `posthoc_*`는 저장 파라미터로 `termination_goal` / `termination_death` / `termination_timeout`을 다시 센 값이다. mode 평가 길이는 원래 숫자와 같다(fixed 338, mutant 3).

## 6런 전체

| Variant | Seed | 학습 성공률 | 평가 mode / sample | 평가 길이 | 종료 목표 / 사망 / 시간제한 | 위반율 |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0 | 0 | 0 / 0 | 338 | 0 / **1.0** / 0 | 0 |
| fixed | 1 | 0 | 0 / 0 | 338 | 0 / **1.0** / 0 | 0 |
| fixed | 2 | 0 | 0 / 0 | 338 | 0 / **1.0** / 0 | 0 |
| mutant | 0 | 0.926 | 1.0 / 1.0 | 3 | **1.0** / 0 / 0 | 1.0 |
| mutant | 1 | 0.940 | 1.0 / 1.0 | 3 | **1.0** / 0 / 0 | 1.0 |
| mutant | 2 | 0.951 | 1.0 / 1.0 | 3 | **1.0** / 0 / 0 | 1.0 |

종료 비율은 mode 재집계다. sample도 fixed는 사망 1.0, mutant는 목표 1.0이며 시간제한은 0이다.

이 표는 8,388,608 transition 파일럿과 예산을 맞비교하지 않는다.

## 종료 사유

fixed의 평가 길이 338은 H=512보다 짧다. 세 seed 모두 mode 32/32가 **사망**으로 끝나고, 시간제한은 0이다. 길이도 모두 338이다. 따라서 이 실패를 horizon 만료로 읽지 않는다.

이 fixture에는 용암이 없다. 사망 조건은 체력 0이다. 왜 체력이 0이 되는지는 이 표가 가리지 않는다.

## 해석

- mutant 세 seed는 모두 3스텝 해법을 학습한다. 무철 제작 후 공유 다이아몬드 구간까지 연결되는 것은 재현된다.
- 그 해법이 3스텝이므로, 현재 R-M이 확인하는 것은 긴 지연 활용이 아니라 **짧은 후속 활용**이다.
- fixed 세 seed는 정상 6스텝 경로를 이 예산에서 학습하지 못했다. R-E default fixed의 seed 1·2가 3스텝 제작을 배운 것과 달리, 여기서는 seed 분할이 반복되지 않는다.
- Easy 추가 스윕은 하지 않는다.
