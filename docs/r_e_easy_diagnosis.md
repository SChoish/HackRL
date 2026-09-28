# R-E Easy 진단과 12런 비교

2026-09-28. 보충 fixture 튜닝은 여기서 멈춘다. 기존 R-E는 Easy 확인용으로 유지한다.
다음 연구 단위는 R-M이며, 이 문서는 그 선택 근거로 커밋한다.

셀 단위 수치는 [r_e_replenish_compare_12run.json](r_e_replenish_compare_12run.json)과 같다.

## 코드와 설정

| 항목 | SHA / 위치 |
|---|---|
| 공식 PPO 파일럿 | [`98a7f86`](https://github.com/SChoish/HackRL/commit/98a7f86) |
| 진단 로그·체크포인트·시작 조건 분리 | [`d1280f4`](https://github.com/SChoish/HackRL/commit/d1280f4) |
| 보충 fixture와 12런 큐 | [`b469646`](https://github.com/SChoish/HackRL/commit/b469646) |
| 12런 실행 SHA | `b469646` |
| 12런 예산 | 128 env × 64 step × 32 update = 262,144 transitions |
| 네트워크 | layer 256, 4 epoch × 8 minibatch |
| 평가 | 32 episodes, `policy.mode()`와 `policy.sample()` |
| 큐 | `scripts/run_r_e_replenish_compare_queue.sh` |
| 원시 산출물 | `runs/r_e_replenish_compare/` (체크포인트는 저장소 밖) |

`recovered_episodes`는 목표 성공과 다르다. 목재가 목표 전에 0이 된 뒤 다시 1 이상이 되고, 같은 에피소드에서 목표에 도달한 횟수다.

## 보충 fixture fixed: 목표 성공 vs 소진 후 복구

| Seed | 학습 목표 성공 | 평가 mode / sample | 목재 소진 에피소드 | 소진 후 복구 성공 |
|---:|---:|---:|---:|---:|
| 0 | 0 / 2415 | 0 / 0 | 1496 (61.9%) | **0** |
| 1 | 1 / 3418 | 0 / 0 | 3392 (99.2%) | **0** |
| 2 | 1 / 2614 | 0 / 0 | 1960 (75.0%) | **0** |

세 seed 모두 평가 목표 성공은 0이다. seed 1·2의 학습 성공 1회는 복구 경로가 아니다 (`recovered_episodes = 0`).

## 12런 전체

| Fixture | Variant | Seed | 학습 성공률 | 평가 mode / sample | 평가 길이 | 목재 소진률 | 복구 성공 |
|---|---|---:|---:|---:|---:|---:|---:|
| default | fixed | 0 | 0 | 0 / 0 | 128 | 0.90 | 0 |
| default | fixed | 1 | 0.915 | 1.0 / 1.0 | 3 | 0.07 | 0 |
| default | fixed | 2 | 0.883 | 1.0 / 1.0 | 3 | 0.11 | 0 |
| default | mutant | 0 | 0.991 | 1.0 / 1.0 | 1 | 0.01 | 0 |
| default | mutant | 1 | 0.990 | 1.0 / 1.0 | 1 | 0.01 | 0 |
| default | mutant | 2 | 0.991 | 1.0 / 1.0 | 1 | 0.01 | 0 |
| r_e_replenish | fixed | 0 | 0 | 0 / 0 | 128 | 0.62 | 0 |
| r_e_replenish | fixed | 1 | 0.00029 | 0 / 0 | 128 | 0.99 | 0 |
| r_e_replenish | fixed | 2 | 0.00038 | 0 / 0 | 128 | 0.75 | 0 |
| r_e_replenish | mutant | 0 | 0.992 | 1.0 / 1.0 | 1 | 0.01 | 1 |
| r_e_replenish | mutant | 1 | 0.992 | 1.0 / 1.0 | 1 | 0.01 | 0 |
| r_e_replenish | mutant | 2 | 0.992 | 1.0 / 1.0 | 1 | 0.01 | 9 |

보충 mutant의 복구 1회·9회는 1스텝 결함 해법이 주된 성공인 가운데 나온 극소수다. 평가 복구율은 0이다.

이 표는 8,388,608 transition 파일럿과 예산을 맞비교하지 않는다.

목재 소진률은 재료 보존 능력의 증거가 아니다. 성공 정책은 3스텝에 끝나므로 128스텝 실패 정책보다 재료를 낭비할 기회 자체가 적다.

## 해석

- 확인된 것은 같은 설정에서 seed별 학습 결과가 갈렸다는 점이다. 기존 seed 1·2는 정상 3스텝 경로를 PPO가 학습하고, seed 0은 실패한다. “초기 성공 경험에 민감하다”는 아직 가설이다. 초기 성공 시점과 이후 학습 곡선을 비교하기 전에는, 탐험과 초기 성공 경험의 차이를 가능한 설명으로만 둔다.
- 보충 fixture는 복구 가능성만 바꾼 개입이 아니다. 자원 타일이 추가되면서 관측과 가능한 행동 경로도 달라진다. 이 결과가 지지하는 것은, 이 보충 배치가 해당 예산에서 정상 경로 학습을 개선하지 못했다는 점까지다. 재료 소진이 무관하다는 뜻은 아니다.
- Easy mutant는 결함 발동과 1스텝 활용을 확인했다. Easy에서 모델 크기 스윕을 확대할 이유는 작다.
- 재료 소진을 기존 R-E 실패의 주된 원인으로 단정하지 않는다.

## 다음 단위: R-M

정상 제작과 무철 제작 이후에 동일한 다이아몬드 채굴 구간을 둔다. 기존 정상 경로도 가능한 상태로 유지한다. 질문은 제작 결함을 발동하는 데서 그치지 않고, 얻은 도구를 후속 목표 달성에 연결하는가다. 이번 12런은 그 단계로 넘어가기 위한 개발 근거이며, Easy 추가 스윕은 하지 않는다.

R-M 6런 수치와 종료 사유는 [r_m_compare.md](r_m_compare.md)에 있다. 현재 fixture의 mutant 해법은 3스텝이므로 짧은 후속 활용 확인으로 읽는다.

## 파일럿 6런과 R-E 두 시작 진단

파일럿 6런(`98a7f86`, seed 0, 8,388,608 transitions)은 `runs/pilot_pe_seed0/`에 보존했다. L-E `eval_repeat_harvest_episode_rate`는 mutant 1.0 / fixed 0.0이다.

R-E fixed 짧은 진단(`d1280f4` 이전 작업 트리, `runs/re_fixed_diagnosis_20260928T114712Z/`):

| 시작 | 학습 성공 | 첫 성공 | mode / sample |
|---|---:|---|---:|
| 정상 reset | 0 / 2395 | 없음 | 0 / 0 |
| 철 채굴 후, 재료 유지 | 0.990 | update 0 | 1.0 / 1.0 |

후반 시작은 이동·채굴을 생략하므로, 목재 소진만이 실패 원인이라고 확정하지 않는다. 그 두 셀은 reset 시 `ever_iron` 시드 이전이다.
