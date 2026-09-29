# HackRL 기본 목표·훈련·스윕 v1

2026-09-29 · 설계 기준. 이 문서는 기본 능력 학습부터 goal conditioning을 사용하는 `hackrl_gc_v1` 트랙의 목표, 훈련, 평가와 스윕을 고정한다. 기존 고정 목표 PPO 코드·실행·결과는 별도 버전으로 보존하며 새 GC 결과와 합치지 않는다. 이 문서 자체는 구현 완료나 학습 실행을 뜻하지 않는다.

## 0. 범위와 선행 의존성

이 문서는 결함을 새로 정의하거나 기존 결함을 바꾸지 않는다. 현재 저장소에 있는 R/B/L 계열은 [버그와 과제 명세](bug_specifications_ko.md)와 [9개 과제 후보](benchmark_catalog_ko.md)를 계속 따른다.

제안 원문이 참조한 `exploit_benchmark_redesign_ko.md`는 현재 작업 트리와 Git 기록에 없다. 따라서 아래의 `TICK-CLAIM`과 `PACK-RESTORE`는 **목표·훈련 인터페이스 후보**로만 고정한다. 두 후보를 구현하거나 학습 큐에 넣기 전에는 각각의 결함 커널, fixed/mutant 전이, reset, 정상·예외 경로와 manifest를 별도 기준 문서에 먼저 커밋해야 한다. 이 문서만으로 두 후보를 기존 H0/H1/H2 또는 R/B/L 과제에 임의 대응하지 않는다.

버전 경계는 다음과 같다.

- 기존 고정 목표 PPO 트랙: 현재 코드와 기존 문서·결과를 그대로 보존한다.
- `hackrl_gc_v1`: 아래 12개 기본 목표로 정상 능력을 학습하고, 같은 checkpoint를 fixed 계속 학습과 mutant 적응으로 분기한다.
- `classic136`: 원래 Craftax-Classic 분포에서 원본 목표 목록을 참조·재현할 때만 사용하는 별도 프로파일이다.

## 1. 채택할 구조와 확인한 근거

LEO는 관측으로 판정할 수 있는 의미 목표를 Craftax-Classic에 136개, full Craftax에 512개 정의한다. 학습 중 한 번 이상 관측한 목표에서 명령 목표를 균등하게 뽑고, 목표를 달성하면 세계 상태를 유지한 채 새 목표를 선택한다. 가치 계산에서는 목표 달성을 pseudo-termination으로 처리한다.[1]

HackRL은 이 중 **goal-conditioned 정책, seen-goal curriculum, 상태 유지형 목표 전환**을 첫 기준선인 GC-PPO에 채택한다. LEO의 all-goals update는 별도 개입이다. 명령하지 않은 목표까지 학습하는 효과는 두 번째 방법인 Dual LEO (PPO)에서만 비교하며, all-goals 전이를 PPO의 on-policy 표본인 것처럼 재라벨링하지 않는다.

검토 기준은 공식 `purejaxgcrl` 커밋 [`eafccd9`](https://github.com/MichaelTMatthews/purejaxgcrl/tree/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7)이다.[2] 논문·코드의 숫자는 설계 출발점이며 HackRL fixture에서 검증된 설정이나 재현 결과가 아니다.

## 2. 목표 프로파일

### 2.1 `classic136`

원본 Classic 목록은 재고 54개, 도구 6개, 방향별 인접 블록 60개, 방향별 인접 몹·화살 16개다. 재고 목표는 10칸 one-hot 수량 슬롯과 맞으므로 단순한 `수량 >= n`이 아니다. 도구 목표처럼 여러 슬롯을 표시한 목표도 원본 `goal_achieved`에서는 하나 이상 일치하면 참인 OR 조건이다. 원료와 시설 상태를 동시에 요구하는 HackRL 목표가 필요하면 별도 AND predicate를 작성해야 한다.[2]

`classic136`을 단순한 무몹 작업장에 그대로 강제하지 않는다. 작업장에 없는 몹·용암·자원을 목표 수를 맞추기 위해 추가하지 않으며, 원래 Craftax 지도 재현과 결함 작업장 학습을 같은 프로파일로 기록하지 않는다.

### 2.2 작업장별 `workshop12`

각 후보는 fixed에서 정상적으로 달성할 수 있는 12개 기본 목표를 가진다. `raw_material`은 manifest가 지정하는 실제 재고 필드이며 보상 전용 가상 수치가 아니다. `adjacent/*`는 상하좌우 어느 방향이든 인접하면 참이고, `*_ge_k`는 실제 수치가 `k` 이상이면 참이다. 이 두 규칙은 원본 `classic136`에서 변경한 HackRL 정의다.

| 그룹 | TICK-CLAIM | PACK-RESTORE | 개수 |
| --- | --- | --- | ---: |
| 재고 | `inventory/raw_material_ge_{1,2,3}` | `inventory/raw_material_ge_{1,2,3}` | 3 |
| 접근 | `adjacent/raw_source`, `adjacent/reservation_device`, `adjacent/delivery_point` | `adjacent/raw_source`, `adjacent/storage`, `adjacent/delivery_point` | 3 |
| 정상 시설 상태 | `facility/reservation_present`, `facility/reservation_absent` | `facility/storage_empty`, `facility/storage_has_raw_material` | 2 |
| 정상 물체 상태 | `visible/crop_ripe`, `visible/crop_unripe` | `inventory/packed_item_ge_1`, `record/structure_present` | 2 |
| 목표 연결 | `delivery/count_ge_1`, `delivery/count_ge_3` | `delivery/count_ge_1`, `delivery/count_ge_3` | 2 |

이 문서의 평가 별칭 `deliver_3`은 두 후보 모두 canonical goal ID `delivery/count_ge_3`을 뜻한다. 실행 설정과 결과에는 별칭과 canonical ID를 함께 저장한다.

모든 predicate는 정책에 허용된 관측만으로 판정한다. 납품 누계나 구조 기록이 현재 관측에 없다면 fixed/mutant 공통 관측 필드로 먼저 명세하고 추가해야 한다. 숨은 계약 상태, mutation ID, 위반 판정기와 분석 전용 `info`는 predicate나 정책 입력에 쓰지 않는다.

객체 양화도 manifest에 고정한다. `adjacent/*`는 에이전트의 상하좌우 네 칸 중 해당 타입이 하나라도 있음을 뜻하고 `visible/*`는 관측 가능한 격자에 해당 상태의 객체가 하나라도 있음을 뜻한다. `facility/*`가 가리키는 시설은 후보 manifest의 고유 object ID로 결속한다. 같은 역할의 시설이 여러 개라면 ID별 목표로 분리하거나 집계 규칙을 명시하기 전에는 실행하지 않는다.

동시 수확, 복제, 음수 재고, 고아 객체와 위반 발생은 기본 목표로 만들지 않는다. PACK-RESTORE의 정상 구조 기록 생성과 포장은 학습할 수 있지만 복제 순서나 exploit 시연은 제공하지 않는다. 기본 목표가 기회 상태 방문을 늘리는 효과도 사전학습의 일부이므로 이를 자발적인 무지식 발견과 동일시하지 않는다.

### 2.3 predicate 계약

행동 `a_t` 이후 reset 전 관측을 `o_{t+1}^{terminal}`이라 할 때 목표 보상과 종료는 다음처럼 계산한다.

$$
r_t^g = \mathbf{1}\{P_g(o_{t+1}^{terminal})\},
\qquad
goal\_done_t = r_t^g.
$$

각 predicate는 다음 검사를 통과해야 한다.

1. 단일 목표 판정과 모든 목표를 벡터화한 판정이 같은 값을 낸다.
2. fixed/mutant에서 같은 관측은 같은 목표 결과를 낸다.
3. 경계값 `k-1`, `k`, `k+1`과 관측 밖·인접 네 방향을 검사한다.
4. 분석 전용 숨은 상태를 바꿔도 허용 관측이 같으면 판정이 바뀌지 않는다.
5. 실제 세계 종료 transition에서는 reset 관측이 아니라 reset 전 terminal observation으로 판정한다.

## 3. 환경·시간·인터페이스 계약

- 같은 후보의 fixed/mutant는 목표 ID, 목표 표현, 관측 공간과 행동 공간이 동일하다.
- 생존 변수를 고정한 첫 작업장의 시간 제한은 `H=128`을 초기값으로 둔다.
- 목표 성공은 세계 시간을 되돌리거나 세계 상태, 재고, 시설·장치 타이머를 reset하지 않는다.
- `goal_done`과 `world_done`을 별도 필드로 보존한다. `world_done`은 사망, 실제 환경 종료 또는 시간 제한이다.
- 실제 세계 종료에서는 reset을 정확히 한 번 수행한다. rollout 경계는 어느 종료에도 해당하지 않는다.
- fixed의 정상 경로가 `H` 안에 존재하는지 스크립트로 먼저 검증한다.
- 후보 사이에 관측, 행동 또는 목표 표현이 다르면 checkpoint를 그대로 공유하지 않는다.

## 4. 목표 학습 루프

1. 첫 행동과 첫 rollout 전에 512개 worker를 각각 reset하고, 그 직후의 초기 관측 512개 전체에서 학습 seed의 `seen_goals`를 초기화한다. 명령하지 않았어도 이 초기 관측에서 참인 목표는 포함한다. 빈 집합이면 임의 목표를 대신 뽑지 말고 fixture 초기 목표 구성 오류로 중단한다.
2. 정상 기능 사전학습 동안 `seen_goals`에서 균등하게 후보 목표를 뽑는다. 현재 관측에서 이미 참이면 `already_satisfied_at_command=1`인 0-step 성공으로 기록하고, learner transition·보상·gradient 없이 아직 참이 아닌 seen 목표에서 다시 뽑는다. 그런 목표가 하나도 없으면 sampler 상태 오류로 중단한다. 따라서 실제 명령 구간은 거짓인 predicate에서 시작한다.
3. rollout에서 새로 관측된 기본 목표는 update 경계에서 seed별 집합에 합친다. 같은 rollout 안에서는 재선택 후보가 되지 않는다. 목표 선택과 0-step 성공은 환경 transition이 아니다.
4. 실제 명령 목표에 행동을 한 뒤 `o_{t+1}^{terminal}`에서 predicate와 세계 종료를 함께 계산한다. 목표를 달성하면 `+1`, 그 외에는 `0`이다. 원본 업적·체력 보상, 계약 위반 보상과 내재 보상은 외재 목표 critic에 합산하지 않는다. 단일 명령 구간의 return 범위는 `[0,1]`이다.
5. `goal_done=True, world_done=False`이면 해당 명령의 value bootstrap과 GAE 연결을 끊고 `o_{t+1}^{terminal}`의 세계 상태를 그대로 다음 명령 시작 상태로 쓴다. `world_done=True`이면 목표 성공 보상을 먼저 기록한 뒤 환경을 정확히 한 번 reset하고 reset 관측에서 다음 목표를 뽑는다. 두 값이 동시에 참이어도 terminal 상태에서 새 목표를 뽑지 않는다. rollout 경계는 이 상태 머신을 바꾸지 않는다.
6. 목표별 방문·명령·0-step 성공·행동 후 성공 횟수와 성공까지 걸린 환경 transition 수를 기록한다. 0-step 성공을 새 기술 습득이나 정책 성공률로 세지 않는다. 이미 참인 목표를 아예 후보에서 제외하는 sampler는 별도 공통 버전으로만 비교한다.
7. 기본 능력 평가는 각 목표가 시작부터 참이 아닌 사전 정의 유효 상태에서 수행하고, 자연 reset 분포 결과도 별도 보존한다. 최종 활용 평가는 `deliver_3` 하나를 고정하며 중간 목표로 자동 전환하지 않는다. `deliver_3` 성공, 세계 종료 또는 시간 제한에서 평가를 끝낸다.

`seen_goals`는 학습 seed별로 관리한다. 정상 checkpoint를 분기할 때 fixed/mutant에 동일한 seen 상태를 복제한다. 주 적응 실험은 명령 목표를 `deliver_3`으로 고정해 두 커널의 sampler 분포 차이를 제거한다.

Dual LEO의 보조 Q는 동일한 12개 정상 목표 전체를 계속 학습할 수 있다. 명령 목표 전환은 모든 Q head의 종료가 아니다. 각 head `g`의 bootstrap은 `P_g(o_{t+1}^{terminal})` 또는 실제 `world_done`에서만 끊는다.

## 5. `hackrl_gc_v1` 학습 프로파일

공식 CraftaxGC PPO 기본값은 1,024 env, rollout 64, minibatch 2,048, epoch 1, 폭 1,024, LR `2e-4`, gamma `0.995`, GAE `0.95`, entropy `0.005`다. 지도에 convolution을 적용하고, critic 출력에는 sigmoid를 쓰며, LR annealing과 optimistic reset이 기본 활성이다.[2] 아래 프로파일은 이 중 일부만 채택한 HackRL 개발 설정이며 논문 재현 설정이 아니다.

| 항목 | `hackrl_gc_v1` | 계약 또는 변경 이유 |
| --- | --- | --- |
| learner | GC-PPO | 기존 PPO 배선과 새 GC 인터페이스를 연결하는 첫 기준선 |
| 관측·목표 | symbolic map + 보이는 수치·시설 상태 + goal 표현 | 숨은 계약·mutation ID 제외. 단일·all-goal 판정 일치 |
| 공간 인코더 | conv 1층, 32 features, 3×3, ReLU | 공식 PPO 구조 참고 |
| shared embedding / actor / critic | dense 1 / 2 / 4층, 폭 512 | 공식 깊이 유지, 폭 축소. 초기화 후 실제 파라미터 수 기록 |
| 병렬 환경 / rollout | 512 / 64 | update당 32,768 transitions |
| epoch / minibatch | 1 / 1,024 samples | update당 32 minibatches |
| optimizer / LR | Adam `eps=1e-5`, `2e-4` 고정 | 예산별 prefix의 초기 LR 경로를 같게 유지 |
| gamma / GAE | `0.995` / `0.95` | 공식 PPO 출발값 |
| clip / entropy / value coefficient | `0.2` / `0.005` / `0.5` | 공식 PPO 출발값 |
| gradient clip / value output | `1.0` / sigmoid | 단일 목표 return `[0,1]`; 내재 보상 critic에 자동 적용하지 않음 |
| reset | worker별 독립 reset | optimistic reset은 별도 처리량 옵션이며 평가에서는 사용하지 않음 |
| 목표 샘플링 | 기본 학습: 균등 seen / 적응: `deliver_3` | 목표 분포와 명령 횟수 기록 |
| checkpoint | 전체 train state + 환경 RNG·sampler·정규화·설정 | 파라미터 파일만 저장한 것을 resume 상태라 부르지 않음 |

`workshop12`의 goal 표현은 canonical 순서의 12차원 one-hot이다. 지도는 stride 1·`SAME` padding의 3×3 convolution 뒤 flatten하고, 관측 manifest가 열거한 수치·시설 feature 및 goal one-hot과 이어 붙인 뒤 shared dense 1층에 넣는다. actor와 critic은 여기서 각각 hidden dense 2층과 4층으로 분기하며 출력층은 이 개수에 포함하지 않는다. 모든 hidden activation은 ReLU다. convolution은 LeCun-normal kernel과 zero bias, hidden dense는 orthogonal `sqrt(2)`와 zero bias, actor 출력은 orthogonal `0.01`, critic 출력은 orthogonal `1.0`과 zero bias로 초기화한다. 첫 버전은 action mask와 learner-level running input normalization을 쓰지 않으며 각각 `none`으로 기록한다. 수치 feature의 고정 인코딩과 범위는 관측 manifest가 규정한다.

Resume checkpoint에는 적어도 model·optimizer state, global update와 environment-step counter, worker별 환경 상태와 RNG, worker별 현재 goal, episode 누계와 시간, `seen_goals`, sampler RNG, rollout cursor, normalization 통계, LR·BC 등 schedule 위치, 전체 실행 설정과 code revision을 넣는다. 저장 직전과 복원 직후 한 update를 같은 입력으로 실행해 action distribution, loss, 다음 train state가 허용 오차 안에서 같은지 검사한다.

기존 1,345차원 flat 입력을 공식 wrapper의 고정 slice로 무조건 변환하지 않는다. 새 시설 필드와 수치 범위를 포함한 관측 명세를 먼저 작성한다. 정상적으로 볼 수 있는 음수·큰 재고도 fixed/mutant 공통 인코딩에서 보존하되, 계약 위반 여부를 별도 feature로 제공하지 않는다.

원본 Classic wrapper는 10-class one-hot을 사용하므로 범위 밖 수치가 표현에서 사라질 수 있다. 또한 원본 코드에는 `float16(value)/10`을 정수로 단순 절삭할 때 수량 슬롯이 하나 낮아지는 문제를 피하려고 round를 적용한 수정이 있다.[2] HackRL 인코더는 수치 범위와 반올림 규칙을 명시적으로 검사한다.

## 6. 알고리즘 순서

| 순서 | 비교 | 묻는 것 |
| ---: | --- | --- |
| 1 | GC-PPO | 기본 목표와 상태 유지형 학습만으로 정상 조작·목표 연결이 확보되는가 |
| 2 | Dual LEO (PPO) | 같은 종류의 경험에서 명령하지 않은 목표까지 학습하는 것이 발견·이용에 도움이 되는가 |
| 3 | GC-PPO + RND | 기회 상태 접근 실패가 남을 때 탐험 보조가 결과를 바꾸는가 |
| 4 | PQN / LEO / Dual LEO (PQN), 필요 시 RNN | Q-learning 방식이나 관측 이력의 영향이 남는가 |

Dual LEO (PPO)의 출발값은 공식 코드의 policy cloning coefficient `0.1`, value cloning `0`, argmax policy target이다.[2] 보조 Q-network의 학습·고정 파라미터 수, optimizer step과 실측 시간을 별도로 보고한다. BC annealing을 사용하면 총 decay horizon을 실행 전에 고정하고 예산별 checkpoint에서 다시 시작하지 않는다. 정상→mutant 적응의 주 비교는 optimizer와 BC 스케줄을 이어가며, BC 재시작은 별도 개입이다.

GC-PPO와 Dual의 온라인 데이터는 이후 행동이 달라져 동일하지 않다. 성능 차이를 all-goals 학습만의 순수 효과로 단정하지 않는다. 필요하면 같은 고정 rollout에서 보조 학습 유무만 바꾸는 별도 실험으로 경험 획득과 학습 효과를 분리한다. 12-goal 집합에서도 Dual의 우위를 가정하지 않는다.

## 7. 훈련량과 스윕

LEO 논문의 PPO·PQN·LEO 튜닝은 방법별 100 runs × 200M timesteps이고, 주요 CraftaxGC 곡선은 5 seeds로 보고된다. 공개 sweep YAML의 seed 목록은 `[0,1]`이므로 논문의 5-seed 결과와 같은 실행 명세로 취급하지 않는다.[1,2]

첫 GC-PPO 실행은 다음 순서를 따른다. 기존 R-M 큐와 결과 디렉터리를 재사용하지 않는다.

| 단계 | 고정 조건·셀 | 비용과 산출물 |
| --- | --- | --- |
| 배선 교정 | TICK-CLAIM, GC-PPO, `deliver_3`, fixed/mutant × 3 seeds, 32 updates | 셀당 1,048,576; 총 6,291,456. 기존 6셀의 목적·예산을 유지하되 새 GC 버전으로 분리 |
| 기본 능력 | TICK-CLAIM fixed에서 `workshop12`, GC-PPO × 3 seeds, 512 updates | 셀당 16,777,216; 총 50,331,648. update 0/32/128/256/512 저장 |
| 정상 이력과 활용 | `B_pre` 3수준 `{0; 4,194,304; 16,777,216}` × fixed/mutant × 3 seeds, 각 적응 128 updates | 18 branches × 4,194,304 = 75,497,472. 명령 목표는 `deliver_3` 고정 |
| 두 번째 방법 | TICK-CLAIM에서 같은 기본·적응 설계의 Dual LEO (PPO) | 앞 단계의 측정·기능 검증 뒤 추가. 자동 실행하지 않음 |

위 행렬과 수치는 TICK-CLAIM 한 후보 기준이다. PACK-RESTORE는 자체 결함·관측·평가 manifest와 6셀 배선 교정을 통과한 뒤 같은 행렬을 별도 버전으로 복제하며, 현재 총비용에는 포함하지 않는다. GC-PPO 한 방법의 TICK-CLAIM 기본 능력+적응 비용은 125,829,120 transitions이며, 배선 교정·평가·튜닝은 별도다. `4.19M`과 `16.78M` 정상 checkpoint는 같은 학습 실행의 prefix이지 독립 seed가 아니다. `B_pre=0`도 같은 GC 인터페이스와 네트워크를 사용하고, fixed/mutant에 같은 초기 train state를 복제한다.

이 표는 **같은 추가 적응 예산**을 비교한다. 총비용 우위를 주장하려면 scratch에 `B_pre+B_adapt`를 주는 추가 대조군이 필요하다. `16.78M`이 정상 능력 확보를 보장하지 않으며, 성공 seed만 분기하지 않고 모든 seed를 포함한다. 개발·튜닝·최종 확인에 쓸 실제 seed 정수와 중복 금지 규칙은 결과를 보기 전에 run manifest에 고정한다. 최종 확인은 선택에 쓰지 않은 5개 seed를 기본값으로 하며, 자원 때문에 바꾸면 첫 최종 실행 전에 이유와 수를 고정하고 모든 비교 셀에 같이 적용한다.

후속 스윕은 한 축씩 수행한다.

- 훈련량: 기본 실행의 1.05/4.19/8.39/16.78M prefix 곡선을 사용한다. 필요할 때만 관련 셀 전체를 공통으로 67.11M까지 연장한다. LR 경로는 바꾸지 않는다.
- 크기: 폭 `{256,512,1024}`. 구조·기본 목표·예산·학습률을 고정하고 첫 비교는 GC-PPO 한 방법·한 후보에서 수행한다. 실제 actor/critic 파라미터 수와 wall time을 보고한다.
- 설정 민감도: 필요할 때 LR `{2e-4,3e-4}` × entropy `{0.002,0.005,0.01}`의 최대 6개 설정을 각 4.19M·공통 개발 seed 2개로 비교한다. 실행 수·예산을 맞추고 튜닝 비용을 보고한다. 정상 기본 목표의 검증 성능으로 선택하며 최종 exploit 시험을 선택에 쓰지 않는다.
- 목표 집합: `workshop12`와 `deliver_3`만 명령하는 조건을 비교해 curriculum의 기여를 본다. 접근·시설·수량 목표 제거는 별도 ablation이다. `classic136`은 해당 목표들이 가능한 일반 Craftax 분포로 확장할 때만 쓴다.

ETA는 새 conv/GC 구현의 compile·학습·평가 처리량을 측정한 뒤 계산한다. 기존 flat PPO 처리량이나 LEO 논문의 장비 시간을 환산하지 않는다.

## 8. 평가와 보고

기본 능력과 exploit 활용을 한 평균으로 합치지 않는다.

| 표 | 필수 항목 |
| --- | --- |
| 기본 능력 | 목표별 성공률, 목표 그룹별 평균, 명령 시작부터 이미 만족한 비율, 목표별 명령 횟수, 최초 seen update, 성공까지 걸린 시간 |
| 발견·활용 | 고정 `deliver_3` 성공, 위반 여부, 최초 위반 시점, 기회 노출, 직접 이득, 동결 정책 재현, 종료 사유 |

동결 평가는 학습 sampler, optimizer와 정규화 통계를 갱신하지 않는다. 학습 중 live success나 seen 목표만의 평균을 최종 성능으로 대체하지 않는다. 같은 배치에서 결정적 정책을 반복한 결과를 새 독립 표본으로 세지 않으며 평가 상태, layout, 학습 seed와 행동 RNG를 구분한다.

큐를 만들기 전에 후보별 평가 manifest를 불변 입력으로 저장한다. 여기에는 layout과 시작 상태 ID·생성법, 목표별 episode 수, horizon, deterministic/stochastic 행동 규칙, 행동 RNG seed, 반복 단위, 집계·분산·신뢰구간 규칙과 `기회 노출`, `직접 이득`, `위반`, `재현`의 판정식을 모두 넣는다. 하나라도 비어 있으면 결과 비교를 시작하지 않는다.

목표 성공률은 해당 predicate로 정의한 능력만 입증한다. 이를 일반적인 정상 조작 능력으로 넓혀 말하려면 훈련에 쓰지 않은 layout·시작 상태와 reward predicate를 재사용하지 않는 독립 state-transition/effect oracle에서 같은 결론이 나와야 한다. exploit 위반·직접 이득 판정기도 reward predicate 및 정책 입력과 분리해 구현하고, 같은 사전 상태·행동에서 fixed와 mutant가 달라지는 counterfactual fixture로 검사한다. 결함 설계자와 평가기 작성자가 같더라도 이 독립 검사를 생략하지 않는다.

보류 목표 평가에서는 `명령하지 않음`과 `학습하지 않음`을 구분한다. LEO가 해당 목표의 reward와 Q head를 업데이트했다면 명령 경험 밖의 전이 학습이지 완전히 새로운 목표 일반화가 아니다. 처음부터 모든 목표를 학습·평가하는 `workshop12` 실험도 새 목표 일반화라 부르지 않는다.

## 9. 구현 인계와 실행 전 게이트

구현 순서는 다음과 같다.

1. 후보별 결함·manifest 기준 문서와 fixed/mutant 정상·예외 경로를 확정한다.
2. `goal_catalog`와 관측 기반 predicate를 구현하고 단일/all-goal 동등성 검사를 추가한다.
3. 구조화 관측 인코더와 GC reward/goal-switching wrapper를 구현한다.
4. GC-PPO, 전체 checkpoint·resume, 명령 목표별 동결 평가를 구현한다.
5. terminal/reset 관측 순서와 정확히 한 번의 world reset을 회귀 검사한다.
6. 배선 교정 6셀 전에 파라미터 수, JIT compile, update 처리량과 checkpoint 왕복을 측정한다.
7. GC-PPO 게이트 통과 뒤에만 Dual LEO의 보조 Q, head별 종료와 BC 스케줄을 추가한다.

아래 조건을 모두 통과하기 전에는 학습 큐를 생성하지 않는다.

- 큐에 넣을 각 후보의 결함 명세와 manifest가 저장소에 존재한다. TICK-CLAIM만 실행할 때 PACK-RESTORE 명세까지 요구하지 않으며 그 반대도 같다.
- 실행 행렬의 후보, 방법, 단계, seed, budget과 평가 manifest가 결과 생성 전에 고정되어 있다.
- 모든 12개 목표가 fixed에서 달성 가능하고 관측만으로 판정된다.
- 독립 effect·violation oracle이 reward predicate와 분리되어 있고 fixed/mutant counterfactual fixture를 통과한다.
- 목표 성공, 세계 종료, 시간 제한과 rollout 절단의 bootstrap mask가 각각 검증된다.
- checkpoint 복원 뒤 정책, optimizer, RNG, sampler, seen 목표와 스케줄이 이어진다.
- 평가가 학습 상태를 변경하지 않고 reset 전 terminal observation을 사용한다.

## 출처와 확인 위치

1. Matthews et al., [**Goal-Conditioned Agents that Learn Everything All at Once**](https://arxiv.org/abs/2605.23551), arXiv:2605.23551v1 (2026). §3.1–3.2, §4–5.1, Appendix A·D·F·G에서 all-goals update, Dual LEO, 목표 수·구조, seen-goal curriculum, pseudo-termination, 튜닝 규모와 5-seed 결과를 확인했다.
2. 공식 [`purejaxgcrl@eafccd9`](https://github.com/MichaelTMatthews/purejaxgcrl/tree/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7). [`goal_listing_craftax_classic.txt`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/envs/craftax/goal_listing_craftax_classic.txt), [`craftax_goals.py`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/envs/craftax/craftax_goals.py), [`ppo.py`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/ppo.py), [`dual_leo_ppo.py`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/dual_leo_ppo.py), [`actor_critic_gc.py`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/models/actor_critic_gc.py), [`wrappers.py`](https://github.com/MichaelTMatthews/purejaxgcrl/blob/eafccd995ac2736df5fbc05ac4bb6b5d58cfcce7/wrappers.py)와 `sweeps/craftax/*.yaml`을 확인했다. 이는 코드 읽기이며 원본 학습 재현이 아니다.
