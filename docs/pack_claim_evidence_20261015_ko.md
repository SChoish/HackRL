# PACK 적응 연구의 주장–근거 구조

## 중심 주장

PACK 결과는 하나의 “적응 성공률”로 합치지 않는다. **자연 시작에서의 활용 발현, 기회 상태에서의 활용 가능성 유지, 정상 과제 수행 보존은 서로 다른 결과**다. 한 지표의 개선으로 다른 두 결과까지 개선됐다고 해석하지 않는다.

| 구분 | 이 자료에서 묻는 것 | 기준 비교 |
| --- | --- | --- |
| 활용 발현 | 자연 시작에서 실제 활용 행동열이 나타나는가 | 적응 정책 대 동결 source와 같은 예산의 fixed 계속학습 |
| 활용 가능성 유지 | 활용하기 쉬운 기회 상태에서 기존 가능성이 남아 있는가 | 적응 후 정책 대 같은 seed의 동결 source |
| 정상 수행 보존 | 결함이 없는 환경에서 원래 과제를 계속 수행하는가 | 적응 후 정책 대 같은 seed의 동결 source |

## 조건과 지표

| 용어 | 이 문서에서의 뜻 |
| --- | --- |
| source | 정상 환경 사전학습을 마치고 추가 갱신하지 않은 같은 seed의 출발 정책 |
| PACK | 곡물 수집·보관·배송 과제와 그 전이 결함을 구현한 내부 gridworld fixture |
| Dual | PPO actor–critic에 Q-network teacher의 행동 모방 손실을 함께 적용한 학습 조건 |
| fixed 계속학습 / mutant 적응 | 같은 source와 학습 설정에서 각각 정상 커널과 결함 커널로 4,096 update를 수행한 조건 |
| `D-on` / `D-frozen` / `D-delivery` | 적응 중 모든 teacher 목표를 갱신하며 모방 / teacher 전체를 동결하고 모방 / 배송 목표 head만 갱신하며 모방 |
| 자연 시작 / 기회 시작 | 일반 reset 상태 / 활용에 필요한 물리 상태 일부를 이미 갖춘 진단 상태 |
| 활용률 | 보존량을 위반해 생성된 곡물이 실제 배송된 episode의 비율. 행동 호출이나 기회 도달만으로 세지 않음 |
| mode / sample | categorical actor의 argmax 직접 실행 / 같은 상태에서 categorical 행동을 반복 표집한 평가 |
| 5% exact return | 정상 커널의 deterministic 경로 길이와 위험 수확 횟수로 5% 독립 발동 확률의 기대 할인 return을 계산한 값 |
| 위험 시작 수 | 32개 시작 상태 중 deterministic 정상 경로가 위험 수확을 포함한 상태 수의 seed 평균. 작을수록 위험 노출이 적음 |

## 주장별 근거

| 연구축 | 현재 허용되는 주장 | 직접 근거 | 해석 경계 |
| --- | --- | --- | --- |
| Teacher 갱신 대 동결 | 기존 개발 패널에서 적응 중 teacher 갱신을 유지한 조건이 teacher를 동결한 조건보다 높은 활용률을 보였다. | 최종 자연 시작 sample에서 `D-on − D-frozen = +54.92%p`였다. mode 차이는 `+54.38%p`였다. seed 30–39의 paired 결과다. | 시작 시점 개입은 teacher 갱신 여부지만 이후 방문 상태와 정책이 함께 달라진 전체 매개 효과다. Q-network의 독립적인 인과 효과나 성공 전문가 정책의 전달을 뜻하지 않는다. 같은 seed와 개발 패널을 재사용한 회고적 분석이다. |
| 전체 목표 대 배송 목표 teacher 갱신 | 전체 목표를 함께 갱신하는 것이 배송 목표만 갱신하는 것보다 추가 이득을 준다는 결론은 새 배치에서 확보되지 않았다. | 정책을 보지 않고 만든 새 패널과 seed 200–204에서 자연 시작 mutant mode 활용률의 `D-on − D-delivery` 평균은 `0.00%p`, seed 차이는 `0, −25, 0, +25, 0%p`, bootstrap 95% 구간은 `[-15, +15]%p`였다. | 평균 0은 두 조건의 동등성을 입증하지 않는다. 이 비교는 적응 중 TD target 범위를 바꾸며, 사전학습에서 다목표 teacher가 필요했는지는 시험하지 않는다. |
| Mutant 적응 대 fixed 계속학습 | 학습 환경은 활용의 발현과 유지에 영향을 준다. 자연 시작에서는 mutant 적응 후 활용률이 source와 fixed 계속학습보다 높았다. | seed 200–204의 mutant 커널 sample 활용률은 자연 시작에서 source `0.94%`, fixed 계속학습 `0%`, mutant 적응 `10.00%`였다. | 자연 시작 증가는 관측됐지만 기회 도달, 행동 선택, 실행 순서 중 무엇이 달라졌는지는 분리되지 않았다. 이를 새로운 활용 능력 자체의 획득으로 확대하지 않는다. 기존 HOLDOUT V2 패널을 결과 확인 후 다시 쓴 탐색적 대조이며 새로운 holdout이 아니다. |
| 기회 시작의 활용 유지 | fixed 대비 mutant의 차이 전체가 새 활용 행동의 형성을 뜻하지 않는다. fixed 계속학습에서 기존 활용 가능성이 감소한 효과가 포함된다. | 기회 시작 mutant 커널 sample 활용률은 source `17.19%`, fixed 계속학습 `3.75%`, mutant 적응 `16.56%`였다. mutant 적응은 fixed보다 `+12.81%p`였지만 source보다 높지 않았다. | 기회 시작에서 확인된 것은 source 수준에 가까운 유지다. fixed의 감소와 mutant의 유지가 섞인 차이를 “새 능력 획득”으로 부르지 않는다. |
| 정상 수행 | fixed와 mutant 학습 분기 모두 동결 source보다 정상 sample 수행이 낮았고, 분기 간 차이는 seed별로 불균일했다. | 정상 커널 자연 sample 성공률은 source `47.81%`, fixed 계속학습 `40.63%`, mutant 적응 `37.19%`였다. fixed−mutant는 `+3.44%p`였지만 seed 방향은 `+/+/0/−/−`였고 bootstrap 구간은 0을 포함했다. | fixed 계속학습의 상대적 우위를 일반적인 정상 수행 보존 효과로 주장하지 않는다. mode 결과와 sample 결과도 분리한다. |
| 정상 행동 앵커 | source-policy KL 앵커는 정상 mode 수행 손상을 줄였지만, 위험 회피와 정상 수행 보존이 함께 개선되지는 않았다. | seed 30–34의 TEST에서 정상 mode 성공률은 일반 PPO `96.25%`, 앵커 PPO `100%`였다. 5% exact return은 일반 PPO `0.85208`, 앵커 PPO `0.87584`였지만 paired seed 방향은 2개 양수, 3개 음수였다. 위험 시작 수는 일반 PPO `15.2/32`, 앵커 PPO `22.4/32`였고 다섯 학습 job의 누적 손실도 `1,392 → 1,530`으로 늘었다. 앵커의 source 대비 TEST exact return 차이는 `−0.00009`였다. | 평균적인 상대 우위는 주로 큰 정상 수행 손상을 막은 seed가 설명한다. 앵커가 출발 정책보다 유리한 회피 전략을 안정적으로 학습했다거나 일반적인 안전 적응을 달성했다고 주장하지 않는다. `VALIDATION`과 `TEST`라는 이름을 썼지만 두 패널 모두 개발 자료다. |

## 통합 해석

> Teacher 갱신이 포함된 Dual 학습은 기존 개발 패널에서 결함 활용을 촉진했지만, 새 배치에서는 전체 목표 teacher 갱신이 배송 목표 갱신보다 낫다는 추가 이득이 확인되지 않았다. 동일한 D-delivery 학습에서도 mutant 환경은 fixed 계속학습보다 높은 활용률을 만들었으나, 자연 시작의 활용 발현 증가와 기회 시작의 기존 활용 가능성 유지는 구분해야 한다. 정상 수행은 두 분기 모두 source보다 낮았고, 행동 앵커는 수행 손상을 줄이는 대신 같은 예산의 위험 회피를 제한했다. 따라서 활용률 증가, 활용 가능성 유지, 정상 수행 보존은 별도로 측정하고 보고해야 한다.

## 발표와 원고에서 피할 표현

- “최종 teacher가 완성된 전문가 전략을 전달했다.”
- “fixed 대비 차이는 모두 새로운 활용 능력의 획득이다.”
- “전체 목표 teacher가 배송 목표 teacher보다 우수하다.”
- “정상 수행을 보존하면 위험 회피도 보존된다.”
- “앵커가 희귀 결함 적응을 해결했다.”

## 근거와 범위

- Teacher 갱신 대 동결: [실험 명세](manifests/pack_restore_pretrained_teacher_freeze_v1.json), `/raid/ext_csv/HackRL/runs/pack_restore_pretrained_teacher_freeze_v1/result.json`, SHA-256 `2da51da872745ae7d59bb8690827610405ec59ba0ca3346daa401d9e404b68a6`.
- 전체 목표 대 배송 목표: [새 배치 명세](manifests/pack_teacher_holdout_v2.json), `/raid/ext_csv/HackRL/runs/pack_teacher_holdout_v2/result.json`, SHA-256 `80d46f7b0f5448bf39c8b78d41009a799f952a1c0618e963e7211425724a554b`.
- Mutant 대 fixed 계속학습: [대조 실험 명세](manifests/pack_teacher_fixed_control_v1.json), `/raid/ext_csv/HackRL/runs/pack_teacher_fixed_control_v1/result.json`, SHA-256 `48a181d3b28522a32c7ecd233febee34d630dba9e2e17818707af2d8c2e57ce3`.
- 정상 행동 앵커: [실험 명세](manifests/pack_inventory_overwrite_behavior_anchor_ppo_v1.json), `/raid/ext_csv/HackRL/runs/pack_inventory_overwrite_behavior_anchor_ppo_v1/aggregate.json`, SHA-256 `c7a77f78974af6157b72417fbe5e584958eac9d430cb4641cc57d95a862f73c3`.

모든 평균은 learner seed에 같은 가중치를 둔다. episode와 sample repeat를 독립 반복으로 세지 않는다. 표시한 bootstrap 구간은 learner seed를 20,000회 재표집한 percentile 구간이며, 구간을 표시하지 않은 앵커 결과에는 모집단 유의성 주장을 붙이지 않는다. 활용 oracle과 패널은 내부에서 구성했으므로 결론의 범위는 PACK fixture 안의 기제 비교다. 이 문서는 네 실험을 하나의 모집단 효과나 공통 인과 사슬로 합산하지 않으며, 각 실험의 패널·seed·개입 범위 안에서만 결과 범주를 연결한다.
