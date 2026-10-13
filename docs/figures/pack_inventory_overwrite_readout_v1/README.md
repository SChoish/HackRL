# PACK inventory-overwrite 발표용 시각자료

첨부 원고의 7개 본문 장에 넣을 독립 시각자료다. 아래 파일은 이 README와 같은 디렉터리에 있다. 각 그림은 밝은 배경용 고해상도 PNG와 편집 가능한 SVG를 함께 제공한다. 그림 내부 표기는 영문이고 이 문서의 캡션 안내는 한글이다. 전체 미리보기는 [contact_sheet.png](contact_sheet.png), 입력 파일 해시와 해석 주의사항은 [sources.json](sources.json)에 있다.

| 원고 장 | 파일 | 권장 캡션 |
|---|---|---|
| Exploitation and Recovery | [PNG](01_exploitation_and_recovery_paths.png) · [SVG](01_exploitation_and_recovery_paths.svg) | 실제 결정적 경로 한 사례다. 혼합 학습 정책은 mutant에서 8-step 지름길을 쓰고, fixed에서 네 번 차단된 뒤 22-step 우회로 성공했다. 모집단 평균이 아니다. |
| PPO Can Avoid Harmful Bugs | [PNG](02_ppo_avoids_harmful_bug.png) · [SVG](02_ppo_avoids_harmful_bug.svg) | 재고 덮어쓰기 전이와 seed 30–34의 최종 mode 성공·손실률을 함께 표시한다. 손실 후 복구가 아니라 이후 episode의 손실 예방 결과다. `Protect`는 첫 수확 전에 납품하거나 저장한 경로다. |
| Smaller Updates | [PNG](03_smaller_updates_tradeoff.png) · [SVG](03_smaller_updates_tradeoff.svg) | 1/4 LR은 정상 수행을 보존했지만 같은 예산에서 위험 시작 수를 덜 줄였다. 5% return 개선은 다섯 seed 중 두 seed가 주도했다. |
| More Frequent Training | [PNG](04_training_frequency_transfer.png) · [SVG](04_training_frequency_transfer.svg) | 모든 정책을 동일한 5% 발동 확률에서 비교한다. 30%·1/4 LR은 관측된 최고 평균일 뿐 최적점이 아니다. |
| Seed-Specific Failure | [PNG](05_seed_specific_failure.png) · [SVG](05_seed_specific_failure.svg) | p=1은 네 seed에서 비교군을 이겼지만 seed 35의 큰 정상 과제 실패가 평균을 뒤집었다. seed 35를 제외하지 않는다. |
| Measure Risk + Task | [PNG](06_risk_and_task_decomposition.png) · [SVG](06_risk_and_task_decomposition.svg) | return 분해는 측정치의 항등식이다. 정상 수행 저하의 내부 원인을 인과적으로 분리한 결과가 아니다. |
| Frozen Switch Diagnostic | [PNG](07_frozen_policy_switch_diagnostic.png) · [SVG](07_frozen_policy_switch_diagnostic.svg) | 제안된 무학습 oracle 진단이다. 위험 행동을 금지하거나 성공 행동을 강제하지 않는다. 아직 실행 결과가 아니다. |

## 읽는 법

- `fixed`는 정상 커널, `mutant`는 벽 통과 결함 커널이다.
- `risky start`는 정상 커널의 결정적 경로에서 위험 수확을 한 번 이상 선택한 seed–시작 상태 쌍이다. 실제 손실 횟수가 아니다.
- 1번 그림의 `C`는 delivery camp, `I`는 item이다.
- 4번 그림의 작은 점은 learner seed, 큰 점은 조건 평균이다.
- 5번 그림의 비교군은 30%·1/4 LR 조건이다.
- 7번 그림의 `pi_S`와 `pi_H`는 각각 동결된 출발 정책과 p=1 적응 정책이다.

## 사용

- PNG는 그대로 배치하고 자르지 않는 것을 권장한다. 글자·색·선 두께를 수정할 때는 SVG를 사용한다.
- 7번 그림은 결과가 아니라 다음 실험 설계임을 슬라이드에서 유지한다.
- 각 그림의 색 외에도 레이블, 기호, 수치를 함께 남겨 조건을 구분한다.

## 재렌더링

저장소 루트에서 실행한다.

```bash
/home/ext_csv/miniconda3/envs/offrl/bin/python \
  scripts/render_pack_inventory_overwrite_readout_assets.py
```

렌더러는 학습이나 rollout을 수행하지 않는다. 고정된 입력 JSON의 SHA-256이 바뀌면 실행을 중단한다.

## 공통 색상

- ink `#17232d`
- ground `#f6f8fa`
- adaptation blue `#3274a1`
- preservation blue `#55a6d9`
- harm red `#d95f59`
- p=1 purple `#6f4c9b`
- success green `#3b8f6b`
