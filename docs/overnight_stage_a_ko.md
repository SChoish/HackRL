# 1차 야간 스윕 계획

2026-09-28. 원격 [`e9446eb`](https://github.com/SChoish/HackRL/commit/e9446eb)의 몹 초기화 수정과 인과 지표를 받은 뒤, 첫 밤은 **18런, 28,311,552 transitions**로 잡는다.

기존 배치·목표·보상·H=512는 유지한다. 성공률 0은 정상 결과로 남긴다. 큐는 수정 fixture의 몹 재등장, 비수치, 저장 실패에서만 중지한다.

## 공통 설정

| 항목 | 값 |
|---|---|
| 과제 | R-M |
| 네트워크 | PPO 폭 256, 4 epoch × 8 minibatch |
| 롤아웃 | 128 env × 64 step |
| 학습률 | `2e-4` 명시. A1만 기존과 같이 anneal |
| 평가 | 32 episodes, mode / sample |
| 원시 산출물 | `runs/overnight_stage_a/` (체크포인트는 저장소 밖) |
| 큐 | `scripts/run_stage_a_queue.sh` |

GPU 0=fixed, GPU 1=mutant. 각 GPU는 A1을 끝낸 뒤 A2를 실행한다.

## A1: 오류 영향

기존(legacy) / 수정(patched) fixture × fixed/mutant × seed 0–2. 셀당 32 update = 262,144 transitions. 12런.

몹 슬롯 초기화만 바꾼 효과를 보기 위해 이전 6런과 같은 LR annealing을 쓴다.

## A2: 정상 학습 확인

수정 fixture × fixed/mutant × seed 0–2. 셀당 512 update = 4,194,304 transitions. 6런.

LR는 `2e-4` 고정. **0 / 262,144 / 1,048,576 / 4,194,304 transition**에서 체크포인트와 mode/sample 평가를 남긴다. 예산별 별도 학습은 하지 않는다.

## 아침 결정

| 결과 | 다음 행동 |
|---|---|
| 수정 후 fixed도 학습됨 | 저장한 정상 체크포인트로 사전학습량별 적응 비교 |
| fixed가 여전히 실패 | 철 채굴·곡괭이 제작·후속 채굴 중 병목 확인 |
| 양쪽 모두 즉시 성공 | 현재 fixture의 규모 스윕 중단, 후속 활용 부담을 별도 설계 |
| 수정 전후 차이가 큼 | 기존 실패 해석을 수정하고 새 환경 버전을 기준으로 진행 |

완료 시간은 첫 짧은 A1 셀의 실제 처리량으로 산정한다.
