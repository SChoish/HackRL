# HackRL

기존 강화학습 환경에 실제 게임 버그의 **메커니즘을 참고한 작은 변형**을 적용하고, 에이전트가 이를 발견·재현·활용하는 능력을 연구합니다.

핵심 질문은 **알고리즘, 탐험 방식, 신경망 크기에 따라 환경의 예외적 동작을 활용하는 능력이 어떻게 달라지는가?** 입니다.

- [연구계획서 v2](docs/research_plan_ko.md): 연구 질문, 가설, 후보 선택과 진행 기준
- [GBGallery 분류·9개 과제 후보](docs/benchmark_catalog_ko.md): 공식 분류, 오류 계열 × 잠정 난이도, 첨부 사례의 시사점
- [버그·과제 명세](docs/bug_specifications_ko.md): 코드 변경 지점, 정상/예외 경로, 독립 판정기
- [실험 프로토콜](docs/experiment_protocol_ko.md): 온라인 설정·스윕 예산·평가와 오프라인 확장
- [GC 기본 목표·훈련·스윕 v1](docs/gc_training_sweep_ko.md): LEO를 참고한 12-goal 정상 사전학습, fixed 계속 학습/mutant 적응과 단계별 예산. 설계만 확정했으며 아직 구현·학습하지 않았다.
- [R-E Easy 진단·12런 비교](docs/r_e_easy_diagnosis.md): 목표 성공과 재료 소진 후 복구 성공을 구분한 수치. 보충 fixture 튜닝은 여기서 멈춘다.
- [R-M 6런 비교](docs/r_m_compare.md): 짧은 후속 활용(3스텝)과 평가 종료 사유. 원시 수치는 `docs/r_m_compare_6run.json`.
- 과제 구성: 자원 전제 조건·공간 범위·상태 갱신의 3개 계열 × Easy/Medium/Hard, 난이도당 3개씩 총 9개 후보입니다. 같은 결함을 다른 맥락에서 평가하며 난이도는 파일럿으로 검증합니다.
- 구현 순서: 세 결함의 회귀 검사 → Easy 3개 → 활용 이득이 검증된 Medium → Hard. 생존·수확의 중간/어려움 후보는 특히 실행 가능성 검증이 필요합니다.
- 기존 고정 목표 트랙: 온라인 PPO/PPO+RND → PPO 크기 비교와 PQN-FF → 필요한 경우 기억·적응 진단
- 새 GC 트랙: GC-PPO → Dual LEO (PPO) → 필요한 경우 RND·PQN·RNN. 기존 결과와 별도 버전으로 기록

현재 세 root mutation, Easy 3개, Medium R-M 개발 fixture와 고정 목표 PPO 파일럿 학습·평가 흐름이 구현되어 있습니다. GC-PPO와 TICK-CLAIM/PACK-RESTORE 작업장은 아직 설계 단계입니다. Hard와 나머지 Medium, 본 실험 결과도 아직 없습니다. 아래 계획의 변형은 원본 환경에서 확인된 취약점이나 GBGallery 버그의 직접 재현을 의미하지 않습니다.

## 구현 상태

Craftax 1.6.1을 기반으로 세 root mutation과 독립 전이 판정기를 구현했습니다.

- `fixed`: 원본 Craftax-Classic 전이
- `h0`: 식물 수확 후 성장 나이 초기화 누락
- `h1`: 철 0개에서 철 곡괭이 제작 허용
- `h2`: 상호작용 대상의 지도 경계 검사 누락

Easy 과제는 월드 생성기를 사용하지 않는 16×16 fixture, 목표 전용 보상,
목표·사망·128스텝 종료 조건을 사용합니다. R-M은 같은 시작 재고에 다이아몬드
목표와 H=512를 둡니다. 정상 경로는 fixed/mutant 모두에서, 짧은 활용 경로는
mutant에서만 성공하도록 회귀 검사합니다. 원본 업적·체력 보상은 학습 보상에서
제외하고 `info["HackRL/original_reward"]`에 기록합니다.

### PPO 파일럿

PPO의 네트워크·GAE·손실·optimizer는
`Craftax_Baselines@7ce36fa`에서 이식했습니다. 정확한 출처와 MIT 라이선스,
HackRL 변경 범위는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에
기록되어 있습니다.

공통 벡터 래퍼는 worker별 독립 reset을 사용하고, 전이의
`terminal_observation`과 정책이 다음에 받는 reset 관측을 분리합니다.
정책 입력은 1,345차원 관측뿐이며 목표 성공·위반·반복 수확은 평가 정보로만
집계합니다. 정상 규칙 기반 action mask는 사용하지 않습니다.

아래 명령은 Easy 3개 × fixed/mutant의 학습·평가 배선 확인용 짧은
파일럿입니다. 작은 고정 fixture와 매우 적은 업데이트를 사용하므로 성능
비교 결과로 해석하면 안 됩니다.

```bash
JAX_PLATFORMS=cpu python scripts/run_ppo_pilot.py \
  --all-pairs --num-envs 2 --num-steps 4 --num-updates 1 \
  --update-epochs 1 --num-minibatches 1 --layer-size 16 \
  --eval-episodes 2
```

### `offrl` 환경에서 실행

```bash
conda activate offrl
python -m pip install -e '.[dev]'
JAX_PLATFORMS=cpu python -m pytest -q
```

환경 생성 예시:

```python
from hackrl import EasyTask, HackRLEasySymbolicEnvNoAutoReset

fixed_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=False)
mutant_env = HackRLEasySymbolicEnvNoAutoReset(EasyTask.R_E, mutant=True)
```

변형 ID와 위반 플래그는 정책 관측에 추가하지 않습니다. 판정 결과는
`info["HackRL/violation"]`에만 기록됩니다.
