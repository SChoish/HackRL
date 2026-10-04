# 환경 영상과 결과 그림

TICK-CLAIM, PACK-RESTORE, CRAFT-REMAIN은 실제 게임에서 보고된 증상을 참고해
작성한 합성 작업장이다. Minecraft/Factorio의 원본 코드나 패치를 실행하는
재현이 아니다. 초기 Craftax 변형 트랙과 구분한다.

## 실행

저장소 루트에서 설치한다. 학습 체크포인트와 평가 원자료는 Git에 포함되지 않는다.

```bash
python -m pip install -e '.[render,dev]'
export PYTHONPATH="$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

기본 데이터 위치는 현재 checkout의 `runs/`다. 다른 위치의 기존 산출물을
사용하려면 `HACKRL_ROOT=/path/to/HackRL`을 지정한다. 이 경로 아래에
`runs/`가 있어야 한다. 기록 코드 SHA는 **데이터 위치가 아닌 실행 코드의
checkout**을 식별해야 하므로 아래 provenance 구현은 실행 코드 위치를 사용한다.

학습을 다시 하지 않고 저장된 Dual 체크포인트에서 mode 행동을 기록한다.
기본 예시는 PACK/TICK seed 20, CRAFT seed 21의 adapt 4096이다.

```bash
python -m report_figures.record_pack_trace
python -m report_figures.record_tick_trace
python -m report_figures.record_craft_trace
```

이 명령은 학습을 시작하지 않지만 CPU에서 체크포인트 복원과 평가를 수행한다.
JSON에는 실제 before/after 상태, 관측, 행동, 이벤트가 기록된다. 새 trace에는
config/state SHA256, 기록 코드 SHA와 dirty 여부, mode, 초기 상태와 사례 선택
규칙이 포함된다. 파일 경로만 같은 다른 체크포인트와 구분할 수 있다.

저장한 trace만으로 렌더링한다. 렌더러는 JAX나 환경 커널을 호출하지 않는다.

```bash
python -m report_figures.render_pack_pixel
python -m report_figures.render_tick_craft_pixel
```

다른 trace의 출력 경로를 분리하는 예:

```bash
python -m report_figures.render_pack_pixel \
  --trace /path/to/trace.json --output /path/to/rendered
python -m report_figures.render_tick_craft_pixel \
  --environment craft --trace /path/to/trace.json --output /path/to/rendered
```

출력은 presentation/analysis PNG, GIF, MP4와 핵심 이벤트 정지 화면이다.
기존 영상은 자동으로 고쳐지지 않는다. provenance/selection이 없는 기존
trace는 해당 필드를 `unrecorded`/`not recorded`로 표시하므로, 최종 발표용은
새 recorder로 trace부터 다시 만드는 것을 권한다.

## 화면과 사례를 읽는 방법

- 픽셀 지도와 HUD는 **관전자 화면**이다. 정책의 전체 입력을 재현한 화면이 아니다.
- 노란 선은 저장된 관측 배열의 크기와 플레이어 위치로 계산한 시야와 화면 crop의
  교집합이다. 화면 경계는 게임의 벽이 아니다. CRAFT의 주변은 열린 바닥이다.
- PACK/TICK은 trace의 실제 `walkable`을 사용한다. CRAFT도 새 trace에는
  통과 가능 지형을 기록하며, 구형 trace에는 기존 v1 커널의 열린 지형을 적용한다.
- 고정된 화면은 전체 기록 경로를 포함한다. 7×7 tile보다 넓은 경로는 조용히
  자르지 않고 오류로 알린다. 그런 trace에는 더 큰 viewport가 필요하다.
- 같은 정책을 두 커널에서 각각 실행하므로 상태가 갈라진 뒤 행동도 달라질 수 있다.
  `storyboards.py`의 검증된 동일 행동열 비교와 구분한다.
- TICK은 32개 검증 시작 상태, CRAFT는 두 초기 phase 중 **양쪽 성공 및 mutant
  활용 조건을 만족하면서 길이 이득이 가장 큰 사례**를 고른다. 동률은 낮은 layout,
  phase 순이다. 예시의 길이 차이를 평균 효과로 보고하면 안 된다.
- PACK은 validation layout 0, loaded의 지정 사례다. 이것도 모집단 평균이 아니다.
- 영상은 mode다. CRAFT의 기존 sample 결과 2/5를 영상의 측정치처럼 표시하지 않는다.
- 위반 및 초과 납품은 분석 지표다. PACK/TICK의 초과 납품은 평가와 같은
  위반 잔액 집계이며, 개별 곡물의 물리적 출처를 추적했다는 뜻은 아니다.
- 완료한 쪽은 추가 전이 없이 정지한다. 목표 미달 종료는 success로 표시하지 않는다.
- 수치 없는 음성 사례와 0-transition trace도 렌더링할 수 있다. 없는 이벤트의
  핵심 장면을 만들거나 활용 사례를 요구하지 않는다. 기본 recorder의 TICK/CRAFT
  선택기는 여전히 활용 예시 선택용이므로, 적격 사례가 없으면 명시적으로 실패한다.

## 결과 그림

```bash
python -m report_figures.plots
```

`runs/figures_report_v1/results/`에 저장한다. 실제로 읽은 평가 JSON의 SHA256은
`sources.json`에 기록한다. 누락 파일을 0으로 대체하지 않는다.

- E: **mutant 적응 정책을 fixed 커널에서 평가한 성공률**. fixed 계속 학습 정책의
  성공률이나 사전학습 능력 곡선과 같지 않다.
- D: EDV 그림은 mutant 커널에서 mutant 적응 정책과 fixed 계속 학습 정책의
  초과 납품률 차이 U다. 학습 곡선 및 원인 분리 그림은 축 이름대로 mutant 적응
  정책의 초과 납품률 자체다. 대조군이 0인지와 관계없이 두 정의를 구분한다.
- V: 같은 mutant 적응 정책의 mutant minus fixed 할인 return 차이.
- TICK/PACK EDV는 mode, CRAFT EDV는 sample이다. 합산하지 않는다.
- 점은 학습 seed다. 5-seed bootstrap 구간은 표본 내 불확실성 요약이며,
  에피소드를 독립 학습 반복으로 간주하거나 모집단 변동이 없음을 입증하지 않는다.
- 교사/모방 2×2는 전체 이력 개입, BC-on/off는 동일 Dual 사전학습 이후의 개입이다.
  teacher-only와 GC-PPO 사이에는 RNG 경로 차이가 있다.

## 검증

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m pytest tests/test_report_rendering.py -q
```

회귀 검사는 실제 커널 지형, 관측 범위, 같은 행동열의 실제 전이, 음성/빈 trace,
EDV 비교 대상과 음수 차이 표시를 확인한다. 검증용 scripted trace는 학습된
정책의 성공 사례나 새로운 실험 결과가 아니다.
