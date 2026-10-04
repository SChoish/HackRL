# 렌더 코드 Scientific Review — 2026-10-04

검토 기준: `f6374f04fc5b36490fd7b7b6d16f788648d567f2`의
`scripts/report_figures/` 10개 파일과 관련 환경·평가 전이 코드.

## 판단

실제 before/after 전이를 저장하고 렌더러가 JSON만 읽는 분리는 유지할 가치가 있다.
EDV 그림의 U 및 같은 정책의 커널 간 할인 return 차이 계산도 의도한 비교와
일치한다. 그러나 기존 영상의 지형, 시야 표시 및 선택 사례의 설명에는
연구 해석을 바꿀 수 있는 오류·누락이 있어 수정했다.

## 확인된 문제와 수정

| 중요도·상태 | 위치 | 확인된 근거와 결과 | 수정 유형·조치 |
|---|---|---|---|
| 높음·확인됨 | `render_pack_pixel._draw_map` | 7–11 사각형 전체를 바닥으로 그림. 실제 validation 지형은 추가 장애물을 포함하며 layout 변환도 적용함 | 코드 수정: trace `walkable`과 전체 경로에서 지형·화면 범위 계산 |
| 높음·확인됨 | `render_tick_craft_pixel._draw_base_map` | CRAFT 화면 crop 테두리를 벽으로 그림. 실제 커널은 세 시설 외에 16×16 범위가 열려 있음 | 코드 수정: 실제 열린 지형을 기록·사용. crop을 물리적 경계로 표시하지 않음 |
| 높음·확인됨 | TICK/CRAFT `_draw_*_map` | 지도 바깥 사각형 전체를 `policy view`로 표시하며 플레이어 위치와 무관함 | rewrite·코드 수정: 실제 observation 배열과 위치로 FOV 교집합 계산. 전체 화면은 관전자 화면이라고 명시 |
| 중간·확인됨 | `record_tick_trace.main`, `record_craft_trace.main` | 성공·활용한 후보 중 최대 길이 이득을 선택하지만 trace/영상에 선택 규칙이 없었음 | evidence gap 보완: 후보 수·적격 수·선택 규칙·이득·동률 규칙을 JSON에 저장하고 최대 이득 선택임을 자막에 표시 |
| 중간·확인됨 | 픽셀 렌더의 header | seed·update·positive seed 수를 상수로 표시. CRAFT mode 영상에 sample 모집단의 2/5를 함께 표기 | rewrite: 영상 메타데이터에서 읽고 모집단 상수를 제거. 단일 mode 사례임을 표시 |
| 중간·확인됨 | recorder provenance | 체크포인트 경로만으로는 동일 이름 파일 교체를 구분할 수 없음 | evidence gap 보완: config/state SHA256, 실행 코드 SHA·dirty 여부 기록 |
| 중간·확인됨 | `_state_at`, `_key_steps`, held 상태 | 빈 trace는 마지막 항목 접근으로 실패. 활용 없는 trace는 필수 `next()`로 실패. 모든 held 상태를 success로 설명 | 코드 수정: 빈·음성 trace 처리, 존재하는 사건만 출력, 실제 성공 여부로 held 문구 분리 |
| 중간·확인됨 | `plots._panel` | signed U 축 하한 −0.05 고정은 그보다 작은 음수 차이를 가림 | 코드 수정: 실제 값에 맞춘 축, 0 기준선. 의도적으로 U<0인 자료로 회귀 검사 |
| 표현 개선 | `plots` | E가 어느 정책인지 불명확하고 전체 학습 개입과 적응 중 개입의 구분이 그림에서 약함 | rewrite: E는 mutant 적응 정책의 fixed 평가라고 명시. 두 개입의 범위와 RNG 차이를 캡션에 표시 |
| 표현·재현성 개선 | 전체 렌더 스크립트 | 호스트 절대 경로, 그래픽 의존성 누락, 지도 하단과 자막 중첩, legacy still의 첫 행 일부 잘림 | 코드 수정: checkout 상대 경로/데이터 root, render extra·CLI 안내, 화면 영역 및 축 범위 수정 |

최대 이득 사례를 시연에 사용하는 것 자체를 금지한 것은 아니다.
그 사례를 평균 효과나 무작위 대표 사례로 설명하지 않도록 수정했다.
이미 명시돼 있던 mode/sample 분리, 최초 관측 checkpoint와 정확한 발견 시점의
구분, 정책별 커널 비교 계산은 유지했다.

## 검증과 한계

- CPU 렌더 회귀 검사 **11 passed**.
- PACK 검증 장애물, CRAFT 열린 바닥, 관측 범위 이동, 빈·실패·무활용 trace,
  생성량 배지, 메타데이터와 EDV 비교 대상을 확인했다.
- 세 환경의 실제 커널에 검증된 행동열을 입력하고 before/after JSON 및
  presentation/analysis PNG 생성을 확인했다. 이는 **scripted 검증**이다.
- 세 환경의 GIF/MP4 파일 출력도 확인했다. GIF 500ms/frame, MP4 2fps 및 프레임 수를 검사했다.
- `compileall`과 `git diff --check` 통과.
- 서버의 학습 체크포인트와 원시 curve JSON은 이 checkout에 없다.
  따라서 실제 저장 정책의 영상, 전체 결과 그림 수치 및 BC pairing 해시를
  독립적으로 다시 계산했다는 결론은 내리지 않는다.
- 학습 커널·보상·정책 업데이트 코드는 수정하지 않았다. 새 학습도 실행하지 않았다.
- 고정된 7×7 viewport 밖으로 넓게 이동하는 trace는 잘라 보여주지 않고 명시적으로
  거부한다. 일반적인 대형 맵 재생기를 완성했다는 의미는 아니다.

실행 방법과 그림별 지표 정의는 [rendering.md](../rendering.md)에 정리했다.
