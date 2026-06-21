# 1. 결정 사항

**기하학적 규칙(rule-based) 기반의 Ground Contact 분류기 개발을 여기서 종결하고, Layer 5 (LLM Object Refiner) 개발로 전환한다.**

---

# 2. 종결 배경

## 2.1 시도한 접근

`mask_area`(객체 마스크 면적)를 기준으로 "손에 들려 있음(held)"과 "바닥에 놓임(placed)"을 구분하는 절대 임계값(threshold)을 찾으려 했음

- 1차 가설: `mask_area ≥ 15,000px` → 공중 부양(held_or_elevated)
- 검증 결과: log14~15(holdout 세션)에서 즉시 반증됨
    - 손에 든 스마트폰의 `mask_area` 중앙값이 10,838px로 임계값보다 낮게 나타남
    - 원인은 거리(depth)에 따른 원근 왜곡 — 카메라에서 멀어질수록 같은 "손에 든 상태"라도 마스크 면적이 거리의 제곱에 반비례해 줄어듦

## 2.2 보정 시도

거리 왜곡을 보정하기 위해 `target_z²`를 곱한 정규화 지표 `Mask_norm = mask_area × target_z²`를 제안하고 3-state 분류 기준(`placed_on_floor` / `held_or_elevated` / `unknown`)을 새로 수립함

## 2.3 한계 인정

- 검증 표본이 세션 단위로 2개(77개, 124개 프레임)에 불과해 일반화 근거가 약함
- 새로운 임계값 역시 카메라 각도, 조명, 사물 종류가 바뀌면 다시 깨질 가능성이 높음 → "수렴하는 문제"가 아니라 "조건이 늘어날수록 발산하는 문제"
- YOLO 자체의 오탐지(스마트폰을 `bottle`로 인식, 배경 사물을 `suitcase`로 고정 인식 등)가 있을 경우 기하 레이어는 입력 샘플 수가 0이 되어 연산이 마비됨
    - 즉, 사물의 의미(Semantic Identity)가 먼저 정해지지 않으면 기하 정보만으로는 원근 왜곡을 풀 수 없음

---

# 3. 전환 이유

## 3.1 연구 설계와의 정합성

연구 개요에서 정의한 핵심 기여도는 "기하학적 규칙을 완벽하게 다듬는 것"이 아니라, **비전 인지 + LLM 판단 + 동작 생성을 하나의 Embodied AI 파이프라인으로 통합하는 것**임. 기하 레이어는 보조 전처리 단계이지 연구의 최종 목적지가 아님.

## 3.2 일정상의 이유

전체 일정(50일)을 고려할 때, 본 단계는 이미 충분한 시간을 투입해 "규칙 기반 단독 해결은 불가능하다"는 결론을 확보함. 추가 투입 대비 학술적/기술적 이득이 낮음.

## 3.3 인과관계 역전

당초 파이프라인 가정은 "기하 레이어 검증 → LLM 의미 보정" 순서였으나, 실험 결과 이 순서 자체가 반박됨. 의미 정보(LLM)가 기하 해석에 선행 또는 병렬로 필요하다는 것이 데이터로 확인됨.

이는 단순한 구현 순서 변경이 아니라, 시스템 아키텍처 수준의 가정 수정에 해당한다.

초기 설계에서는 Geometry Layer가 객체 상태를 충분히 결정할 수 있다고 가정하고, LLM은 그 결과를 후처리하는 역할로 정의하였다.

그러나 실험 결과, 동일한 기하 정보(mask_area, centroid_y, target_z)를 가지더라도 객체의 실제 의미(스마트폰, 병, 리모컨 등)에 따라 상태 해석이 달라질 수 있음이 확인되었다.

따라서 Geometry → Semantics의 단방향 파이프라인이 아니라, Geometry와 Semantics가 상호 보완적으로 상태를 추론하는 구조로 연구 가정을 수정한다.

---

# 4. 다음 단계 (Layer 5: LLM Object Refiner)

## 4.1 목표

기하 레이어가 출력한 노이즈 섞인 수치 데이터를 LLM에게 전달하여, 사물의 실제 의미적 상태를 자연어/구조화된 판단으로 보정

## 4.2 입력 예시

```json
{
  "class": "bottle",
  "mask_area": 10838,
  "centroid_y": 404,
  "target_z": 4.86
}
```

정확히는

```python
{
  "class": "bottle",
  "confidence": 0.41,
  "mask_area": 10838,
  "centroid_x": 612,
  "centroid_y": 404,
  "target_z": 4.86,
  "mask_norm": 255930.7,
  "context": "1인칭 시점, 사용자가 실내 공간을 촬영하며 이동 중",
  "anchor_info": {
    "frames_observed": 45,
    "centroid_variance": 12.8,
    "depth_variance": 0.21
  }
}
```

Semantic Interpretation Layer는 단순히 객체 클래스명만 입력받지 않는다.

기하 레이어가 산출한 공간 정보(mask_area, centroid, depth), 비전 모델의 신뢰도(confidence), 그리고 시간적 연속성(anchor_info)을 함께 활용하여 객체의 실제 의미와 상태를 추론한다.

특히 confidence 값은 "YOLO가 해당 객체를 얼마나 확신하는가"를 나타내는 중요한 신호이며, 낮은 confidence는 객체 정체성 자체를 재검토해야 함을 의미한다.

## 4.3 기대 출력

- 사물의 실제 정체성 보정 (예: "bottle"로 오탐지된 객체가 실제로는 "스마트폰"일 가능성 판단)
- 상태 보정 (예: "사람이 조작 중인 상태"로 의미론적 라벨링)

Semantic Interpretation Layer의 출력은 단순한 클래스 보정이 아니라, 객체의 의미론적 상태를 계층적으로 표현하는 구조를 목표로 한다.

1. Object Identity
    - 실제 객체 정체성 추론
    - 예: bottle → smartphone
2. Object State
    - 객체의 현재 물리적 상태 추론
    - 예: held_by_user
    - 예: placed_on_floor
    - 예: placed_on_table
3. Interaction State
    - 객체가 현재 어떤 상호작용 맥락에 있는지 추론
    - 예: currently_in_use
    - 예: idle
    - 예: being_observed

예시는

```python
{
  "object_identity": "smartphone",
  "object_state": "held_by_user",
  "interaction_state": "currently_in_use",
  "reasoning": [
    "객체가 화면 중심부에 지속적으로 위치함",
    "깊이 변화가 작음",
    "1인칭 촬영 상황과 일치함"
  ]
}
```

## 4.4 처리 방침

- 현재의 `Mask_norm` 3-state 분류기는 **폐기하지 않고 "잠정 v0 버전"으로 코드베이스에 유지**
    - LLM이 1차 판단을 내릴 때 참고할 보조 신호(fallback guard)로 활용
- 정식 임계값 확정이 아닌, "동작 가능한 베이스라인" 수준으로 취급

Semantic Interpretation Layer 역시 최종 목표는 객체 의미 추론의 완전 자동화가 아니라, 후속 행동 결정(Action Planning) 단계에 전달 가능한 수준의 의미론적 상태 표현(Semantic State Representation)을 생성하는 것이다.

따라서 초기 버전에서는 추론 정확도보다 일관된 출력 스키마 확보를 우선 목표로 한다.

---

# 5. 기존 가설에 대한 최종 판정 (요약)

| 가설 | 판정 | 비고 |
| --- | --- | --- |
| mask_area가 단독으로 Ground Contact를 설명한다 | 부분 지지 | 거리 정규화 없이는 임계값 붕괴 |
| centroid_y가 단독으로 설명한다 | 부분 지지 | 세션 간 절대값 변동 큼 |
| mask_area + centroid_y 조합이 우수하다 | 지지 | 상호 보완적이나 의미 정보 없이는 한계 |
| 기하 레이어 검증 후 LLM 보정으로 순차 진행한다 | 반박 | 의미 정보 선행/병렬 필요로 아키텍처 수정 |

---

# 6. 액션 아이템

1. `Mask_norm` 3-state 분류기 코드를 v0으로 동결하고 별도 모듈로 분리
2. LLM Object Refiner 모듈(`llm/` 디렉토리) 설계 시작 — 프롬프트 설계 및 입출력 스키마 정의
3. 기하 레이어 ↔ LLM Refiner 간 데이터 인터페이스(JSON 스키마) 확정
4. 표본 부족 문제는 추후 정식 평가 단계(성공 기준: 시나리오 30개, 70% 정확도)에서 재검증하기로 보류

Semantic State Schema v1 정의

- Object Identity
- Object State
- Interaction State

3계층 구조를 표준 출력 포맷으로 확정

향후 Action Planner와 Avatar Controller는 해당 스키마만 참조하도록 인터페이스를 고정한다.