# 7-3. 의미-기하 융합 구조 실험 — 개정판 보고서 (v3)

> **문서 성격**: 초판(7-3)의 disparity 데이터(3,674 프레임)를 재분석하여 실패 원인을 규명하고, **Metric Depth 모델로 재촬영(3,923 프레임)하여 원근 보정 가설을 검증**하였다. 최종적으로 목적이 다른 **두 특징을 역할 분리**한 아키텍처를 정립한다.
>
> - **방법 ①(크기·거리 불변)**: `metric mask_norm` — 원근 보정 가설을 검증 (실험 F에서 성공)
> - **방법 ②(행동 분류)**: `raw floor_margin` — held_in_hand / elevated 분리 (실험 C에서 우월성 확인)

---

## 0. 변경 이력

### 0.1 초판(7-3) 대비

| 항목 | 초판 | 개정판(v3) | 근거 |
| --- | --- | --- | --- |
| 거리 의존 원인 | "상대적 깊이 한계"(정성) | `target_z`=disparity(∝1/Z) → $Mask_{norm}\propto 1/Z^4$ (수식적 필연) | 실험 A |
| Scale&Shift 효과 | 미검증 | 글로벌 보정은 **일반화 실패**(교차검증), Metric 모델은 **검증 성공** | 실험 B·E·F |
| 원근 보정 가설 | 실패로 종결 | **Metric 깊이로 검증 성공** (CV 1.4→0.11) | 실험 F ★ |
| 상태 분류 특징 | $Mask_{norm}$ 단일(실패) | **`raw floor_margin`** (거리별 89~100%) | 실험 C |
| 최종 구조 | 단일 특징+단일 threshold | **2방법 역할 분리 + 거리구간 게이팅** | §5 |

### 0.2 데이터셋 (본 보고서는 2개 데이터셋 사용)

| | Dataset-1 (disparity) | Dataset-2 (metric) |
| --- | --- | --- |
| 파일 | `data/mask_norm_log.csv` | `data/mask_norm_metric_log_clean.csv` |
| 깊이 모델 | DA-V2-Base (relative) | **DA-V2-Metric-Indoor-Base** |
| 프레임 수 | 3,674 | 3,923 → 정제 3,099 |
| 거리 라벨 | 500·1000·1500·2000 mm | 1.0·1.5·2.0·2.5 m |
| 객체 | bottle·keyboard·cell phone·mouse | book·bottle |
| 사용 실험 | A·B·C·D·E | **F** |

---

## 1. 실험 목적

1. **$Mask_{norm}$ 거리 의존의 수식적 규명** — 오차가 아닌 필연임을 증명.
2. **원근 보정 가설 검증** — "거리에 따른 원근 왜곡을 수식적으로 보정 → $Mask_{norm}$ 거리 불변"이 성립하는지, Metric 깊이로 검증.
3. **행동 분류 특징 선정** — held_in_hand / elevated를 실제로 분리하는 특징 도출.
4. **2방법 아키텍처 정립** — 크기 추정과 행동 분류를 분리한 파이프라인 설계.

---

## 2. 실험 환경

### 2.1 파이프라인
```
Camera → YOLOv8n(conf=0.10) → SAM → Depth 모델 → 특징 계산(mask_norm, floor_margin)
```

| 모델 | 역할 | 비고 |
| --- | --- | --- |
| YOLOv8n | 탐지(bbox) | COCO pretrained |
| SAM | 세그멘테이션 | bbox 프롬프트 → mask_area |
| Depth | 깊이 | Dataset-1: relative / Dataset-2: **metric** |

### 2.2 카메라 캘리브레이션 (재캘리브레이션)
| | 값 |
| --- | --- |
| RMS 오차 | **0.2619 px** (초판 0.8에서 개선) |
| $f_x, f_y$ | 964.86, 964.45 |
| $c_x, c_y$ | 636.84, 359.35 (1280×720 중앙에 근접) |

### 2.3 Semantic Prior ($A_{real}$)
| 객체 | 실측 | $A_{real}$ (mm²) |
| --- | --- | --- |
| cell phone | 163.6×78.1 | 12,773.16 |
| bottle | 215×65 | 13,975.0 |
| mouse | 113×61 | 6,893.0 |
| keyboard | 430×150 | 64,500.0 |
| **book** | 274×215 | 58,910.0 |

---

## 3. 실험 수식

### 3.1 크기 특징 — $Mask_{norm}$
$$Mask_{norm} = \frac{mask\_area \times target\_z^{\,2}}{A_{real}}$$

**거리 의존성 유도**: 투영 모델에서 $mask\_area \approx f^2 A_{phys}/Z^2\ (\propto 1/Z^2)$.
- 설계 의도: $target\_z \propto Z$ → $mask\_area\cdot target\_z^2 \propto \text{const}$ (거리 불변)
- disparity 사용 시: $target\_z \propto 1/Z$ → $Mask_{norm} \propto 1/Z^4$ (**거리 보상이 증폭**)
- **Metric 사용 시**: $target\_z \propto Z$ → $Mask_{norm} \to f^2 A_{phys}/A_{real} = \text{const}$ (가설 성립)

### 3.2 분류 특징 — `floor_margin`
$$floor\_margin = floor\_depth - target\_z$$
바닥 깊이와 물체 깊이의 차이(상대 깊이). 물리적으로 "물체가 바닥에서 얼마나 떠 있나"를 반영 → held(손 높이) vs elevated(표면 높이) 분리 신호.

---

# PART Ⅰ. 방법① — 크기·거리 불변 (원근 보정 가설)

## 4. 실험 A — `target_z`가 disparity임을 확인 (Dataset-1)

거리별 평균 `target_z`:

| 거리 | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| mean `target_z` | 7.401 | 5.049 | 2.909 | 1.933 |

거리가 멀수록 **단조 감소** → disparity(∝1/Z) 확정. §3.1의 $1/Z^4$ 거동과 일치 (bottle held 500→2000mm: 29.28→0.46, 약 64배 감소).

## 5. 실험 B·E — Scale&Shift 글로벌 보정의 한계 (Dataset-1)

### 5.1 글로벌 피팅 (실험 B)
$target\_z \approx s\cdot(1/Z)+t$ 피팅: **$s=3422.2,\ t=0.7501$, corr=0.754**.
복원 $Z_{rec}=s/(target\_z-t)$로 재계산 시 in-sample CV는 1.4→0.5로 개선되나, **이는 낙관적(피팅에 전 거리 사용)**.

### 5.2 Leave-One-Distance-Out 교차검증 (실험 E)
3개 거리로 $s,t$ 피팅 → 남긴 1개 거리에 적용(정직한 out-of-sample):

| | raw | global(in-sample) | **LODO(정직)** | ideal(oracle) |
| --- | --- | --- | --- | --- |
| 전체 통합 CV | 1.41 | 0.53 | **1.39** | 0.13 |

- **$s,t$ 불안정**: held-out에 따라 $s$ 37% 변동, $t$ 부호까지 반전(-1.24~+1.15) → 단일 글로벌 보정 부재.
- **LODO CV(1.39) ≈ raw(1.41)**: 못 본 거리엔 글로벌 보정이 **무보정과 동급**. 원거리에선 역수 $s/(target\_z-t)$가 노이즈를 폭발.

> **결론**: Dataset-1(relative)을 후처리해선 가설을 충족 못 함. **Metric 깊이 재수집이 유일한 길** → 실험 F. (이 음성 결과가 재촬영의 정당화 근거)

## 6. 실험 F — Metric 재촬영으로 원근 보정 가설 검증 ★ (Dataset-2)

### 6.1 방법
- 깊이 모델: **Depth-Anything-V2-Metric-Indoor-Base-hf** (미터 깊이 직접 출력)
- 객체: book(58,910mm²), bottle / 상태: elevated, held_in_hand / 거리: 1.0·1.5·2.0·2.5m
- 원본 프레임 저장(재추론 대비), 새 캘리브레이션 적용
- **총 3,923 프레임** 수집

### 6.2 이상값 정제
정지 물체는 깊이가 일정해야 하므로, 그룹 중앙값에서 **깊이 ±35% 벗어난 물리적 스파이크**(z가 4.x로 튀는 프레임)를 제거.
- 3,923 → **3,099행** (21% 제외). 원본 CSV는 보존.
- 제거는 주로 **투명 PET 병**(깊이가 투과되어 스파이크)과 일부 book 세션에 집중.

### 6.3 결과 — 거리 불변성 (CV, 낮을수록 불변)

| object·state | 옛 disparity | **metric(정제)** | 판정 |
| --- | --- | --- | --- |
| **bottle · elevated** | ~1.4 | **0.11** | ✅ **ideal(0.13) 도달** |
| book · elevated | ~1.4 | 0.29 | ✅ 검증 |
| book · held_in_hand | ~1.4 | 0.24 | ✅ 검증 |
| bottle · held_in_hand | ~1.4 | 0.33 | △ 부분 |

**거리 추종(z 중앙값)과 mask_norm 평탄성:**

| object·state | z(1.0/1.5/2.0/2.5) | mask_norm(1.0/1.5/2.0/2.5) |
| --- | --- | --- |
| bottle · elevated | 1.16 / 1.66 / 2.06 / 2.53 | 1.09 / 0.97 / 0.89 / 0.81 |
| book · elevated | 1.05 / 1.48 / 2.24 / 2.77 | 1.02 / 0.30 / 1.19 / 1.18 |

- **z가 거리를 정확히 추종** → Metric 깊이 정상. §3.1 예측대로 mask_norm이 거의 평탄.
- **bottle·elevated는 CV 0.11로 ideal 수준** — 원근 보정 가설이 완전히 성립.

### 6.4 해석
- **가설 검증 성공**: CV가 disparity의 ~1.4에서 metric의 **0.11~0.33으로 3~13배 개선**. 초판의 실패는 수식이 아니라 **disparity 센서** 탓이었음이 증명됨.
- **elevated > held**: 정지(elevated)가 손 흔들림이 없어 훨씬 깨끗. **bottle·held(0.33)는 투명 PET+손**으로 z 스케일이 왜곡(2.24~3.38)되어 신뢰 낮음. → 검증엔 **elevated가 적합**.
- **잔차의 출처가 바뀜**: 남은 CV는 이제 깊이가 아니라 **mask_area(세그멘테이션·자세)** 노이즈. 예) book 1.5m elevated는 z는 정확(1.48)하나 mask_norm만 낮음(0.30) — 마스크가 작게 잡힌 단일 이상 세션.

---

# PART Ⅱ. 방법② — 행동 분류

## 7. 실험 C — 분류 특징 비교 (Dataset-1)

held vs elevated 분리력을 Fisher 판별비 + 단일 threshold 최적 정확도로 평가.

### 7.1 거리 전체 통합
| 특징 | bottle Acc | keyboard Acc |
| --- | --- | --- |
| mask_norm(raw) | 65.5% | 62.3% |
| target_z | 69.4% | 70.3% |
| **floor_margin** | **96.6%** | 72.4% |

### 7.2 거리별 (거리 앎 가정, 각 거리 내 최적 threshold)
| 객체 | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| **floor_margin** bottle | 100% | 100% | 100% | 100% |
| mask_norm bottle | 100% | 100% | 84% | 97.8% |
| **floor_margin** keyboard | 99.3% | 89.3% | 100% | 99.3% |
| mask_norm keyboard | 97.0% | 78.3% | 99.7% | 98.7% |

→ **모든 거리에서 `floor_margin`이 `mask_norm`을 능가.** 상태를 나누는 신호는 "물체 크기"가 아니라 **"바닥 대비 높이"**.

### 7.3 단, floor_margin도 거리 의존적
disparity 공간 값이라 거리에 따라 값이 이동(예 bottle elevated: -6.49→1.88). **단일 글로벌 threshold 불가 → 거리 구간별 threshold 필요.**

## 8. 실험 D — 분류 특징의 metric 변환은 역효과 (Dataset-1)

"방법①처럼 분류 특징도 metric 변환하면 더 좋아지나?" 검증:

| 객체 | raw floor_margin | metric 변환 |
| --- | --- | --- |
| bottle | 96.6% | 75.4% ↓ |
| bottle @2000 | 100% | 62.3% ↓ |

역변환 $Z=s/(target\_z-t)$의 원거리 노이즈 증폭 때문에 **악화**. → **분류 브랜치는 raw `floor_margin`을 사용**.

---

# PART Ⅲ. 통합 — 2방법 아키텍처

## 9. 최종 구조

두 방법은 목적이 다르고, 좋은 깊이에 대한 반응도 반대다:

| | 방법① 크기·거리 불변 | 방법② 행동 분류 |
| --- | --- | --- |
| 특징 | **metric** mask_norm | **raw** floor_margin |
| 깊이가 정확해지면 | 개선(거리 불변 달성) | (raw가 최적, metric 변환은 역효과) |
| 검증 | 실험 F (CV 0.11) | 실험 C (89~100%) |
| 거리 의존 | metric으로 제거 | 남음 → 구간별 threshold |

```
                   ┌─ 방법① 거리·크기 (metric mask_norm) ─────────┐
raw depth ─────────┤   거리 구간 판정 + 거리 불변 크기            │──┐
(target_z,         └────────────────────────────────────────────┘  │ 거리구간
 floor_margin,     ┌─ 방법② 행동 분류 (raw floor_margin) ─────────┐  │ 선택
 mask_area)        │   구간별 threshold로 held/elevated 판정     │◀─┘
                   └────────────────────────────────────────────┘
```
- **방법①**이 "몇 m 구간인가"를 판정 → **방법②**가 그 구간의 threshold로 상태 분류. 두 방법은 독립적이나 A가 B를 게이팅.

## 10. 종합 결론

| 질문 | 결론 |
| --- | --- |
| $Mask_{norm}$ 거리 의존 원인 | disparity → $1/Z^4$ (수식적 필연) |
| 기존 데이터 후처리로 보정 가능? | ❌ (LODO CV 1.39 ≈ raw) → 재촬영 필요 |
| Metric으로 원근 보정 가설 검증? | ✅ **성공** (bottle·elevated CV 0.11 ≈ ideal) |
| 행동 분류 주 특징? | **raw floor_margin** (거리별 89~100%) |
| 분류 특징 metric 변환? | ❌ 역효과 → raw 사용 |
| 최종 구조 | **2방법 역할 분리 + 거리구간 게이팅** |

---

## 11. 한계 및 향후 방향

### 11.1 한계
1. **투명·held 조건 신뢰 낮음**: bottle(투명 PET)은 깊이 투과로 스파이크 다발(21% 제거의 주원인), held는 손 흔들림으로 노이즈. **검증은 elevated·불투명 물체 위주가 적합**.
2. **book 1.5m elevated 단일 이상 세션**: 깊이는 정확하나 마스크가 작게 잡혀 mask_norm 저하. 해당 세션 재촬영 시 book CV도 개선 예상.
3. **분류 검증은 Dataset-1(disparity) 기준**: 방법②(floor_margin)는 아직 metric 데이터에서 재확인 전. metric에선 floor_margin 부호가 반전되며, 상대 높이가 실제 물리량이 되어 **분류가 더 개선될 가능성**이 있음(향후 검증).
4. **단일 카메라**: MacBook 내장 카메라 한정.

### 11.2 향후 방향
1. **방법② metric 재검증**: metric 데이터로 floor_margin 분류 정확도 재측정(부호 반전 반영, 거리 구간별 threshold 재수집).
2. **book 1.5m elevated 재촬영**으로 방법① 데이터 보강.
3. **거리 구간별 threshold 테이블** 구성 후 코드(`vision/reasoning/affordance_engine.py`) 반영.
4. **독립 검증셋**(통제 조건 밖)으로 두 방법의 일반화 확인.
5. **elevated·불투명 물체 추가 수집**으로 방법① 일반화 강화.

---

## 부록. 재현 정보
- Dataset-1: `data/mask_norm_log.csv` (3,674) — 실험 A·B·C·D·E
- Dataset-2: `data/mask_norm_metric_log_clean.csv` (3,099, 원본 3,923에서 깊이 ±35% 스파이크 제거) — 실험 F
- Metric 모델: `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf`
- 캘리브레이션: RMS 0.2619, $f_x$=964.86, $f_y$=964.45, $c_x$=636.84, $c_y$=359.35
- 글로벌 disparity 피팅: `target_z ≈ 3422.2·(1/Z) + 0.7501`, corr=0.754
- 수집 스크립트: `tests/test_mask_norm_metric.py`
