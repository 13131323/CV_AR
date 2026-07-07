# 7-3. 의미-기하 융합 구조 실험 — 최종 보고서

> **핵심 성과**: 초판에서 **실패**로 종결됐던 "거리에 따른 원근 왜곡을 수식적으로 보정 → $Mask_{norm}$ 거리 불변" 가설을, **Metric Depth 모델 재촬영으로 검증 성공**하였다(두 독립 물체에서 CV 1.4 → 0.055·0.112, ideal 수준). 나아가 목적이 다른 두 특징을 **역할 분리**한 아키텍처를 정립한다.
>
> - **방법 ①(크기·거리 불변)**: `metric mask_norm` — 원근 보정 가설 (실험 F에서 검증)
> - **방법 ②(행동 분류)**: `raw floor_margin` — held / elevated 분리 (실험 C)

---

## 0. 데이터셋

| | Dataset-1 (disparity) | Dataset-2 (metric) |
| --- | --- | --- |
| 파일 | `data/mask_norm_log.csv` | `data/mask_norm_metric_log_clean.csv` |
| 깊이 모델 | DA-V2-Base (relative) | **DA-V2-Metric-Indoor-Base** |
| 프레임 | 3,674 | 3,562 → 정제 2,900 |
| 거리 | 500·1000·1500·2000 mm | 1.0·1.5·2.0·2.5 m |
| 객체 | bottle·keyboard·cell phone·mouse | book·bottle |
| 사용 실험 | A·B·C·D·E | F·G |

---

## 용어 · 약자 정리

### (a) 모델 · 파이프라인
| 약자 | 정식 명칭 | 설명 |
| --- | --- | --- |
| YOLO(v8n) | You Only Look Once v8 nano | 객체 탐지 모델. bbox 출력. n=최소 크기 |
| SAM | Segment Anything Model | 세그멘테이션. bbox 프롬프트로 마스크 생성 |
| DA-V2 | Depth Anything V2 | 단안 깊이 추정 모델 |
| bbox | bounding box | 물체를 감싸는 사각 경계 |
| ROI | Region of Interest | 관심 영역(여기선 바닥 검출용 하단 슬릿) |
| conf | confidence | YOLO 탐지 신뢰도 임계값(=0.10) |

### (b) 기하 · 특징 기호
| 기호 | 읽기 | 의미 · 단위 |
| --- | --- | --- |
| $mask\_area$ | 마스크 면적 | SAM 마스크 픽셀 수 [px²] |
| $target\_z$ | 타깃 z | 물체 영역 깊이(마스크 평균). Dataset-1=disparity, 2=metric[m] |
| $Mask_{norm}$ | 마스크놈 | 정규화 크기 특징 $=mask\_area\cdot target\_z^2/A_{real}$ [식1] |
| $floor\_depth$ | 바닥 깊이 | 하단 슬릿 ROI의 깊이 퍼센타일 [식5] |
| $floor\_margin$ | 바닥 마진 | $=floor\_depth-target\_z$, 상대 깊이(분류 특징) [식6] |
| $A_{real}$ | 실제 면적 | Semantic Prior에 저장한 실측 물리 면적 [mm²] (고정 상수) |
| $A_{phys}$ | 물리 면적 | 그 프레임에 실제 보이는 물체 면적. 정면·정확 시 $=A_{real}$ |
| $Z$ | 거리 | 카메라–물체 실제 거리(GT, 줄자 실측) |
| $Z_{rec}$ | 복원 거리 | Scale&Shift로 disparity에서 복원한 metric 거리 [식10] |
| $f$ | 초점거리 | 카메라 초점거리 [px]. $Mask_{norm}$ 수식엔 미포함 |

### (c) 깊이 종류
| 용어 | 의미 |
| --- | --- |
| relative / disparity(역깊이) | 상대 깊이. **가까울수록 값 大**, $\propto 1/Z$. 절대 단위 없음 |
| metric depth(절대 깊이) | 미터 단위 실제 깊이. **멀수록 값 大**, $\propto Z$ |
| affine-invariant | 이미지마다 $s,t$ 미지의 아핀 변환만큼 자유로운 깊이(=disparity 계열) |
| Scale & Shift | disparity를 metric으로 정렬하는 아핀 보정 $target\_z\approx s/Z+t$ [식9] |
| $s,t$ | Scale(기울기)·Shift(절편). 최소자승으로 추정 |

### (d) 통계 · 평가 지표
| 약자 | 정식 | 의미 |
| --- | --- | --- |
| CV | Coefficient of Variation(변동계수) | $\sigma/\mu$. 거리별 평균의 CV=거리 불변성 지표 [식8]. 낮을수록 불변 |
| Fisher($F$) | Fisher discriminant ratio(판별비) | $(\mu_h-\mu_e)^2/(\sigma_h^2+\sigma_e^2)$ [식13]. 높을수록 분리 |
| Acc | balanced accuracy(균형정확도) | held·elevated 각 정답률의 평균 [식14]. 표본 불균형 보정 |
| MAD | Median Absolute Deviation | $\mathrm{median}(|x-\mathrm{median}|)$. 로버스트 산포 [§1.5] |
| P10 / P90 | 10 / 90 퍼센타일 | 바닥 깊이 추출용(disparity=P10, metric=P90) |
| RMS | Root Mean Square error | 캘리브레이션 재투영 오차(=0.2619px) |
| corr | Pearson correlation | 피팅 적합도 [식11] |
| median | 중앙값 | 이상값에 강건한 대표값 |

### (e) 실험 개념
| 용어 | 의미 |
| --- | --- |
| held_in_hand | 물체를 손에 든 상태 |
| elevated | 물체를 탁자/거치대에 세워둔 상태 |
| Semantic Prior | 객체 종류별 실제 크기($A_{real}$) 사전 DB |
| LODO | Leave-One-Distance-Out. 3거리로 피팅→남긴 1거리로 검증(정직한 일반화) |
| ideal / oracle | 실제 거리 라벨을 그대로 넣은 상한(도달 가능한 최선) |
| domain shift | 학습 데이터 분포와 실제 환경의 차이(예: 바닥의 책 미탐지) |
| OOD | Out-Of-Distribution. 학습 분포 밖 입력 |
| spurious correlation / shortcut | 진짜 원인이 아닌 교란변수에 편승한 가짜 신호 |
| foreshortening | 기울임에 의한 겉보기 면적 축소($A_{phys}$ 감소) |
| occlusion | 가림(손 등)에 의한 마스크 결손 |

---

## 1. 계산 방법론 (수식 총정리)

이 절에 모든 계산의 수식을 명시한다. 이후 실험 절은 이 수식들을 적용한 결과다.

### 1.1 크기 특징 $Mask_{norm}$ 과 원근 유도

$$
Mask_{norm} = \frac{mask\_area \times target\_z^{\,2}}{A_{real}} \tag{1}
$$

- $mask\_area$: SAM 마스크의 픽셀 면적 [px²]
- $target\_z$: 물체 영역의 깊이 (Dataset-1은 disparity, Dataset-2는 metric[m])
- $A_{real}$: Semantic Prior 실측 면적 [mm²]

**원근 투영 모델.** 초점거리 $f$, 실제 거리 $Z$, 카메라를 향한 물체 물리면적 $A_{phys}$에 대해 겉보기 픽셀 면적은

$$
mask\_area \;\approx\; \frac{f^2 \, A_{phys}}{Z^2} \quad (\propto 1/Z^2) \tag{2}
$$

(2)를 (1)에 대입:

$$
Mask_{norm} \;\approx\; \frac{f^2 A_{phys}}{Z^2}\cdot\frac{target\_z^{\,2}}{A_{real}} \tag{3}
$$

| $target\_z$ 성질 | 결과 | 의미 |
| --- | --- | --- |
| disparity $\propto 1/Z$ | $Mask_{norm}\propto \dfrac{1}{Z^2}\cdot\dfrac{1}{Z^2}=\dfrac{1}{Z^4}$ | 거리 보상이 **증폭**(실패) |
| metric $\propto Z$ | $Mask_{norm}\to \dfrac{f^2 A_{phys}}{A_{real}}=\text{const}$ | 거리 **소거**(가설 성립) |

> $A_{phys}$는 그 프레임에 실제 보이는 물체 면적으로, 정면·정확 측정 시 $A_{phys}=A_{real}$ 이 되어 완전 상쇄된다. 기울임·가림·세그멘테이션 오차로 $A_{phys}$가 흔들리는 것이 잔차의 원천이다.

### 1.2 깊이 특징 $target\_z$, $floor\_margin$

**물체 깊이** — SAM 마스크 $M$ 영역 깊이맵 $D$의 평균:

$$
target\_z \;=\; \frac{1}{|M|}\sum_{(u,v)\in M} D(u,v) \tag{4}
$$

**바닥 깊이** — 하단 좌·우 슬릿 ROI(탁자 중앙 회피)의 퍼센타일:

$$
floor\_depth \;=\; \mathrm{P}_{k}\big(\{D(u,v): (u,v)\in ROI_{L}\cup ROI_{R}\}\big) \tag{5}
$$

- Dataset-1(disparity): $k=10$ (가장 먼 바닥 = 가장 작은 disparity)
- Dataset-2(metric): $k=90$ (가장 먼 바닥 = 가장 큰 깊이)

**분류 특징:**

$$
floor\_margin \;=\; floor\_depth - target\_z \tag{6}
$$

### 1.3 거리 불변성 지표 — 변동계수 (CV)

각 (객체·상태)에 대해, 거리 $Z\in\{Z_1,\dots,Z_K\}$별 평균

$$
\mu_Z = \frac{1}{N_Z}\sum_{i\in Z} Mask_{norm,i} \tag{7}
$$

를 구한 뒤, **거리별 평균들의 변동계수**로 정의한다:

$$
\boxed{\;CV = \frac{\sigma(\{\mu_{Z_1},\dots,\mu_{Z_K}\})}{\bar\mu}\;},\qquad
\sigma=\sqrt{\frac1K\sum_Z(\mu_Z-\bar\mu)^2},\;\; \bar\mu=\frac1K\sum_Z\mu_Z \tag{8}
$$

$CV\to 0$이면 거리 불변. **$CV$는 $A_{real}$ 값에 불변**이다: (1)에서 $A_{real}$은 상수 제수라 $\mu_Z,\bar\mu$ 모두 같은 비율로 나뉘어 약분된다 → 물체 크기를 대충 알아도 거리 불변성 결론은 동일.

### 1.4 Scale & Shift (disparity → metric 역변환)

affine-invariant disparity 모델을 최소자승 피팅(1차):

$$
(s,t)=\arg\min_{s,t}\sum_i\Big(target\_z_i - s\cdot\tfrac{1}{Z_i} - t\Big)^2 \tag{9}
$$

metric 깊이 복원:

$$
Z_{rec} = \frac{s}{target\_z - t} \tag{10}
$$

적합도(피어슨 상관, $x_i=1/Z_i,\;y_i=target\_z_i$):

$$
corr=\frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum(x_i-\bar x)^2}\sqrt{\sum(y_i-\bar y)^2}} \tag{11}
$$

**LODO(Leave-One-Distance-Out)**: (9)를 3개 거리로만 풀어 $(s,t)$를 얻고, 학습에 안 쓴 남은 거리에 (10)을 적용 → 못 본 거리에서의 정직한 일반화 성능 측정.

### 1.5 이상값 정제

**물리 스파이크 제거(본 보고서 채택).** 정지 물체는 깊이가 일정해야 하므로, (객체·거리·상태) 그룹의 중앙값 대비 편차가 큰 프레임 제거:

$$
\text{제외} \iff \frac{\big|\,target\_z_i - \mathrm{median}_g(target\_z)\,\big|}{\mathrm{median}_g(target\_z)} > 0.35 \tag{12}
$$

(참고 — 초기 MAD 방식: $\text{robust-}z=\dfrac{0.6745(x-\mathrm{median})}{MAD}$, $MAD=\mathrm{median}(|x-\mathrm{median}|)$, $|\text{robust-}z|>3.5$ 제외. 이 방식은 31% 제거로 과했음.)

### 1.6 분류 분리도 — Fisher 비 & 최적 threshold 정확도

**Fisher 판별비** (held $h$ vs elevated $e$):

$$
F=\frac{(\mu_h-\mu_e)^2}{\sigma_h^2+\sigma_e^2} \tag{13}
$$

**단일 threshold 최적 균형정확도.** 후보 $\theta$와 방향 $\mathrm{sgn}\in\{+1,-1\}$에 대해

$$
Acc(\theta,\mathrm{sgn})=\tfrac12\!\left(\frac{|\{h_i:\,\mathrm{sgn}\cdot h_i>\mathrm{sgn}\cdot\theta\}|}{N_h}+\frac{|\{e_j:\,\mathrm{sgn}\cdot e_j\le\mathrm{sgn}\cdot\theta\}|}{N_e}\right) \tag{14}
$$

$$
Acc^\star=\max_{\theta,\;\mathrm{sgn}} Acc(\theta,\mathrm{sgn}) \tag{15}
$$

균형정확도는 held/elevated 표본 수가 달라도 공정하게 평가한다.

---

## 2. 실험 환경

### 2.1 파이프라인
```
Camera → YOLOv8n(conf=0.10) → SAM(bbox 프롬프트) → Depth → 특징 계산
```
### 2.2 재캘리브레이션
| RMS | $f_x$ | $f_y$ | $c_x$ | $c_y$ |
| --- | --- | --- | --- | --- |
| **0.2619 px** (초판 0.8→개선) | 964.86 | 964.45 | 636.84 | 359.35 |

### 2.3 Semantic Prior $A_{real}$
| 객체 | 실측 | $A_{real}$ [mm²] |
| --- | --- | --- |
| cell phone | 163.6×78.1 | 12,773.16 |
| bottle | 215×65 | 13,975.0 |
| mouse | 113×61 | 6,893.0 |
| keyboard | 430×150 | 64,500.0 |
| book | 274×215 | 58,910.0 |

---

# PART Ⅰ. 방법① — 크기·거리 불변 (원근 보정 가설)

## 3. 실험 A — `target_z`가 disparity임을 확인 (Dataset-1)

거리별 평균 `target_z` [식(4)]:

| 거리 [mm] | 500 | 1000 | 1500 | 2000 |
| --- | --- | --- | --- | --- |
| mean `target_z` | 7.401 | 5.049 | 2.909 | 1.933 |

멀수록 **단조 감소** → disparity($\propto 1/Z$) 확정. 식(3)의 $1/Z^4$ 거동과 일치 (bottle held 500→2000mm: 29.28→0.46, ≈64배 감소).

## 4. 실험 B·E — 글로벌 Scale&Shift의 한계 (Dataset-1)

### 4.1 글로벌 피팅 [식(9)]
$$target\_z \approx 3422.2\cdot\tfrac{1}{Z} + 0.7501,\qquad corr=0.754$$

### 4.2 LODO 교차검증 [§1.4]
| | raw | global(in-sample) | **LODO(정직)** | ideal(oracle Z) |
| --- | --- | --- | --- | --- |
| 전체 통합 CV | 1.41 | 0.53 | **1.39** | 0.13 |

- $(s,t)$ 불안정: held-out에 따라 $s$ 37% 변동, $t$ 부호 반전(−1.24~+1.15).
- **LODO CV 1.39 ≈ raw 1.41**: 못 본 거리엔 글로벌 보정이 무보정과 동급. 식(10)의 역수가 원거리($target\_z\to t$)에서 노이즈 폭발.

> **결론**: disparity 후처리로는 가설 충족 불가 → **Metric 재수집 필요**(실험 F). 글로벌 피팅은 실제 거리 GT를 알고도 실패했다는 점에서 재촬영의 정당화 근거.

## 5. 실험 F — Metric 재촬영으로 가설 검증 ★ (Dataset-2)

### 5.1 방법
- 모델: **Depth-Anything-V2-Metric-Indoor-Base-hf** (미터 깊이 직접 출력, GT 앵커·피팅 불필요)
- 객체: book, bottle / 상태: elevated, held_in_hand / 거리: 1.0·1.5·2.0·2.5 m
- 총 3,562 프레임 → 식(12)로 정제 → **2,900행** (제거 18.6%, 대부분 투명 PET 병·held)

### 5.2 결과 — 거리 불변성 [식(8)]

| object·state | 옛 disparity | **metric(정제)** | 판정 |
| --- | --- | --- | --- |
| **book · elevated** | ~1.4 | **0.055** | ✅★ ideal 초과 달성 |
| **bottle · elevated** | ~1.4 | **0.112** | ✅★ ideal(0.13) 수준 |
| book · held_in_hand | ~1.4 | 0.245 | ✅ (손 노이즈) |
| bottle · held_in_hand | ~1.4 | 0.334 | △ (투명+손) |

**거리별 중앙값 (거리 불변이면 4값이 비슷):**

| object·state | $target\_z$ (1.0/1.5/2.0/2.5) | $mask\_area$ | $Mask_{norm}$ |
| --- | --- | --- | --- |
| book·elev | 1.05 / 1.72 / 2.24 / 2.77 | 55220 / 21824 / 14028 / 9112 | **1.03 / 1.07 / 1.18 / 1.16** |
| bottle·elev | 1.16 / 1.66 / 2.06 / 2.53 | — | 1.10 / 0.97 / 0.89 / 0.81 |

- $target\_z$가 거리를 정확히 추종 → metric 정상. $mask\_area$는 $\propto 1/Z^2$로 감소. 두 효과가 상쇄되어 $Mask_{norm}$ 거의 평탄(식 3).
- **book·elevated $Mask_{norm}$ [1.03,1.07,1.18,1.16] → CV 0.055**: 원근 보정 가설 완전 성립.

### 5.3 해석
- **가설 검증 성공·재현성**: 두 독립 물체 모두 elevated에서 CV 0.05~0.11(ideal 수준). 초판 실패는 수식이 아니라 **disparity 센서** 탓이었음이 증명됨.
- **elevated > held**: 손 흔들림·투명(bottle) 때문에 held가 노이즈. 검증엔 **불투명·정지(elevated)**가 적합.
- **잔차 출처 전환**: 남은 CV는 깊이가 아니라 $A_{phys}$(마스크) 노이즈. (book 1.5m 초기 세션은 마스크 붕괴로 $Mask_{norm}$ 0.30이었으나, 재촬영으로 정상화 → CV 0.29→0.055)

---

# PART Ⅱ. 방법② — 행동 분류

## 6. 실험 C·D — 분류 특징 선정 (Dataset-1)

### 6.1 특징 비교 [식(13)(15), 거리 앎 가정, 거리별 최적 threshold]
| 특징 | bottle | keyboard |
| --- | --- | --- |
| mask_norm(raw) | 84~100% | 78~99% |
| **floor_margin** | **100%(전 거리)** | **89~100%** |

→ 상태 신호는 "물체 크기"가 아니라 **"바닥 대비 높이"**(floor_margin). 옛 disparity에서 floor_margin은 방향(held>elevated)이 일관됐다.

### 6.2 분류 특징 metric 변환은 역효과 (실험 D)
floor_margin을 식(10)으로 metric 변환 시 bottle 96.6%→75.4%로 **악화**(원거리 역수 노이즈). → **분류 브랜치는 raw floor_margin 사용**.

## 7. 실험 G — Metric floor_margin threshold 분석 (Dataset-2) ⚠️

metric 데이터에서 식(6)의 floor_margin으로 거리별 threshold[식(15)] 산출:

| obj | dist | held중앙 | elev중앙 | threshold | 방향 | Acc |
| --- | --- | --- | --- | --- | --- | --- |
| book | 1.0 | 2.328 | 2.220 | 2.255 | held> | 98.3% |
| book | 1.5 | 1.804 | 1.506 | 1.651 | held> | 94.3% |
| book | 2.0 | 1.184 | 1.460 | 1.375 | held**<** | 98.3% |
| book | 2.5 | 0.563 | 0.962 | 0.670 | held**<** | 100% |
| bottle | 1.0~2.5 | (낮음) | (높음) | 2.30/1.80/1.35/0.91 | held< | 100% |

**⚠️ 이 threshold는 확정하면 안 됨:**
1. **book 방향 반전**: 가까울 땐 held>elevated, 멀 땐 held<elevated로 규칙이 뒤집힘 → 일관된 물리 신호 없음, 거리별 과적합.
2. **bottle 100%는 artifact**: 투명 PET가 held일 때 깊이 과대추정(z 2.24~3.38) → floor_margin 하락. 진짜 "높이" 신호 아님.

→ metric floor_margin은 바닥검출+물체깊이 둘 다에 의존해 지금 데이터로는 지저분. **분류 threshold 확정엔 불투명·정자세 클린 데이터 재수집 필요**. (분류는 오히려 disparity가 깔끔했음)

---

## 7.5 그래프

**Fig.1 — Disparity vs Metric (핵심 결과)**
disparity는 $\propto 1/Z^4$로 붕괴(로그축, 2桁), metric은 0.8~1.2로 평탄 → 가설 검증.
![Fig.1](figures/fig1_disparity_vs_metric.png)

**Fig.2 — Metric mask_norm 거리별 분포 (box plot)**
elevated에서 거리별 분포가 겹침 = 거리 불변. book CV 0.055, bottle CV 0.112.
![Fig.2](figures/fig2_metric_boxplot.png)

**Fig.3 — target_z가 실제 거리를 추종**
metric 깊이가 참 거리를 선형 추종(z≈Z) → §3.1 전제 성립.
![Fig.3](figures/fig3_z_tracking.png)

**Fig.4 — 거리 불변성 CV 비교**
metric(elevated) 0.05~0.11 vs 옛 disparity 1.4 vs ideal 0.13.
![Fig.4](figures/fig4_cv_comparison.png)

**Fig.5 — mask_area $\propto 1/Z^2$ 확인**
겉보기 면적이 거리 제곱에 반비례(식 2)함을 실측 확인.
![Fig.5](figures/fig5_maskarea.png)

**Fig.6 — floor_margin 분류 (함정)**
book은 held/elevated 방향이 거리마다 뒤집힘 → 분류 threshold 확정 불가(실험 G).
![Fig.6](figures/fig6_floor_margin.png)

---

# PART Ⅲ. 통합

## 8. 2방법 아키텍처
| | 방법① 크기·거리 불변 | 방법② 행동 분류 |
| --- | --- | --- |
| 특징 | **metric** mask_norm [식(1)] | **raw** floor_margin [식(6)] |
| 깊이 정확 ↑ | 개선(CV↓) | raw 최적, metric 변환 역효과 |
| 검증 | 실험 F (CV 0.055) | 실험 C (89~100%) |
| 거리 의존 | metric으로 제거 | 남음 → 거리 구간별 threshold |
| threshold | 없음(상수) | floor_margin, 구간별 (미확정) |

```
                   ┌─ 방법① (metric mask_norm) : 거리구간 판정 + 거리불변 크기 ─┐
raw depth ─────────┤                                                          │──┐구간
(target_z, floor,  └──────────────────────────────────────────────────────────┘  │선택
 mask_area)        ┌─ 방법② (raw floor_margin) : 구간별 threshold로 held/elev ──┐◀─┘
                   └──────────────────────────────────────────────────────────┘
```

## 9. 종합 결론
| 질문 | 결론 |
| --- | --- |
| $Mask_{norm}$ 거리 의존 원인 | disparity → $1/Z^4$ [식(3)], 수식적 필연 |
| 기존 데이터 후처리로 보정? | ❌ LODO CV 1.39 ≈ raw |
| Metric으로 가설 검증? | ✅ **성공** (book·elev CV 0.055, bottle·elev 0.112) |
| 분류 주 특징? | **raw floor_margin** (disparity 89~100%) |
| 분류 metric 변환? | ❌ 역효과 |
| metric 분류 threshold 확정? | ❌ 아직(방향 불일치·투명 artifact) → 클린 재수집 필요 |

## 10. 한계 및 향후
1. **투명·held 신뢰 낮음**: bottle(투명)·held는 깊이 노이즈. 검증은 불투명·elevated 위주가 적합.
2. **방법② metric 미확정**: floor_margin 분류를 metric 클린 데이터(불투명·정자세)로 재수집해 거리 구간별 threshold 확정 필요.
3. **글로벌 근사**: §4의 metric 역변환은 장면별 $s,t$를 하나로 근사(corr 0.75). per-frame 정렬은 프레임당 앵커 2개↑ 필요.
4. **단일 카메라**: MacBook 내장 한정.
5. 향후: 거리 구간별 floor_margin threshold 테이블 확정 후 `vision/reasoning/affordance_engine.py` 반영, 독립 검증셋 일반화 확인.

---

## 부록. 재현 정보
- Dataset-1: `data/mask_norm_log.csv` (3,674) — 실험 A·B·C·D·E
- Dataset-2: `data/mask_norm_metric_log_clean.csv` (2,900, 원본 3,562에서 식(12) 정제) — 실험 F·G
- Metric 모델: `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf`
- 캘리브레이션: RMS 0.2619 / $f_x$964.86 $f_y$964.45 $c_x$636.84 $c_y$359.35
- 글로벌 disparity 피팅: `target_z ≈ 3422.2·(1/Z)+0.7501`, corr 0.754
- 정제 기준: 그룹 중앙값 대비 깊이 ±35% 초과 제거 [식(12)]
- 수집 스크립트: `tests/test_mask_norm_metric.py`
