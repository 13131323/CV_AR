아래처럼 작성하면 작업 기록으로 충분합니다.

---

# 작업 내용 (2026-07-03)

## 1. Camera Calibration 적용(7-2)

### 작업 내용

* OpenCV 체커보드 기반 Camera Calibration 수행
* 총 36장의 이미지 중 34장을 사용하여 Calibration 완료
* Calibration 결과를 `vision/stream.py`에 반영
* Camera Matrix와 Distortion Coefficients를 상수로 분리

```python
CAMERA_MATRIX
DIST_COEFFS
```

* `WebcamStream` 생성 시

```python
self.camera_matrix = CAMERA_MATRIX.copy()
self.dist_coeffs = DIST_COEFFS.copy()
```

형태로 사용하도록 변경

---

## 2. Spatial3DConverter 수정

### 기존

고정값

```
fx = 900
fy = 900
cx = 640
cy = 360
```

사용

→ 실제 카메라와 맞지 않는 임시 Intrinsic Parameter 사용

---

### 변경

`camera_matrix`를 생성자에서 전달받도록 수정

```python
Spatial3DConverter(camera_matrix)
```

생성 시

```python
fx
fy
cx
cy
```

를 Camera Matrix에서 추출하도록 변경

---

### 이유

실제 Camera Calibration 결과를 사용해야

* 2D → 3D 역투영
* 좌표 변환
* 거리 계산

오차를 줄일 수 있기 때문

---

## 3. Vision Pipeline 수정

WebSocket 서버에서

기존

```python
Spatial3DConverter()
```

↓

변경

```python
Spatial3DConverter(camera_matrix=CAMERA_MATRIX)
```

으로 수정

이를 통해 모든 3D 좌표 계산이 동일한 Calibration 값을 사용하도록 통일

---

# Geometry Layer 검토(7-3)

7-3 내용 검토

기존 Geometry Layer는

```
mask_area
floor_depth_delta
target_z
```

만을 이용한 Rule-based 분류기였음

하지만 실험(Log02~Log15)에서

같은 스마트폰이라도

거리만 달라져

```
mask_area
```

가 크게 변하는 현상을 확인

즉

기존 Rule은

거리 변화에 일반화되지 못함

---

# Semantic Prior 적용

논문에서 제안한

Semantic-Geometric Fusion 구조를 코드에 반영

객체 종류별 Prior 추가

```python
semantic_prior_db
```

예)

```
cell phone
chair
bottle
suitcase
```

---

## 변경한 수식

기존

```
mask_area
```

만 사용

↓

변경

```
Mask_norm =
(mask_area × target_z²)
/ A_real
```

여기서

* mask_area : SAM Mask 면적
* target_z : 카메라와 객체 거리
* A_real : 객체의 실제 크기에 대한 Prior

---

## 변경 이유

단안 카메라는

```
면적 ∝ 1 / 거리²
```

특성을 가지므로

멀어질수록 같은 물체도

Mask Area가 급격히 감소

이를 보정하기 위해

거리와 카메라 내부 파라미터를 함께 사용하는 정규화 수식을 적용

또한 객체마다 실제 크기가 다르므로

Semantic Prior를 함께 사용하도록 구조 변경

---

# 확인한 사항

로그 출력 추가

```
chair: mask=80000.0, z=2.10, prior=250000.0, mask_norm=1.41120000
cell phone: mask=10838.0, z=1.10, prior=12000.0, mask_norm=1.09283167
```

형태로 계산 결과 확인

현재는

threshold

```
0.8
```

은 임시값이며

추후 실제 로그를 기반으로 다시 결정해야 함

---

# 해야 할 일

### 1.

실제 영상 여러 개에서

```
mask_norm
```

분포 수집

---


### 2.

Threshold 결정

예)

```
held_in_hand
0.42~0.73

elevated
0.05~0.18
```

처럼 실제 데이터 기반으로 결정

---

### 3.

Threshold를 코드에 반영

현재

```python
if mask_norm < 0.8:
```

↓

실험 결과 기반 값으로 수정

---

# 기타 사항

* Camera Calibration 값은 Vision Pipeline 전체에서 공통 사용하도록 통일
* 기존 임시 Intrinsic Parameter(900, 640, 360)는 제거
* Geometry Layer가 Semantic 정보를 활용하도록 구조 개선
* 현재는 **Semantic-Geometric Fusion 구조 구현 단계까지 완료**되었으며, **Threshold 최적화 및 성능 검증은 후속 실험이 필요**하다.
