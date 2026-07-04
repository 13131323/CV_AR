# CV-AR

CV-AR 프로젝트의 성능 개선 사항과 Windows 환경 설정 및 실행 방법을 정리한 문서입니다.

## 목차

- [성능 및 기능 개선](#성능-및-기능-개선)
  - [VLM 추론 지연 개선](#vlm-추론-지연-개선)
  - [비전 파이프라인 최적화](#비전-파이프라인-최적화)
  - [멀티스레딩 구조](#멀티스레딩-구조)
  - [어포던스 기반 명령 시스템](#어포던스-기반-명령-시스템)
  - [기타 수정 사항](#기타-수정-사항)
- [Windows 환경 설정](#windows-환경-설정)
- [서버 및 Unity 실행](#서버-및-unity-실행)
- [추가 검증 항목](#추가-검증-항목)

## 성능 및 기능 개선

### VLM 추론 지연 개선

현재 VLM 추론에는 약 **11~18초**가 소요됩니다. 이를 줄이기 위해 Micro CoT와 이미지 다운샘플링을 적용했습니다.

#### Micro CoT

- 변경 파일: `llm/interpreter.py`
- 설정 변수: `USE_MICRO_COT`
- 기본값: `True`
- 배경 묘사와 추론 과정을 각각 15단어 이내로 제한하는 프롬프트를 사용해 출력 토큰 수를 줄입니다.

#### VLM 입력 이미지 경량화

- 변경 파일: `llm/interpreter.py`
- 설정 변수: `ENABLE_VLM_IMAGE_DOWNSAMPLING`
- 처리 함수: `prepare_vlm_image()`
- 이미지의 가로 해상도를 최대 320px로 제한합니다.
- 원본의 가로세로 비율을 유지합니다.
- 가로 해상도가 이미 320px 이하인 경우 추가로 축소하지 않습니다.
- JPEG 압축 품질을 70으로 설정해 전송량을 줄입니다.

> 이미지 품질 저하로 인해 인식 오류가 발생하는지 추가 검증이 필요합니다.

### 비전 파이프라인 최적화

기존에는 YOLO → SAM → Depth Anything V2가 매 프레임 순차적으로 실행되었습니다. 연산량을 줄이기 위해 SAM과 Depth 결과에 캐싱을 적용했습니다.

- 변경 파일: `llm/server_websocket.py`
- YOLO는 비전 분석마다 실행합니다.
- SAM은 최초 1회 실행한 뒤, 기본적으로 비전 분석 5회마다 다시 실행합니다.
- 이전 프레임과 객체 ID 및 개수가 모두 일치하면 YOLO 바운딩 박스의 IoU를 계산합니다.
- IoU가 임계값 이상일 때 이전 SAM 결과를 재사용합니다.
- 객체 ID, 개수 또는 IoU 조건 중 하나라도 충족하지 못하면 캐싱 차례에도 SAM을 다시 실행합니다.
- SAM 실행 시 Depth도 함께 계산하며, SAM 결과를 재사용할 때 Depth 결과도 함께 재사용합니다.

기본 설정값은 다음과 같습니다.

```python
SAM_INTERVAL = 5
SAM_IOU_THRESHOLD = 0.7
```

IoU는 다음과 같이 계산합니다.

```text
IoU = 두 바운딩 박스가 겹친 면적 / 두 바운딩 박스가 차지하는 전체 면적
```

### 멀티스레딩 구조

비전 연산과 WebSocket 전송이 같은 루프에서 동작할 때 발생하는 전송 지연을 줄이기 위해 역할을 네 개의 스레드로 분리했습니다.

- 변경 파일: `llm/server_websocket.py`

| 스레드 | 역할 |
|---|---|
| WebSocket 네트워크 서버 | 서버 실행 및 Unity와의 데이터 송수신 이벤트 처리 |
| `ai_worker_thread` | Vision·Geometry Layer 연산, 결과 전송, VLM 작업 생성 및 호출 주기 판단 |
| `vlm_worker_thread` | 대기열의 작업을 가져와 VLM을 호출하고 판단 결과 전송 |
| `main_vision_loop` | 카메라 영상을 창에 표시하는 메인 스레드 |

`ai_worker_thread`는 Vision 및 Geometry 데이터를 약 6fps로 계속 전송합니다. VLM 호출이 필요하면 크기가 1인 큐에 작업을 넣으며, 기존 대기 작업이 있으면 이를 버리고 최신 프레임으로 교체합니다.

VLM 호출을 별도 스레드로 분리해 추론이 진행되는 동안에도 Vision 및 Geometry 데이터를 계속 최신 상태로 유지합니다.

### 어포던스 기반 명령 시스템

기존의 Observe, Grasp, Drink 태그에 Sit, Open, Press, Read, Write를 추가했습니다.

- 변경 파일: `llm/schemas.py`
- 변경 파일: `llm/interpreter.py`
- 변경 파일: `unity/CV_AR/Assets/Scripts/AI/ActionPlanner.cs`
- 변경 파일: `unity/CV_AR/Assets/Scripts/Avatar/AvatarController.cs`

```python
AffordanceTag = Literal[
    "Observe",
    "Grasp",
    "Drink",
    "Sit",
    "Open",
    "Press",
    "Read",
    "Write",
]
```

- VLM은 명시된 어포던스 목록 안에서 객체에 가장 적합한 행동을 선택합니다.
- 선택 결과를 `action_trigger`로 설정해 Unity에 전송합니다.
- `APPROACH_AND_INTERACT` 상태의 객체는 큐로 관리하며 순서대로 이동하고 상호작용합니다.
- 한 객체와 상호작용한 뒤 1초 동안 대기하고 다음 객체로 이동합니다.
- 큐의 최대 크기는 5입니다.

### 기타 수정 사항

- `vision/depth/depth_estimator.py`: 불필요한 시각화용 `visual_depth` 계산을 제거했습니다.
- `vision/segmentaion/segmenter.py`: `overlay_cached_masks()`의 캐싱 조건 연산 로직을 제거했습니다.

## Windows 환경 설정

초기 macOS/Linux 기준 실행 환경을 Windows에서도 사용할 수 있도록 가상환경 활성화 방식, 실행 명령, GPU 백엔드 감지 및 경로 처리를 조정했습니다.

### 가상환경 활성화

프로젝트 루트에서 사용하는 셸에 맞는 명령을 실행합니다.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD**

```cmd
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

PowerShell 실행 정책으로 인해 활성화되지 않는 경우, 현재 사용자 범위의 정책을 변경한 뒤 다시 실행합니다.

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### GPU 백엔드 자동 감지

실행 환경에 따라 CUDA, MPS, CPU 순서로 사용 가능한 연산 장치를 선택합니다.

```python
if torch.cuda.is_available():
    self.device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    self.device = "mps"
    gpu_name = "Apple MPS"
else:
    self.device = "cpu"
    gpu_name = "CPU"
```

| 환경 | 선택되는 백엔드 |
|---|---|
| Windows + NVIDIA GPU + CUDA 지원 PyTorch | `cuda:0` |
| macOS Apple Silicon + MPS 지원 PyTorch | `mps` |
| GPU 미지원 또는 관련 PyTorch 미설치 | `cpu` |

Windows에서 NVIDIA GPU를 사용하려면 CUDA 버전에 맞는 PyTorch가 필요합니다. 다음 명령으로 CUDA 사용 가능 여부를 확인할 수 있습니다.

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
```

결과가 `True`이면 CUDA를 사용할 수 있고, `False`이면 CPU 모드로 실행됩니다.

### 운영체제별 차이

| 항목 | macOS/Linux | Windows PowerShell |
|---|---|---|
| 가상환경 활성화 | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| 서버 실행 | `python -m llm.server_websocket` | `python -m llm.server_websocket` |
| 경로 구분자 | `/` | `\` |
| GPU 백엔드 | `mps` 또는 `cpu` | `cuda:0` 또는 `cpu` |

## 서버 및 Unity 실행

Python 서버는 패키지 import 경로 문제를 방지하기 위해 **프로젝트 루트에서 모듈 방식으로 실행**합니다.

```powershell
cd C:\Projects\CV-AR
.\.venv\Scripts\Activate.ps1
python -m llm.server_websocket
```

다음과 같은 직접 실행 방식은 사용하지 않습니다.

```powershell
python llm/server_websocket.py
```

AR 연동 테스트는 다음 순서로 진행합니다.

1. 프로젝트 루트에서 Python 가상환경을 활성화합니다.
2. `python -m llm.server_websocket`을 실행합니다.
3. 모델 로딩이 완료되고 OpenCV 웹캠 창이 표시되는지 확인합니다.
4. Unity에서 `CV_AR` 프로젝트를 엽니다.
5. Unity Editor의 Play 버튼을 누릅니다.
6. WebSocket을 통해 `FAST_STREAM`, `SUCCESS` 메시지가 수신되는지 확인합니다.

## 추가 검증 항목

- VLM 입력 이미지의 해상도와 JPEG 품질 저하가 인식 정확도에 미치는 영향
- `SAM_INTERVAL = 5`가 속도와 정확도 사이에서 적절한 값인지 여부
- `SAM_IOU_THRESHOLD = 0.7`이 캐싱 판단 기준으로 적절한지 여부
- VLM 추론 시간이 지나치게 길어질 경우 해당 결과를 폐기할지, 최신 데이터로 교체할지 여부
