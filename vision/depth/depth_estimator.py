import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image

class DepthEstimator:
    def __init__(self):
        # Windows + NVIDIA GPU면 CUDA 사용
        # 하드웨어 자동 감지 예외 처리 (M4 맥북 등)
        if torch.cuda.is_available():
            self.device = "cuda:0"
            gpu_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            gpu_name = "Apple MPS"
        else:
            self.device = "cpu"
            gpu_name = "CPU"
        
        # 런타임 장치 로드 실패 시 CPU Fallback 예외 처리 적용
        try:
            print(f"Depth 모델을 [{self.device}] 장치에서 로드 시도 중...")
            self.pipe = pipeline(
                task="depth-estimation",
                model="depth-anything/Depth-Anything-V2-Base-hf",
                device=self.device
            )
        except Exception as e:
            print(f"[경고] 장치 로드 실패: {e}. CPU(-1)로 대체합니다.")
            self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", device=-1)
        print("Depth Anything V2 모델이 성공적으로 준비되었습니다.")

    def get_depth_map(self, frame):
        """
        입력 프레임의 해상도를 동적으로 감지하여 시각화용(0-255)과 정밀 계산용(float) 맵을 분리 반환합니다.
        """
        h, w = frame.shape[:2]
        
        # 1. OpenCV BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # [TypeError 해결] NumPy 배열을 Hugging Face 파이프라인이 요구하는 PIL Image 객체로 정밀 래핑
        pil_image = Image.fromarray(rgb_frame)
        
        # 2. 깊이 추론 실행
        pipe_output = self.pipe(pil_image)
        predicted_depth = pipe_output["predicted_depth"]
        
        # 데이터 타입 검증 (Tensor로 강제 통일 및 보간 준비)
        if not isinstance(predicted_depth, torch.Tensor):
            if isinstance(predicted_depth, Image.Image):
                predicted_depth = torch.from_numpy(np.array(predicted_depth)).float()
        
        # 3. 원본 해상도로 정밀 보간 (Raw Float Depth Map 확보 - 5단계 연산용)
        raw_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        return {"raw_depth": raw_depth}
             
        ##### visual depth 연산은 메인로직에서 불필요함으로 제거
        
        # # 4. 시각화용 0~255 정규화 맵 생성
        # d_min, d_max = raw_depth.min(), raw_depth.max()
        # if d_max - d_min > 1e-5:
        #     visual_depth = (raw_depth - d_min) / (d_max - d_min) * 255.0
        # else:
        #     visual_depth = np.zeros_like(raw_depth)

        # return {
        #     "visual_depth": visual_depth.astype(np.uint8),
        #     "raw_depth": raw_depth
        # }

    def get_object_depth(self, depth_dict, mask_bool):
        """
        SAM 마스크(True/False 행렬) 영역만 슬라이싱하여 오염 없는 순수 객체 깊이 통계를 추출합니다.
        """
        if mask_bool is None or np.sum(mask_bool) == 0:
            return {"mean_relative_depth": 0.0, "min_relative_depth": 0.0, "max_relative_depth": 0.0}

        # 정밀 계산용 raw_depth 행렬에서 마스크 영역 픽셀만 추출
        roi_pixels = depth_dict["raw_depth"][mask_bool]

        return {
            "mean_relative_depth": round(float(np.mean(roi_pixels)), 4),
            "min_relative_depth": round(float(np.min(roi_pixels)), 4),
            "max_relative_depth": round(float(np.max(roi_pixels)), 4)
        }

# =====================================================================
# 🚀 [4단계 단독 모듈 구동 및 마그마 컬러 가시화 테스트 베드]
# =====================================================================
if __name__ == "__main__":
    from vision.stream import WebcamStream
    stream = WebcamStream()
    estimator = DepthEstimator()
    
    print("==========================================================")
    print("CV_AR: 4단계 Depth 단독 검증을 시작합니다.")
    print("-> 가까운 물체는 밝고 환하게(노란색/흰색), 먼 배경은 어둡게(보라색/검은색) 출력됩니다.")
    print("-> 'q'를 누르면 안전하게 종료됩니다.")
    print("==========================================================")
    
    while True:
        ret, frame = stream.get_frame()
        if not ret: 
            break
        
        # 실시간 깊이 지도 데이터 딕셔너리 획득
        depth_data = estimator.get_depth_map(frame)
        
        # 의사 컬러맵(Magma)을 입혀 시각적 직관성 확보
        color_depth = cv2.applyColorMap(depth_data["visual_depth"], cv2.COLORMAP_MAGMA)
        
        # 화면 팝업 출력
        cv2.imshow("CV_AR - Depth Anything V2 (Validated)", color_depth)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
            
    stream.release()
    cv2.destroyAllWindows()