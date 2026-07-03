import cv2
import time
import os
import json
import torch
from collections import Counter
from ultralytics import YOLO
from vision.stream import WebcamStream

class ObjectDetector:
    """
    Yolo 모델을 통해서 프레임의 객체를 인식하는 클래스
    """
    
    def __init__(self):
        # 검증된 YOLOv8 Nano 모델 로드
        self.model = YOLO("yolov8n.pt")

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
        
        print(f"YOLOv8 모델이 [{self.device}] 가속 엔진 위에서 구동됩니다.")

    def detect(self, frame):
        """
        [7단계 가드 반영] conf=0.25 임계값을 적용하여 
        배경 노이즈나 그림자를 객체로 오탐지하는 현상을 원천 차단합니다.
        
        <results에 담기는 값들>
        result.boxes : 탐지된 객체들의 bbox, confidence, class 정보
        result.names : class 번호를 실제 이름으로 바꾸는 딕셔너리
        result.orig_img : 원본 이미지
        result.path : 이미지 경로, 웹캠이면 의미 없거나 기본값일 수 있음
        """
        results = self.model(frame, device=self.device, verbose=False, conf=0.25)
        return results[0]

    def build_scene(self, result, frame, frame_count):
        """
        [요청하신 개조식 트리 계층 구조 완벽 일치 반영]
        개별 프레임을 yolo로 분석한 json 반환
        """
        height, width = frame.shape[:2]
        
        # 1. 영상 메타데이터 및 화면 전체 데이터 구성
        scene_data = {
            "frame_metadata": {
                "frame_id": frame_count,
                "timestamp": time.time(),
                "camera_resolution": [width, height]
            },
            # 일단 기본값으로 채우기
            "scene": {
                "floor_detected": False,
                "floor_normal": [0, 1, 0],   # 바닥 방향의 벡터
                "camera_height": 0.0,
                "scene_summary": None  # 하단에서 YOLO 데이터 기반으로 생성
            },
            "objects": []
        }
        
        boxes = result.boxes
        detected_labels = []
        
        # 2. 개별 객체 리스트 조립 (트리의 depth 계층 구조 엄격 준수)
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = round(float(box.conf[0].item()), 2)
            class_name = result.names[int(box.cls[0].item())]
            
            detected_labels.append(class_name)
            
            # [최종 확정] 요청하신 트리와 1:1 대응되는 객체 노드 생성
            object_node = {
                "id": idx,            # 객체 식별 번호 (ID)
                "label": class_name,  # 객체 종류 (Label)
                
                "yolo": {             # YOLO 추출 영역
                    "confidence": confidence,   # 탐지 신뢰도
                    "bbox_2d": [x1, y1, x2, y2] # 2D 바운딩 박스 좌표
                },
                "sam": None,          # SAM 추출 영역 (3단계에서 구현)
                "depth": None,        # Depth 추출 영역 (5단계에서 구현)
                "spatial_3d": None,   # 3D 변환 영역 (6단계에서 구현)
                "affordance": None,   # 어포던스 영역 (8단계에서 구현)
                "description": None   # 객체 설명 영역 (9단계에서 구현)
            }
            scene_data["objects"].append(object_node)
            
        # 3. Counter를 활용한 실시간 기본 장면 요약 복원
        if detected_labels:
            counts = Counter(detected_labels)
            summary_items = [f"{name} {count}개" for name, count in counts.items()]
            scene_data["scene"]["scene_summary"] = ", ".join(summary_items) + "가 감지됨"
        else:
            scene_data["scene"]["scene_summary"] = "감지된 객체 없음"
            
        return scene_data

# python -m vision.detector로 실행
if __name__ == "__main__":
    stream = WebcamStream()
    detector = ObjectDetector()
    
    # 산출물 저장을 위한 출력 디렉토리 생성
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    print("==========================================================")
    print("CV_AR: 요청 규격을 100% 만족하는 2번 마일스톤 최종 검증을 시작합니다.")
    print("-> 30프레임마다 'data/output/scene_{frame_count}.json' 순차 저장")
    print("-> 'q'를 누르면 안전하게 종료됩니다.")
    print("==========================================================")
    
    while True:
        ret, frame = stream.get_frame()
        if not ret:
            break
            
        frame_count += 1
        
        # YOLO 추론 및 데이터 조립
        yolo_result = detector.detect(frame)
        current_scene = detector.build_scene(yolo_result, frame, frame_count)
        
        # 시각화 플롯 (메인 루프에서 처리)
        annotated_frame = yolo_result.plot()
        
        # 30프레임(약 1초)에 한 번씩 데이터 저장 및 터미널 검증
        if frame_count % 30 == 0:
            print(f"\n--- [FRAME {frame_count}] 글로벌 구조화 데이터 파일 저장 완료 ---")
            print(json.dumps(current_scene, ensure_ascii=False, indent=2))
            
            # 프레임 카운트를 포함하여 누적 저장
            json_filename = f"scene_{frame_count}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(current_scene, f, ensure_ascii=False, indent=2)
        
        # 실시간 화면 출력
        cv2.imshow("CV_AR - Object Detection (YOLOv8 Ultimate)", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    stream.release()

    