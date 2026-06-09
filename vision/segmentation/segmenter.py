import cv2
import numpy as np
import os
import torch
from ultralytics import SAM
import json

# 상위 패키지 모듈 참조 정합성 유지
from vision.stream import WebcamStream
from vision.detector import ObjectDetector

# 글로벌 설정 제어 스위치
SAVE_MASKS = False     # 디스크 폭발 방지를 위해 False 유지
CONF_THRESHOLD = 0.25  # 너무 낮지 않은 안정적인 시니어 권장 임계값 적용

class ObjectSegmenter:
    def __init__(self):
        # M4 맥북 가속 환경과 가중치 이름 표준 정합 완료
        self.model = SAM("sam_b.pt")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.mask_dir = "data/masks"
        
        if SAVE_MASKS:
            os.makedirs(self.mask_dir, exist_ok=True)
            
        print(f"SAM 모델이 [{self.device}] 가속 엔진 위에서 성공적으로 로드되었습니다.")

    def segment_objects(self, frame, scene_data):
        """
        YOLO의 2D 바운딩 박스를 Batch(일괄)로 묶어 SAM에 단 '1회'만 추론을 요청함으로써
        루프 연산 병목을 완전히 제거한 최고 효율의 분할 메서드입니다.
        """
        if not scene_data["objects"]:
            return frame, scene_data

        frame_id = scene_data["frame_metadata"]["frame_id"]
        
        # 모든 객체의 bbox 2D 좌표를 리스트로 컴프리헨션 수집
        all_bboxes = [obj["yolo"]["bbox_2d"] for obj in scene_data["objects"]]

        # 웅장한 다중 배치 추론 단 1회 쏘기 (test_sam.py로 검증 완료된 문법)
        results = self.model(frame, bboxes=all_bboxes, device=self.device, verbose=False)
        result_masks = results[0].masks

        if result_masks is None:
            return frame, scene_data

        # 마스크 중첩 렌더링 오염을 막기 위한 통합 단일 오버레이 캔버스
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)

        for idx, obj in enumerate(scene_data["objects"]):
            if idx >= len(result_masks.data):
                break
                
            # 개별 바이너리 마스크 추출 및 수치 역산
            mask_bool = result_masks.data[idx].cpu().numpy().astype(bool)
            mask_pixels = int(np.sum(mask_bool))
            
            y_indices, x_indices = np.where(mask_bool)
            cx = int(np.mean(x_indices)) if len(x_indices) > 0 else 0
            cy = int(np.mean(y_indices)) if len(y_indices) > 0 else 0

            mask_path = None
            if SAVE_MASKS:
                mask_filename = f"frame_{frame_id}_{obj['label']}_{obj['id']}.png"
                mask_path = os.path.join(self.mask_dir, mask_filename)
                cv2.imwrite(mask_path, (mask_bool * 255).astype(np.uint8))

            # [확정 계층 구조 스펙 준수] 비어있던 'sam' 방 정밀 데이터로 원자적 치환
            # TODO: 추후 7단계 이후 Tracker 도입 시 전역 유니크 ID로 정합 예정
            obj["sam"] = {
                "mask_path": mask_path,
                "mask_area": mask_pixels,
                "centroid_2d": [cx, cy]
            }

            # 통합 도화지에 초록색 마스크 레이어와 무게중심 빨간 점 누적
            mask_overlay[mask_bool] = [0, 255, 0]
            cv2.circle(mask_overlay, (cx, cy), 5, (0, 0, 255), -1)

        # 모든 루프가 끝난 뒤 원본 영상과 통합 마스크 캔버스를 단 1회 알파 블렌딩
        annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
        return annotated_frame, scene_data


# =====================================================================
# 🚀 [3단계 마일스톤 통합 파이프라인 무조건 실행부]
# 조건문 분기를 삭제하여 파이썬 모듈 호출 시 무조건 카메라가 가동됩니다.
# =====================================================================
stream = WebcamStream()
detector = ObjectDetector()
segmenter = ObjectSegmenter()

frame_count = 0
print("==========================================================")
print("CV_AR: [Batch 추론형 YOLO + SAM] 골든 마스터 파이프라인 통합 가동을 시작합니다.")
print("-> python -m vision.segmentation.segmenter 명령어로 구동 중")
print("-> 'q'를 누르면 안전하게 종료됩니다.")
print("==========================================================")

while True:
    ret, frame = stream.get_frame()
    if not ret:
        break
        
    frame_count += 1
    
    # 1~2단계 파이프라인: YOLO 추론 및 Scene 인터페이스 기반 1차 뼈대 생성
    yolo_result = detector.detect(frame)
    current_scene = detector.build_scene(yolo_result, frame, frame_count)
    
    # 3단계 파이프라인: 초고속 고효율 SAM 일괄 추론 및 데이터 누적 조립
    annotated_frame, final_scene = segmenter.segment_objects(frame, current_scene)
    
    # 30프레임(약 1초)에 한 번씩 마스터 인터페이스 데이터 결합 상태 출력 검증
    if frame_count % 30 == 0:
        print(f"\n--- [FRAME {frame_count}] 글로벌 구조화 데이터 완성형 (YOLO+SAM) ---")
        print(json.dumps(final_scene, ensure_ascii=False, indent=2))
        
    # 최종적으로 픽셀 마스크가 입혀진 실시간 스트림 화면 출력
    cv2.imshow("CV_AR - YOLO + SAM Golden Master Pipeline", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
stream.release()
cv2.destroyAllWindows()