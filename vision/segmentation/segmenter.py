import cv2
import numpy as np
import os
import torch
from ultralytics import SAM
import json
import copy

# 하위 패키지 모듈 참조 정합성 유지
from vision.stream import WebcamStream
from vision.detector import ObjectDetector
from vision.depth.depth_estimator import DepthEstimator
from vision.spatial.transformer import Spatial3DConverter
from vision.spatial.floor_detector import FloorPlaneDetector
from vision.reasoning.relation_graph import SpatialRelationGraph   # ➔ 8단계 관계 그래프 엔진 수입
from vision.reasoning.affordance_engine import AffordanceEngine # ➔ 8단계 어포던스 추론 엔진 수입

# [최적화 스케줄링 상수]
SAM_INTERVAL = 5     
DEPTH_INTERVAL = 10  

class ObjectSegmenter:
    def __init__(self):
        self.model = SAM("sam_b.pt")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"SAM 모델이 [{self.device}] 가속 엔진 위에서 성공적으로 로드되었습니다.")

    def segment_objects(self, frame, scene_data):
        if not scene_data or "objects" not in scene_data or not scene_data["objects"]:
            return frame, scene_data, []

        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        masks_list = []

        all_bboxes = [obj["yolo"]["bbox_2d"] for obj in scene_data["objects"] if obj is not None]
        if not all_bboxes:
            return frame, scene_data, []
            
        results = self.model(frame, bboxes=all_bboxes, device=self.device, verbose=False)
        result_masks = results[0].masks

        if result_masks is None:
            return frame, scene_data, []

        for idx, obj in enumerate(scene_data["objects"]):
            if idx >= len(result_masks.data):
                break
                
            mask_bool = result_masks.data[idx].cpu().numpy().astype(bool)
            mask_pixels = int(np.sum(mask_bool))
            
            y_indices, x_indices = np.where(mask_bool)
            cx = int(np.mean(x_indices)) if len(x_indices) > 0 else 0
            cy = int(np.mean(y_indices)) if len(y_indices) > 0 else 0

            obj["sam"] = {
                "mask_path": None,
                "mask_area": mask_pixels,
                "centroid_2d": [cx, cy]
            }
            masks_list.append(mask_bool)

            mask_overlay[mask_bool] = [0, 255, 0]
            cv2.circle(mask_overlay, (cx, cy), 5, (0, 0, 255), -1)

        annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
        return annotated_frame, scene_data, masks_list

    def overlay_cached_masks(self, frame, scene_data, cached_masks):
        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for idx, obj in enumerate(scene_data["objects"]):
            if idx >= len(cached_masks): break
            mask_bool = cached_masks[idx]
            
            y_indices, x_indices = np.where(mask_bool)
            cx = int(np.mean(x_indices)) if len(x_indices) > 0 else 0
            cy = int(np.mean(y_indices)) if len(y_indices) > 0 else 0
            
            obj["sam"] = {
                "mask_path": None,
                "mask_area": int(np.sum(mask_bool)),
                "centroid_2d": [cx, cy]
            }
            mask_overlay[mask_bool] = [0, 255, 0]
            cv2.circle(mask_overlay, (cx, cy), 5, (0, 0, 255), -1)
            
        annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.4, 0)
        return annotated_frame, scene_data


class SceneDepthAttacher:
    def __init__(self):
        pass

    def attach_depth(self, scene_data, masks_list, depth_data):
        if not scene_data or "objects" not in scene_data or not scene_data["objects"] or not masks_list or depth_data is None:
            return scene_data

        raw_depth_map = depth_data["raw_depth"]
        if len(raw_depth_map.shape) == 3:
            raw_depth_map = np.squeeze(raw_depth_map)
            
        h, w = raw_depth_map.shape[:2]

        for idx, obj in enumerate(scene_data["objects"]):
            if obj is None or not isinstance(obj, dict) or idx >= len(masks_list):
                continue
                
            mask_bool = masks_list[idx]
            if "sam" not in obj or "mask_area" not in obj["sam"]: continue
            mask_pixels = obj["sam"]["mask_area"]

            if mask_pixels == 0:
                obj["depth"] = {"mean_relative_depth": 0.0, "min_relative_depth": 0.0, "max_relative_depth": 0.0}
                continue

            if raw_depth_map.shape != mask_bool.shape:
                mask_bool = cv2.resize(
                    mask_bool.astype(np.uint8),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            roi_pixels = raw_depth_map[mask_bool]
            if roi_pixels.size == 0:
                obj["depth"] = {"mean_relative_depth": 0.0, "min_relative_depth": 0.0, "max_relative_depth": 0.0}
                continue

            roi_pixels_clipped = np.clip(roi_pixels, 0.0, None)

            obj["depth"] = {
                "mean_relative_depth": round(float(np.mean(roi_pixels_clipped)), 4),
                "min_relative_depth": round(float(np.min(roi_pixels_clipped)), 4),
                "max_relative_depth": round(float(np.max(roi_pixels_clipped)), 4)
            }

        return scene_data


# =====================================================================
# 🚀 메인 제어 엔진룸 (YOLO ➔ SAM ➔ Depth ➔ 3D ➔ Floor ➔ Graph ➔ Affordance 대통합)
# =====================================================================
if __name__ == "__main__":
    stream = WebcamStream()
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_engine = DepthEstimator()
    attacher = SceneDepthAttacher()
    spatial_converter = Spatial3DConverter()
    floor_detector = FloorPlaneDetector()
    relation_graph = SpatialRelationGraph() # ➔ 8단계-1 관계 생성기 장착
    affordance_engine = AffordanceEngine()   # ➔ 8단계-2 의미 추론기 장착

    frame_count = 0
    cached_depth_data = None 
    
    last_masks_list = []
    last_labels_sequence = []

    print("==========================================================")
    print("CV_AR: [8단계 어포던스 추론 연동] 대통합 파이프라인 실시간 가동")
    print("-> 30프레임마다 최상위 relations 그래프와 객체별 actions가 인쇄됩니다.")
    print("==========================================================")

    while True:
        ret, frame = stream.get_frame()
        if not ret:
            break
            
        frame_count += 1
        
        # 1. Depth 연산 스케줄링 (10프레임 주기)
        if frame_count == 1 or frame_count % DEPTH_INTERVAL == 0:
            cached_depth_data = depth_engine.get_depth_map(frame)
        
        # 2. YOLO 객체 탐지 (results[0] 패치 반영 완료)
        yolo_result = detector.detect(frame)
        current_scene = detector.build_scene(yolo_result, frame, frame_count)
        current_labels = [obj["label"] for obj in current_scene["objects"] if obj is not None]

        # 캐시 정합성 고속 디스패처
        can_use_cache = (
            frame_count % SAM_INTERVAL != 0 and 
            len(last_masks_list) == len(current_scene["objects"]) and
            current_labels == last_labels_sequence
        )

        if can_use_cache:
            annotated_frame, mid_scene = segmenter.overlay_cached_masks(frame, current_scene, last_masks_list)
            scene_with_depth = attacher.attach_depth(mid_scene, last_masks_list, cached_depth_data)
            scene_with_3d = spatial_converter.process_scene_3d(scene_with_depth)
        else:
            annotated_frame, mid_scene, last_masks_list = segmenter.segment_objects(frame, current_scene)
            scene_with_depth = attacher.attach_depth(mid_scene, last_masks_list, cached_depth_data)
            scene_with_3d = spatial_converter.process_scene_3d(scene_with_depth)
            last_labels_sequence = copy.deepcopy(current_labels)

        # 3. 7단계 가설면 매핑 레이어 주입
        scene_with_floor = floor_detector.update_scene_with_floor(scene_with_3d, cached_depth_data)

        # 4. [8단계-1 핵심 이식] 실시간 3D 기하학 수치 기반 위상 관계 트리플 그래프 빌드
        scene_with_relations = relation_graph.process_scene_relations(scene_with_floor)

        # 5. [8단계-2 핵심 이식] 라벨 속성 + 공간 상태 융합형 동적 어포던스(actions) 최종 추론
        final_scene = affordance_engine.infer_affordances(scene_with_relations)

        # 30프레임마다 결과 출력
        if frame_count % 30 == 0:
            print(f"\n--- [FRAME {frame_count}] 30fps 실시간 8단계 지능형 장면 그래프 완료 ---")
            print(json.dumps(final_scene, ensure_ascii=False, indent=2))
            
        cv2.imshow("CV_AR - Golden Master Unified Pipeline", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    stream.release()
    cv2.destroyAllWindows()