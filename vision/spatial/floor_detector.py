import numpy as np
import cv2
import json

class FloorPlaneDetector:
    """
    [Floor Hypothesis Generator]
    현재는 하단 ROI 기반의 Depth Median 값을 활용하여 바닥 후보(Floor Candidate) 레이어를 추정합니다.
    """
    def __init__(self, floor_roi_ratio=0.25):
        self.floor_roi_ratio = floor_roi_ratio

    def detect_floor(self, depth_data):
        """
        Depth Map의 하단 영역 분포를 분석하여 바닥 가설면의 정보를 반환합니다.
        """
        if depth_data is None or "raw_depth" not in depth_data:
            return {"success": False, "floor_normal": [0.0, 1.0, 0.0], "floor_depth": 0.0, "method": "none"}

        raw_depth = depth_data["raw_depth"]
        if len(raw_depth.shape) == 3:
            raw_depth = np.squeeze(raw_depth)

        h, w = raw_depth.shape[:2]
        
        # 화면 하단 ROI 영역 지정 (바닥 후보 영역)
        roi_start_y = int(h * (1.0 - self.floor_roi_ratio))
        floor_roi = raw_depth[roi_start_y:h, :]

        valid_pixels = floor_roi[floor_roi > 0]
        if valid_pixels.size == 0:
            return {"success": False, "floor_normal": [0.0, 1.0, 0.0], "floor_depth": 0.0, "method": "bottom_roi_median"}

        base_floor_depth = float(np.median(valid_pixels))

        # TODO: RANSAC Plane Fitting 도입 시 실제 normal 추정값으로 대체
        return {
            "success": True, 
            "floor_normal": [0.0, 1.0, 0.0], 
            "floor_depth": base_floor_depth,
            "method": "bottom_roi_median"
        }

    def update_scene_with_floor(self, scene_data, depth_data):
        """
        [최종 피드백 반영] 
        수식은 유지하되, 물리적 의미의 혼선을 막기 위해 필드명을 floor_depth_delta로 변경합니다.
        가까움(Nearness)과 높음(Height)의 개념적 오염을 방지하기 위해 주석을 엄밀하게 수정했습니다.
        """
        if not scene_data:
            return scene_data

        floor_info = self.detect_floor(depth_data)
        
        if "scene" not in scene_data:
            scene_data["scene"] = {}
            
        scene_data["scene"]["floor_detected"] = floor_info["success"]
        scene_data["scene"]["floor_normal"] = floor_info["floor_normal"]
        scene_data["scene"]["floor_depth"] = round(floor_info["floor_depth"], 3)
        scene_data["scene"]["floor_method"] = floor_info["method"]
        scene_data["scene"]["camera_height"] = 0.0

        if "objects" in scene_data and scene_data["objects"]:
            for obj in scene_data["objects"]:
                if obj is None or "spatial_3d" not in obj:
                    continue
                
                obj_z = obj["spatial_3d"].get("z", 0.0)

                if floor_info["success"] and obj_z > 0:
                    # [최종 수정 완료] 
                    # 양수(+): 객체가 바닥 후보 레이어보다 카메라에 가까움 (앞에 있음)
                    # 음수(-): 객체가 바닥 후보 레이어보다 카메라에서 더 멀리 위치함 (뒤에 있음)
                    # 주의: 이 값은 기하학적 '높이(Height)'가 아닌 '상대적 깊이 차이'만을 의미합니다.
                    margin = round(float(floor_info["floor_depth"] - obj_z), 3)
                    obj["floor_depth_delta"] = margin
                else:
                    obj["floor_depth_delta"] = 0.0

        return scene_data


# =====================================================================
# 🚀 7단계 FloorPlaneDetector 최종 정합성 유닛 테스트
# =====================================================================
if __name__ == "__main__":
    detector = FloorPlaneDetector()
    
    mock_depth_map = np.ones((720, 1280), dtype=np.float32) * 8.0
    mock_depth_data = {"raw_depth": mock_depth_map}
    
    mock_scene_data = {
        "scene": {},
        "objects": [
            {
                "id": 0,
                "label": "cup",
                "spatial_3d": {"x": 0.2, "y": 1.5, "z": 5.0}
            }
        ]
    }
    
    updated_scene = detector.update_scene_with_floor(mock_scene_data, mock_depth_data)
    
    print("➔ 7단계 최종 마스터본 JSON 출력:")
    print(json.dumps(updated_scene, indent=2, ensure_ascii=False))
    
    res_delta = updated_scene["objects"][0]["floor_depth_delta"]
    expected_delta = round(8.0 - 5.0, 3) # 3.0
    
    assert updated_scene["scene"]["floor_detected"] is True
    assert res_delta == expected_delta, f"수식 오류: {res_delta} vs {expected_delta}"
    assert "floor_depth_delta" in updated_scene["objects"][0], "필드명 패치 누락"
    
    print("\n✅ 학술적 엄밀성 검증 완료: 7단계 모듈 최종 마감!")