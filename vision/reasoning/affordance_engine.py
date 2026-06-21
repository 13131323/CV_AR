import numpy as np

class AffordanceEngine:
    def __init__(self):
        """
        [8-2단계: 최종 연구 마스터형 어포던스 및 상태 추론기]
        - 피드백을 수용하여 'm(미터)' 단위를 'pseudo-unit'으로 정정하여 학술적 엄밀성을 확보했습니다.
        - description의 오해의 소지가 있던 문구를 'agent distance'로 정확히 수정했습니다.
        """
        # COCO 사물 고유의 정적 잠재 속성 사전 (Object Intrinsic Potential)
        self.property_registry = {
            "person": ["interactive", "communicative"],
            "chair": ["sittable", "movable"],
            "desk": ["placeable_on", "leanable_on"],
            "table": ["placeable_on", "leanable_on"],
            "cup": ["graspable", "drinkable", "placeable_on"],
            "wine glass": ["graspable", "drinkable"],
            "bottle": ["graspable", "drinkable", "openable"],
            "cell phone": ["graspable", "usable"],
            "laptop": ["graspable", "usable", "openable"],
            "book": ["graspable", "readable", "openable"],
            "remote": ["graspable", "usable"],
            "handbag": ["graspable", "openable", "carryable"],
            "backpack": ["graspable", "openable", "carryable"],
            "suitcase": ["graspable", "movable", "openable"],
            "umbrella": ["graspable", "openable"]
        }

        # 정적 속성 -> 실행 행동 매핑 사전
        self.property_to_action = {
            "graspable": "grasp",
            "usable": "use",
            "drinkable": "drink",
            "readable": "read",
            "openable": "open",
            "sittable": "sit",
            "movable": "move",
            "carryable": "carry"
        }

    def infer_affordances(self, scene_data):
        """
        [Properties ➔ Relations ➔ State ➔ Actions 4단 위상 추론 엔진]
        정량적 Pseudo 마진을 분석하여 상황 맥락적 어포던스(Context-conditioned Executable Affordance)를 도출합니다.
        """
        if not scene_data or "objects" not in scene_data or not scene_data["objects"]:
            return scene_data

        objects = scene_data["objects"]
        relations = scene_data.get("relations", [])

        # 에이전트(User) 고유 ID 식별 가드
        person_ids = [obj["id"] for obj in objects if obj is not None and obj["label"] == "person"]
        agent_id = person_ids[0] if person_ids else None
        
        for obj in objects:
            if obj is None:
                continue

            obj_id = obj["id"]
            label = obj["label"]

            # [피드백 2 반영] 계층형 구조화 확정: "properties"와 "actions" 격리 수용
            base_properties = self.property_registry.get(label, ["inspectable"])
            obj["affordance"] = {
                "properties": base_properties,
                "actions": []
            }
            
            # 본인이거나 기준 에이전트가 없는 경우 정적 상태 마감 처리
            if agent_id is None or obj_id == agent_id:
                obj["affordance"]["actions"] = ["interact"] if label == "person" else ["inspect"]
                obj["state"] = "agent_self" if label == "person" else "static_anchor"
                obj["description"] = "Self agent or global reference frame target."
                continue

            # 8-1단계 트리플 기하 정보 파싱 가드 (단방향 near 대응 양방향 검색)
            is_near_agent = False
            agent_distance = float("inf")

            for r in relations:
                if r["predicate"] == "near":
                    if (r["subject_id"] == obj_id and r["object_id"] == agent_id) or \
                       (r["subject_id"] == agent_id and r["object_id"] == obj_id):
                        is_near_agent = True
                        agent_distance = r.get("distance", 1.0)
                        break

            # floor_depth_delta를 활용한 공간 상태(State) 추론
            floor_margin = obj.get("floor_depth_delta", 0.0)
            object_state = "unknown"

            if abs(floor_margin) <= 0.5:
                object_state = "placed_on_floor"
            elif floor_margin > 0.5:
                object_state = "elevated" # 책상 위, 손에 들림 등을 모두 포괄하는 안전한 상태 정의
            elif floor_margin < -0.5:
                object_state = "background_layer"

            # 씬 그래프 탑레벨에 명시적 공간 상태 주입
            obj["state"] = object_state

            # 속성 + 기하 정량 마진 기반 행동 결정 규칙 엔진
            active_actions = []

            for prop in base_properties:
                if prop in self.property_to_action:
                    action_candidate = self.property_to_action[prop]
                    
                    # [피드백 1 반영] m 단사 주석 제거 및 보수적 pseudo-distance 임계값 기반 계층화 적용
                    if action_candidate in ["grasp", "drink", "read", "open", "carry"]:
                        if is_near_agent and agent_distance <= 0.5: # pseudo-distance threshold
                            active_actions.append(action_candidate)
                            
                    elif action_candidate == "use":
                        if is_near_agent and agent_distance <= 1.2:
                            active_actions.append("use")
                            
                    elif action_candidate == "sit":
                        if is_near_agent and agent_distance <= 1.2 and object_state == "placed_on_floor":
                            active_actions.append("sit")
                            
                    elif action_candidate == "move":
                        if is_near_agent and agent_distance <= 1.2:
                            active_actions.append("move")
                            
                elif prop in ["placeable_on", "leanable_on"]:
                    active_actions.append(prop.replace("able", ""))

            # 최종 동적 행동 리스트 빌드
            obj["affordance"]["actions"] = active_actions if active_actions else ["inspect"]
            
            # [피드백 1 반영] 의미론적 오류 전면 정정 (ground distance -> agent distance 및 m 표기 제거)
            obj["description"] = f"Object is {object_state} with agent distance {agent_distance} (pseudo-unit)."

        return scene_data


# =====================================================================
# 🚀 8-2단계 최종 정합성 유닛 테스트
# =====================================================================
if __name__ == "__main__":
    import json
    print("==========================================================")
    print("CV_AR: 8-2단계 완결 마스터본 AffordanceEngine 유닛 테스트")
    print("==========================================================")
    
    engine = AffordanceEngine()
    
    mock_scene_data = {
        "objects": [
            { "id": 0, "label": "person", "spatial_3d": {"x": 0.0, "y": 1.0, "z": 2.0} },
            { "id": 1, "label": "chair", "spatial_3d": {"x": 0.5, "y": 1.1, "z": 2.1}, "floor_depth_delta": -0.1 }, 
            { "id": 2, "label": "cell phone", "spatial_3d": {"x": -0.2, "y": 0.8, "z": 1.1}, "floor_depth_delta": 3.5 } 
        ],
        "relations": [
            { "subject_id": 0, "predicate": "near", "object_id": 1, "distance": 0.5 },
            { "subject_id": 0, "predicate": "near", "object_id": 2, "distance": 0.9 } 
        ]
    }
    
    result = engine.infer_affordances(mock_scene_data)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    assert "sit" in result["objects"][1]["affordance"]["actions"]
    assert "use" in result["objects"][2]["affordance"]["actions"]
    assert "grasp" not in result["objects"][2]["affordance"]["actions"]
    
    print("\n✅ 엄밀성 패치 완료: 8단계 추론기 가상 테스트 완벽 통과!")
    print("==========================================================")