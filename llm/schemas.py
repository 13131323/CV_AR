"""
Semantic Interpretation Layer (Layer 5) 입출력 스키마

입력: Geometry Layer(Layer 4)가 실시간으로 산출한 scene_data의 객체 1개 분량
출력: LLM(Gemini)이 보정한 사물 정체성/상태/상호작용 가능성
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


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

class SemanticInterpretationInput(BaseModel):
    """Geometry Layer -> Semantic Interpretation Layer 입력"""

    object_id: int = Field(..., description="객체 식별 ID (이미지 상의 Bounding Box 번호와 매칭됨)")
    bbox_2d: Optional[list[float]] = Field(None, description="YOLO Bounding Box [x1, y1, x2, y2]")
    detected_class: str = Field(..., description="YOLO가 탐지한 클래스명 (오탐지 가능성 있음)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO 탐지 신뢰도")
    mask_area: int = Field(..., ge=0, description="SAM 마스크 픽셀 면적")
    centroid_y: int = Field(..., description="마스크 중심점의 y좌표 (픽셀, 화면 세로축)")
    target_z: float = Field(..., description="카메라로부터 객체까지의 추정 거리(m), spatial_3d.z")
    near_distance: Optional[float] = Field(None, description="가장 가까운 인접 객체와의 거리(m)")
    floor_depth_delta: Optional[float] = Field(
        None, description="바닥 추정 깊이와 객체 깊이의 차이(m) (바닥 접촉 추론 참고용)"
    )
    raw_spatial_guess: Optional[str] = Field(
        None, description="Geometry Layer가 휴리스틱으로 추정한 공간 정보 (예: on_floor, behind_user). 참고용 후보 신호."
    )
    context: str = Field(
        default="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
        description="촬영 상황/카메라 시점 맥락",
    )


class Identity(BaseModel):
    class_name: str = Field(..., description="GPT가 이미지와 문맥을 보고 최종 보정한 실제 사물 정체성")
    is_person: bool = Field(..., description="해당 객체가 사람(person) 또는 사람의 신체 일부인지 여부")

class SpatialContext(BaseModel):
    camera_relative: str = Field(..., description="카메라(사용자)와의 상대적 위치 (예: in_front_of_user, far_away)")
    environment_relative: Literal["on_floor", "on_surface", "elevated", "floating", "held"] = Field(
        ..., description="지형지물과의 위상 관계 보정 결과"
    )

class SemanticState(BaseModel):
    social_state: Literal["available", "held_by_user", "in_use_by_other"] = Field(
        ..., description="사물의 점유/사용 상태. 아바타가 건드려도 되는지 판별하는 핵심 정보."
    )
    affordances: list[AffordanceTag] = Field(..., description="이 사물에 대해 가능한 행동 목록")

class PlannerDirectives(BaseModel):
    action_policy: Literal["APPROACH_AND_INTERACT", "OBSERVE_ONLY", "IGNORE"] = Field(
        ..., description="ActionPlanner를 위한 권고 행동 정책 (권고안일 뿐 최종 명령은 아님)"
    )
    animation_trigger: str = Field(..., description="상호작용 시 사용할 애니메이션 이름 (예: grasp, drink, nod)")
    is_safe_to_approach: bool = Field(..., description="위험물이나 타인이 사용중인 물건이 아닌지 여부")


class SemanticInterpretationOutput(BaseModel):
    """Semantic Interpretation Layer -> 다음 레이어(Action Planning) 출력 (V2 계층 구조)"""
    
    object_id: int = Field(..., description="입력받은 object_id를 그대로 반환하여 ActionPlanner가 최신 좌표를 매핑하도록 함")
    identity: Identity
    corrected_spatial_relation: SpatialContext
    semantic_state: SemanticState
    planner_directives: PlannerDirectives
    reasoning: str = Field(..., description="상태 판단 및 정책 도출 이유 (한국어, 1문장 내외)")


class SemanticInterpretationBatchInput(BaseModel):
    """한 프레임에 탐지된 객체 여러 개를 한 번에 GPT에 전달하기 위한 컨테이너"""

    context: str
    objects: list[SemanticInterpretationInput]


class SemanticInterpretationBatchOutput(BaseModel):
    """GPT가 한 번에 반환하는 객체별 결과 리스트"""

    results: list[SemanticInterpretationOutput]