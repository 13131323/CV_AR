"""
Semantic Interpretation Layer (Layer 5) 입출력 스키마

입력: Geometry Layer(Layer 4)가 실시간으로 산출한 scene_data의 객체 1개 분량
출력: LLM(Gemini)이 보정한 사물 정체성/상태/상호작용 가능성
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class SemanticInterpretationInput(BaseModel):
    """Geometry Layer -> Semantic Interpretation Layer 입력"""

    detected_class: str = Field(..., description="YOLO가 탐지한 클래스명 (오탐지 가능성 있음)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="YOLO 탐지 신뢰도")
    mask_area: int = Field(..., ge=0, description="SAM 마스크 픽셀 면적")
    centroid_y: int = Field(..., description="마스크 중심점의 y좌표 (픽셀, 화면 세로축)")
    target_z: float = Field(..., description="카메라로부터 객체까지의 추정 거리(m), spatial_3d.z")
    near_distance: Optional[float] = Field(None, description="가장 가까운 인접 객체와의 거리(m)")
    floor_depth_delta: Optional[float] = Field(None, description="바닥 추정 깊이와 객체 깊이의 차이(m)")
    context: str = Field(
        default="1인칭 시점, 실내 공간을 스마트폰 카메라로 촬영 중",
        description="촬영 상황/카메라 시점 맥락",
    )


class SemanticInterpretationOutput(BaseModel):
    """Semantic Interpretation Layer -> 다음 레이어(Action Planning) 출력"""

    object_identity: str = Field(..., description="보정된 실제 사물 정체성 (예: smartphone)")
    object_state: Literal[
        "elevated", "on_floor", "on_surface", "unknown"
    ] = Field(..., description="사물 자체의 순수 물리적 위상 상태 (관계 정보 제외)")
    interaction_state: Literal[
        "held_by_user", "currently_in_use", "available", "not_interactable"
    ] = Field(..., description="사용자/아바타와의 관계 기반 상호작용 가능성")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "LLM이 자체 보고한 확신도. 통계적으로 보정(calibrate)된 값이 아니므로 "
            "정량 평가 지표로 사용하지 말 것. 필터링용 soft signal로만 참고."
        ),
    )
    reasoning: Optional[str] = Field(None, description="판단 근거 1줄 요약 (디버깅용)")


class SemanticInterpretationBatchInput(BaseModel):
    """한 프레임에 탐지된 객체 여러 개를 한 번에 Gemini에 전달하기 위한 컨테이너"""

    context: str
    objects: list[SemanticInterpretationInput]


class SemanticInterpretationBatchOutput(BaseModel):
    """Gemini가 한 번에 반환하는 객체별 결과 리스트"""

    results: list[SemanticInterpretationOutput]