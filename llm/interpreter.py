"""
Semantic Interpretation Layer 메인 로직

Geometry Layer가 산출한 SemanticInterpretationInput을 받아
Gemini에게 의미 해석을 요청하고 SemanticInterpretationOutput으로 파싱하여 반환한다.
"""

import json
import base64
from io import BytesIO
from typing import Any
from PIL import Image

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .schemas import (
    SemanticInterpretationInput,
    SemanticInterpretationOutput,
    SemanticInterpretationBatchInput,
    SemanticInterpretationBatchOutput,
)

client = OpenAI(api_key=OPENAI_API_KEY)

def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

SYSTEM_PROMPT = """너는 1인칭 단안 RGB 카메라로 촬영된 실내 공간의 기하학적 수치와 **카메라 원본 이미지**를 함께 보고,
사물의 실제 정체성과 상태, 공간적 맥락을 파악하여 3D 아바타의 행동 지침(권고안)을 추론하는 공간 분석가다.

규칙:
1. 입력으로 들어오는 `raw_spatial_guess`나 `floor_depth_delta`는 불완전한 기하학적 추정치이다. 이를 맹신하지 말고, 반드시 카메라 원본 이미지를 시각적으로 확인하여 최종 `corrected_spatial_relation`과 `semantic_state`를 보정하라.
2. (중요) `social_state`는 이 객체가 누군가에 의해 점유되어 있는지를 나타낸다. 누군가 손에 쥐고 있거나 사용 중이면 `held_by_user` 또는 `in_use_by_other`로 설정하라.
3. (중요) 사람(`is_person`=true)이거나 `social_state`가 `held_by_user` 또는 `in_use_by_other`인 경우, 아바타가 접근하는 것은 안전하지 않으므로 `planner_directives.action_policy`를 반드시 `IGNORE` 또는 `OBSERVE_ONLY`로 강제하고 `is_safe_to_approach`를 false로 설정하라.
4. `environment_relative`가 `on_floor` 또는 `on_surface`이고 `social_state`가 `available`인 경우에만 `action_policy`를 `APPROACH_AND_INTERACT`로 설정할 수 있다.
5. 추론 이유(`reasoning`)는 반드시 한국어로 1문장 내외로 간결하게 작성하라.
6. 응답 형식은 강제된 JSON 스키마를 완벽히 준수해야 하며, 입력된 객체 배열의 순서와 1:1로 대응해야 한다.
"""




def interpret(input_data: SemanticInterpretationInput) -> SemanticInterpretationOutput:
    """
    객체 1개를 단독으로 해석한다. (단독 검증/디버깅용. 실시간 다중 객체에는 interpret_batch 사용)
    """
    batch_output = interpret_batch(
        SemanticInterpretationBatchInput(context=input_data.context, objects=[input_data]),
        image=None
    )
    return batch_output.results[0]


def interpret_batch(
    batch_input: SemanticInterpretationBatchInput,
    image: Any = None,
) -> SemanticInterpretationBatchOutput:
    """
    한 프레임에 탐지된 객체 전체를 단 1회의 GPT 호출로 일괄 해석한다.
    이미지(image)가 제공될 경우 멀티모달 프롬프트로 작동하여 CoT 과정을 거친다.
    """
    payload_json = batch_input.model_dump_json()

    user_content = []

    # 이미지가 있으면 Base64로 인코딩하여 GPT-4o Vision 형식에 맞게 추가
    if image is not None:
        base64_image = encode_image_to_base64(image)
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })
    
    # 텍스트(YOLO 수치 데이터 JSON) 추가
    user_content.append({
        "type": "text",
        "text": payload_json
    })

    try:
        # OpenAI의 Structured Outputs 기능 사용 (Pydantic 모델 자체를 응답 포맷으로 지정)
        response = client.beta.chat.completions.parse(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            response_format=SemanticInterpretationBatchOutput,
            temperature=0.2,
        )
        
        output = response.choices[0].message.parsed
        
    except Exception as e:
        raise RuntimeError(f"OpenAI GPT 호출 또는 파싱에 실패했습니다: {e}") from e

    if len(output.results) != len(batch_input.objects):
        raise RuntimeError(
            f"입력 객체 수({len(batch_input.objects)})와 "
            f"GPT 결과 수({len(output.results)})가 일치하지 않습니다. "
            f"프롬프트의 1:1 대응 규칙을 다시 점검하세요."
        )

    return output