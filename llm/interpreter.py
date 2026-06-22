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
사물의 실제 정체성과 어포던스를 추론하는 공간 분석가다.

규칙:
1. (중요) 지연 시간 단축을 위해 visual_context와 affordance_reasoning 작성 시 **최대 1문장(15단어 이내)의 명사구 위주로 극히 짧게** 서술하라.
2. 입력 JSON의 object_id는 이미지 상의 객체를 식별하는 번호이다.
3. detected_class는 YOLO의 낮은 정확도 탐지 결과이다. 수치 데이터보다 시각적 확인을 우선하여 진짜 정체성(object_identity)을 판별하라.
4. object_state는 사물 자체의 순수한 물리적 위상(공중에 떠 있는지/바닥인지/다른 표면 위인지)만 판단한다.
5. 앞서 서술한 배경(visual_context)과 사물 상태(object_state)를 종합하여 아바타가 할 수 있는 논리적 행동을 먼저 추론(affordance_reasoning)하라.
6. 마지막으로 최종 상호작용 가능성(interaction_state)을 4가지(held_by_user, currently_in_use, available, not_interactable) 중 하나로 엄격히 결정하라.
7. (중요) 사람(person)이나 사람의 신체 부위는 아바타의 상호작용 대상이 아니다. 사람 객체는 `is_interactable`을 `false`로, `action_policy`를 `ignore_for_avatar`로 강제 설정하라.

반드시 아래 JSON 스키마 형식으로만 응답하라. 정해진 순서(Chain of Thought)를 지켜야 한다.

{
  "results": [
    {
      "visual_context": string (배경 묘사, 15단어 이내),
      "object_identity": string (진짜 사물 이름),
      "object_state": "elevated" | "on_floor" | "on_surface" | "unknown",
      "affordance_reasoning": string (행동 추론, 15단어 이내),
      "interaction_state": "held_by_user" | "currently_in_use" | "available" | "not_interactable",
      "is_interactable": boolean,
      "affordances": ["action1", "action2", ...],
      "action_policy": "ignore_for_avatar" | "approach_and_interact" | "observe_only",
      "confidence": number (0.0 ~ 1.0)
    }
  ]
}

입력은 "context"(촬영 상황)와 "objects"(객체 리스트)로 구성된다.
출력의 "results" 배열 순서는 입력 "objects" 배열 순서와 반드시 1:1로 대응해야 한다.
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