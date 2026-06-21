"""
Semantic Interpretation Layer 메인 로직

Geometry Layer가 산출한 SemanticInterpretationInput을 받아
Gemini에게 의미 해석을 요청하고 SemanticInterpretationOutput으로 파싱하여 반환한다.
"""

import json
from typing import Any

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .schemas import (
    SemanticInterpretationInput,
    SemanticInterpretationOutput,
    SemanticInterpretationBatchInput,
    SemanticInterpretationBatchOutput,
)

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """너는 1인칭 단안 RGB 카메라로 촬영된 실내 공간의 기하학적 수치와 **카메라 원본 이미지**를 함께 보고,
사물의 실제 정체성과 어포던스를 추론하는 공간 분석가다.

규칙:
1. 반드시 함께 전송된 이미지를 시각적으로 먼저 파악하여 주변 배경과 맥락(visual_context)을 서술하라.
2. 입력 JSON의 object_id는 이미지 상의 객체를 식별하는 번호이다.
3. detected_class는 YOLO의 낮은 정확도 탐지 결과이다. 수치 데이터보다 시각적 확인을 우선하여 진짜 정체성(object_identity)을 판별하라.
4. object_state는 사물 자체의 순수한 물리적 위상(공중에 떠 있는지/바닥인지/다른 표면 위인지)만 판단한다.
5. 앞서 서술한 배경(visual_context)과 사물 상태(object_state)를 종합하여 아바타가 할 수 있는 논리적 행동을 먼저 추론(affordance_reasoning)하라.
6. 마지막으로 최종 상호작용 가능성(interaction_state)을 4가지(held_by_user, currently_in_use, available, not_interactable) 중 하나로 엄격히 결정하라.

반드시 아래 JSON 스키마 형식으로만 응답하라. 정해진 순서(Chain of Thought)를 지켜야 한다.

{
  "results": [
    {
      "visual_context": string (주변 배경 묘사),
      "object_identity": string (진짜 사물 이름),
      "object_state": "elevated" | "on_floor" | "on_surface" | "unknown",
      "affordance_reasoning": string (배경과 상태를 고려한 행동 추론),
      "interaction_state": "held_by_user" | "currently_in_use" | "available" | "not_interactable",
      "confidence": number (0.0 ~ 1.0)
    }
  ]
}

입력은 "context"(촬영 상황)와 "objects"(객체 리스트)로 구성된다.
출력의 "results" 배열 순서는 입력 "objects" 배열 순서와 반드시 1:1로 대응해야 한다.
"""

_model = genai.GenerativeModel(
    model_name=GEMINI_MODEL,
    system_instruction=SYSTEM_PROMPT,
)


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
    한 프레임에 탐지된 객체 전체를 단 1회의 Gemini 호출로 일괄 해석한다.
    이미지(image)가 제공될 경우 멀티모달 프롬프트로 작동하여 CoT 과정을 거친다.
    """
    payload = batch_input.model_dump()
    payload_json = json.dumps(payload, ensure_ascii=False)

    contents = [payload_json]
    if image is not None:
        contents.insert(0, image)

    response = _model.generate_content(
        contents,
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    raw_text = response.text

    try:
        data = json.loads(raw_text)
        output = SemanticInterpretationBatchOutput(**data)
    except Exception as e:
        raise RuntimeError(
            f"Gemini 응답을 SemanticInterpretationBatchOutput으로 파싱하는 데 실패했습니다.\n"
            f"원본 응답: {raw_text}"
        ) from e

    if len(output.results) != len(batch_input.objects):
        raise RuntimeError(
            f"입력 객체 수({len(batch_input.objects)})와 "
            f"Gemini 결과 수({len(output.results)})가 일치하지 않습니다. "
            f"프롬프트의 1:1 대응 규칙을 다시 점검하세요."
        )

    return output