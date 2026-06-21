"""
Semantic Interpretation Layer 메인 로직

Geometry Layer가 산출한 SemanticInterpretationInput을 받아
Gemini에게 의미 해석을 요청하고 SemanticInterpretationOutput으로 파싱하여 반환한다.
"""

import json

import google.generativeai as genai

from .config import GEMINI_API_KEY, GEMINI_MODEL
from .schemas import (
    SemanticInterpretationInput,
    SemanticInterpretationOutput,
    SemanticInterpretationBatchInput,
    SemanticInterpretationBatchOutput,
)

genai.configure(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """너는 1인칭 단안 RGB 카메라로 촬영된 실내 공간의 기하학적 수치를 보고,
사물의 실제 정체성과 물리적 상태를 추론하는 공간 분석가다.

규칙:
1. detected_class는 YOLO 탐지 결과이며 오탐지 가능성이 있다. 기하 정보(mask_area, target_z,
   centroid_y)와 의미 정보(detected_class)가 서로 모순되면, detected_class를 무조건 따르지 말고
   더 상식적으로 설명 가능한 object_identity를 우선 추론하라.
2. object_state는 사물 자체의 순수한 물리적 위상(공중에 떠 있는지/바닥인지/다른 표면 위인지)만
   판단한다. "누가 들고 있다" 같은 관계 정보는 object_state가 아니라 interaction_state에서 판단하라.
3. interaction_state는 사용자/아바타와의 관계(들려 있음, 사용 중, 상호작용 가능)를 판단한다.
4. 판단 근거가 불충분하면 무리하게 단정하지 말고 object_state를 "unknown"으로 출력하라.
5. confidence는 너 스스로의 확신 정도를 0~1 사이로 보고하되, 이는 통계적으로 검증된 값이 아니라
   참고용 신호임을 인지하고 과신하지 마라.

반드시 아래 JSON 스키마 형식으로만 응답하라. 다른 설명 텍스트는 포함하지 마라.

{
  "results": [
    {
      "object_identity": string,
      "object_state": "elevated" | "on_floor" | "on_surface" | "unknown",
      "interaction_state": "held_by_user" | "currently_in_use" | "available" | "not_interactable",
      "confidence": number (0.0 ~ 1.0),
      "reasoning": string (판단 근거 1줄)
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
        SemanticInterpretationBatchInput(context=input_data.context, objects=[input_data])
    )
    return batch_output.results[0]


def interpret_batch(
    batch_input: SemanticInterpretationBatchInput,
) -> SemanticInterpretationBatchOutput:
    """
    한 프레임에 탐지된 객체 전체를 단 1회의 Gemini 호출로 일괄 해석한다.
    객체 수만큼 API를 반복 호출하면 지연/Rate Limit 문제가 발생하므로,
    실시간 파이프라인에서는 반드시 이 함수를 사용한다.
    """
    payload = batch_input.model_dump()

    response = _model.generate_content(
        json.dumps(payload, ensure_ascii=False),
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