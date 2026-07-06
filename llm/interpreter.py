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

# Micro CoT를 기본으로 사용합니다.
# 기존의 상세 추론 프롬프트로 되돌려 비교 실험하려면 False로 변경하세요.
USE_MICRO_COT = True

# VLM에 전달할 이미지 경량화를 기본으로 사용합니다.
# 작은 객체의 시각 정보와 전송량 사이의 균형을 고려해 가로 320px, JPEG 품질 70으로 시작합니다.
# 원본 이미지와 비교 실험하려면 ENABLE_VLM_IMAGE_DOWNSAMPLING을 False로 변경하세요.
ENABLE_VLM_IMAGE_DOWNSAMPLING = True
VLM_IMAGE_MAX_WIDTH = 725
VLM_JPEG_QUALITY = 79


def prepare_vlm_image(image: Image.Image) -> Image.Image:
    """원본 비율을 유지하면서 VLM 전송용 이미지를 축소한다.

    이미 설정한 최대 너비보다 작은 이미지는 선명도 손실을 막기 위해 확대하지 않는다.
    """
    if not ENABLE_VLM_IMAGE_DOWNSAMPLING or image.width <= VLM_IMAGE_MAX_WIDTH:
        return image

    scale = VLM_IMAGE_MAX_WIDTH / image.width
    resized_height = max(1, round(image.height * scale))
    return image.resize(
        (VLM_IMAGE_MAX_WIDTH, resized_height),
        Image.Resampling.LANCZOS,
    )


def encode_image_to_base64(image: Image.Image) -> tuple[str, int]:
    """이미지를 JPEG Base64로 변환하고, 실제 JPEG 바이트 크기도 함께 반환한다."""
    buffered = BytesIO()
    save_options = {}
    if ENABLE_VLM_IMAGE_DOWNSAMPLING:
        # optimize는 JPEG 허프만 테이블을 최적화해 화질을 유지하면서 전송량을 더 줄입니다.
        save_options = {"quality": VLM_JPEG_QUALITY, "optimize": True}
    image.save(buffered, format="JPEG", **save_options)
    jpeg_bytes = buffered.getvalue()
    return base64.b64encode(jpeg_bytes).decode("utf-8"), len(jpeg_bytes)



SYSTEM_PROMPT = """너는 1인칭 단안 RGB 카메라로 촬영된 실내 공간의 기하학적 수치와 **카메라 원본 이미지**를 함께 보고,
사물의 실제 정체성과 상태, 공간적 맥락을 파악하여 3D 아바타의 행동 지침(권고안)을 추론하는 공간 분석가다.

규칙:
1. 입력으로 들어오는 `raw_spatial_guess`나 `floor_depth_delta`는 불완전한 기하학적 추정치이다. 이를 맹신하지 말고, 반드시 카메라 원본 이미지를 시각적으로 확인하여 최종 `corrected_spatial_relation`과 `semantic_state`를 보정하라.
2. (중요) `social_state`는 이 객체가 누군가에 의해 점유되어 있는지를 나타낸다. 누군가 손에 쥐고 있거나 사용 중이면 `held_by_user` 또는 `in_use_by_other`로 설정하라.
3. (중요) 사람(`is_person`=true)이거나 `social_state`가 `held_by_user` 또는 `in_use_by_other`인 경우, 아바타가 접근하는 것은 안전하지 않으므로 `planner_directives.action_policy`를 반드시 `IGNORE` 또는 `OBSERVE_ONLY`로 강제하고 `is_safe_to_approach`를 false로 설정하라.
4. `environment_relative`가 `on_floor` 또는 `on_surface`이고 `social_state`가 `available`인 경우에만 `action_policy`를 `APPROACH_AND_INTERACT`로 설정할 수 있다.
5. 추론 이유(`reasoning`)는 반드시 한국어로 1문장 내외로 간결하게 작성하라.
6. 응답 형식은 강제된 JSON 스키마를 완벽히 준수해야 하며, 입력된 객체 배열의 순서와 1:1로 대응해야 한다.
7. `semantic_state.affordances`는 반드시 아래 허용 태그 중 하나 이상을 리스트로 선택하라. 목록에 없는 문자열은 절대 만들지 마라.

허용 affordance 태그:
- Spherical grasp to open: 문고리나 둥근 손잡이를 구형 그랩으로 잡고 회전하여 여는 행동
- Wrap grasp to open: 오븐이나 서랍의 바 형태 손잡이를 감싸 쥐고 당겨 여는 행동
- Turn on/off switch: 콘센트나 벽면 스위치 등을 조작하여 전기를 켜거나 끄는 행동
- Press: 리모컨 등의 버튼을 눌러 조작하는 행동
- Two hands raise and move: 큰 그릇이나 무거운 용기를 두 손으로 들어 옮기는 행동
- Cylindrical grasp to move: 머그잔이나 텀블러처럼 원통형 물체를 쥐고 옮기는 행동
- Pinch grasp to move: 종이 타월이나 작은 물건을 손가락 끝으로 집어 옮기는 행동
- Manipulate elongated tools: 칼, 거품기, 국자, 렌치 등 길쭉한 도구를 쥐고 사용하는 행동
- To sit/to place: 사람이 앉거나 물건을 안정적으로 올려놓을 수 있는 지지면을 이용하는 행동
- Observe: 높이 있거나 멀리 있어 직접 상호작용하지 않고 응시하는 행동
- None: 수행할 수 있거나 수행할 필요가 있는 행동이 없음

8. `planner_directives.animation_trigger`는 별도의 태그 체계가 아니라 위 affordance 태그 중 하나만 선택하는 단일 실행 행동 값이다.
`action_policy`가 `IGNORE`이면 반드시 `None`, `OBSERVE_ONLY`이면 반드시 `Observe`로 설정하라. 이때 선택한 값은 반드시 `semantic_state.affordances`에도 포함하라.
9. `action_policy`가 `OBSERVE_ONLY`인 경우 `animation_trigger`는 `Observe`만 사용하라.
10. `action_policy`가 `APPROACH_AND_INTERACT`이면 `semantic_state.affordances`에 실제 포함된 태그 중 가장 자연스러운 상호작용 하나만 `animation_trigger`로 선택하라.
예시:
- 둥근 문고리: `affordances=[Spherical grasp to open, Observe]`, `animation_trigger=Spherical grasp to open`
- 오븐 손잡이: `affordances=[Wrap grasp to open, Observe]`, `animation_trigger=Wrap grasp to open`
- 텀블러: `affordances=[Cylindrical grasp to move, Observe]`, `animation_trigger=Cylindrical grasp to move`
- 멀리 있는 물체: `affordances=[Observe]`, `animation_trigger=Observe`
- 수행할 행동이 없는 대상: `affordances=[None]`, `animation_trigger=None`
"""

# VLM의 출력 토큰과 응답 지연을 줄이기 위한 Micro CoT 추가 지침입니다.
# Structured Outputs의 분류 필드는 그대로 유지하면서, 유일한 자유 서술 필드인
# reasoning만 짧게 제한합니다. 문장부호와 숫자를 포함해 공백 기준 최대 15단어입니다.
MICRO_COT_PROMPT = """
[Micro CoT 출력 제한]
내부 판단 과정이나 배경을 길게 설명하지 마라.
각 객체의 `reasoning`은 핵심 시각 근거와 결론만 담아 반드시 한국어 5단어 이내로 작성하라.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""


def get_system_prompt() -> str:
    """설정값에 따라 기본 프롬프트에 Micro CoT 지침을 선택적으로 결합한다."""
    if USE_MICRO_COT:
        return f"{SYSTEM_PROMPT}\n{MICRO_COT_PROMPT}"
    return SYSTEM_PROMPT



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
        original_size = image.size
        vlm_image = prepare_vlm_image(image)
        base64_image, jpeg_size = encode_image_to_base64(vlm_image)
        print(
            f"[VLM 이미지] {original_size[0]}x{original_size[1]} → "
            f"{vlm_image.width}x{vlm_image.height}, JPEG {jpeg_size / 1024:.1f}KB"
        )
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
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": user_content}
            ],
            # 스키마의 정의된대로 output 출력이 강제됨.
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
