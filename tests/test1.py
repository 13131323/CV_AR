"""이미지만 사용하는 VLM의 Micro CoT 단어 제한 실험(30 -> 0)."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "test_res" / "test1"
TEXT_RESULT = RESULT_DIR / "test1_res.txt"
CSV_RESULT = RESULT_DIR / "test1_res.csv"
WORD_LIMITS = tuple(range(30, -1, -1))

# 단어 수만 독립 변수로 유지하기 위해 이미지 경량화 조건은 모든 실험에서 고정한다.
VLM_IMAGE_MAX_WIDTH = 320
VLM_JPEG_QUALITY = 70


IMAGE_ONLY_SYSTEM_PROMPT = """
너는 1인칭 실내 카메라 이미지를 분석하는 공간 분석가다.
사용자 메시지에는 이미지 한 장만 주어진다. 이미지에서 명확히 보이는 주요 객체를 직접 탐지하고 분석하라.
이미지에 보이지 않는 객체를 추측하거나 만들어 내지 마라.
각 객체에는 화면에서 왼쪽에서 오른쪽 순서로 0부터 object_id를 부여하라.

corrected_spatial_relation.environment_relative는 on_floor, on_surface, elevated, floating, held 중 선택하라.
semantic_state.social_state는 available, held_by_user, in_use_by_other 중 선택하라.
사람 또는 사람이 들거나 사용 중인 객체에는 접근하지 않도록 안전한 action_policy를 선택하라.
affordances와 animation_trigger는 응답 JSON 스키마에 허용된 값만 사용하라.
반드시 지정된 JSON 스키마로만 응답하라.
"""


def reasoning_prompt(word_limit: int) -> str:
    if word_limit == 0:
        return """
[Micro CoT 출력 제한 - 실험 조건]
각 객체의 `reasoning`은 반드시 빈 문자열(`""`)로 출력하라.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""
    return f"""
[Micro CoT 출력 제한 - 실험 조건]
내부 판단 과정이나 배경을 길게 설명하지 마라.
각 객체의 `reasoning`은 핵심 시각 근거와 결론만 담아 반드시 한국어 {word_limit}단어 이내로 작성하라.
단어 수는 공백으로 구분하여 계산한다.
`reasoning` 외에는 JSON 스키마가 요구하는 값만 출력하라.
"""


def initialise_result_files() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    TEXT_RESULT.write_text(
        "test1: image-only VLM reasoning word-limit experiment (30 -> 0)\n"
        f"started_at: {started_at}\n"
        f"vlm_image_max_width: {VLM_IMAGE_MAX_WIDTH}\n"
        f"vlm_jpeg_quality: {VLM_JPEG_QUALITY}\n\n",
        encoding="utf-8",
    )
    with CSV_RESULT.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            ["experiment", "word_limit", "vlm_inference_seconds", "timestamp", "json"]
        )


def append_result(experiment: int, word_limit: int, elapsed: float, result: dict) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
    logged_result = {
        "vlm_inference_seconds": round(elapsed, 6),
        "vlm_result": result,
    }
    compact_json = json.dumps(logged_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)

    with TEXT_RESULT.open("a", encoding="utf-8") as file:
        file.write(
            f"[experiment {experiment:02d}]\n"
            f"word_limit: {word_limit}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"timestamp: {timestamp}\n"
            f"json:\n{pretty_json}\n\n"
        )

    with CSV_RESULT.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [experiment, word_limit, f"{elapsed:.6f}", timestamp, compact_json]
        )


def main() -> int:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import cv2
    from PIL import Image

    import llm.interpreter as interpreter
    from llm.schemas import SemanticInterpretationBatchOutput
    from vision.stream import WebcamStream

    initialise_result_files()
    interpreter.ENABLE_VLM_IMAGE_DOWNSAMPLING = True
    interpreter.VLM_IMAGE_MAX_WIDTH = VLM_IMAGE_MAX_WIDTH
    interpreter.VLM_JPEG_QUALITY = VLM_JPEG_QUALITY

    stream = WebcamStream()
    print("[test1] 이미지 단독 Micro CoT 실험을 시작합니다. q를 누르면 중단합니다.")
    try:
        for index, word_limit in enumerate(WORD_LIMITS):
            experiment = index + 1

            while True:
                ret, frame = stream.get_frame()
                if not ret:
                    time.sleep(0.01)
                    continue

                cv2.imshow("test1 - image only, reasoning word limit 30 to 0", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test1] 사용자 요청으로 실험을 중단합니다.")
                    return 1

                raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                system_prompt = f"{IMAGE_ONLY_SYSTEM_PROMPT}\n{reasoning_prompt(word_limit)}"

                print(
                    f"[test1] 실험 {experiment}/{len(WORD_LIMITS)} 시작: "
                    f"reasoning 최대 {word_limit}단어"
                )
                started_at = time.perf_counter()
                try:
                    vlm_image = interpreter.prepare_vlm_image(raw_image)
                    base64_image, jpeg_size = interpreter.encode_image_to_base64(vlm_image)
                    print(
                        f"[test1] 이미지 {raw_image.width}x{raw_image.height} -> "
                        f"{vlm_image.width}x{vlm_image.height}, {jpeg_size / 1024:.1f}KB"
                    )
                    response = interpreter.client.beta.chat.completions.parse(
                        model=interpreter.OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        },
                                    }
                                ],
                            },
                        ],
                        response_format=SemanticInterpretationBatchOutput,
                        temperature=0.2,
                    )
                    output = response.choices[0].message.parsed
                    if output is None:
                        raise RuntimeError("VLM이 파싱 가능한 JSON을 반환하지 않았습니다.")
                except Exception as error:
                    print(f"[test1] 실험 {experiment} 실패, 같은 조건으로 재시도합니다: {error}")
                    time.sleep(2.0)
                    continue

                elapsed = time.perf_counter() - started_at
                append_result(
                    experiment,
                    word_limit,
                    elapsed,
                    output.model_dump(mode="json"),
                )
                print(f"[test1] 실험 {experiment}/{len(WORD_LIMITS)} 저장 완료 ({elapsed:.3f}초)")
                break
    except KeyboardInterrupt:
        print("\n[test1] 사용자 요청으로 실험을 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print(f"[test1] 31개 실험 완료: {TEXT_RESULT} / {CSV_RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
