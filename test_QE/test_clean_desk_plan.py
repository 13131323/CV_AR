"""정량평가: VLM 책상 정리 계획 한 문장 생성 테스트.

목적:
- 전체 Vision/Geometry 파이프라인으로 책상 위 객체와 3D 좌표를 수집한다.
- VLM에게 "이 책상을 치워야 한다면 어떻게 할지"를 좌표 기반으로 판단하게 한다.
- 결과는 VLM이 생성한 자연어 청소 시퀀스 한 문장만 CSV/TXT에 저장한다.

생성 파일:
- test_QE/clean_desk_plan_res_XXX.csv
- test_QE/clean_desk_plan_res_XXX.txt
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "llm").is_dir() and (candidate / "vision").is_dir():
            return candidate
    raise RuntimeError("프로젝트 루트를 찾을 수 없습니다. llm/vision 폴더 위치를 확인하세요.")


PROJECT_ROOT = find_project_root()
RESULT_DIR = PROJECT_ROOT / "test_QE"
RESULT_PREFIX = "clean_desk_plan_res"
TARGET_SAMPLES = 30
PROCESS_EVERY_N_FRAMES = 5


CLEAN_DESK_SYSTEM_PROMPT = """너는 1인칭 카메라 이미지와 객체별 3D 좌표를 보고 책상 정리 순서를 계획하는 공간 정리 플래너다.

목표:
- 사용자가 "이 책상을 치워야 한다면 어떻게 할지"를 묻고 있다.
- 입력 객체들의 bbox, centroid_y, object_x, object_y, target_z, floor_depth_delta, near_distance를 근거로 어떤 물체를 어떤 순서로 치울지 결정한다.
- 책상 위 또는 사용자 가까이에 있는 작은 물건부터 우선 정리하고, 큰 물체/가구/고정된 배경 물체는 직접 치우기보다 마지막에 정돈 또는 제외한다.
- 서로 가까운 물체는 한 구역으로 묶어 처리하고, 높이/거리상 먼저 접근 가능한 물체부터 처리한다.
- 사람/신체로 보이는 대상은 청소 대상에서 제외한다.

출력 규칙:
- 반드시 한국어 자연어 한 문장만 출력한다.
- JSON, markdown, 번호 목록, 줄바꿈을 출력하지 마라.
- 문장 안에는 구체적인 순서가 드러나야 한다.
- 가능하면 좌표 근거를 자연스럽게 포함하라. 예: "가까운 오른쪽 앞쪽의 물병부터..."
- 너무 길게 쓰지 말고 한 문장으로 끝내라.
"""


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    max_run_id = 0
    for path in RESULT_DIR.glob(f"{RESULT_PREFIX}_*.csv"):
        suffix = path.stem.removeprefix(f"{RESULT_PREFIX}_")
        if suffix.isdigit():
            max_run_id = max(max_run_id, int(suffix))
    run_id = max_run_id + 1
    return (
        run_id,
        RESULT_DIR / f"{RESULT_PREFIX}_{run_id:03d}.csv",
        RESULT_DIR / f"{RESULT_PREFIX}_{run_id:03d}.txt",
    )


def initialise_result_files(run_id: int, csv_result: Path, txt_result: Path) -> None:
    txt_result.write_text(
        "clean_desk_plan: 좌표 기반 책상 정리 시퀀스 한 문장 생성 테스트\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {now_text()}\n"
        f"target_samples: {TARGET_SAMPLES}\n\n",
        encoding="utf-8",
    )
    with csv_result.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "object_count",
                "vlm_inference_seconds",
                "cleaning_sequence_sentence",
                "objects_json",
            ]
        )


def input_to_cleaning_object(inp) -> dict:
    return {
        "object_id": inp.object_id,
        "bbox_2d": inp.bbox_2d,
        "mask_area": inp.mask_area,
        "centroid_y": inp.centroid_y,
        "object_x": inp.object_x,
        "object_y": inp.object_y,
        "target_z": inp.target_z,
        "near_distance": inp.near_distance,
        "floor_depth_delta": inp.floor_depth_delta,
    }


def append_result(
    run_id: int,
    csv_result: Path,
    txt_result: Path,
    sample: int,
    frame_count: int,
    elapsed: float,
    sentence: str,
    objects: list[dict],
) -> None:
    timestamp = now_text()
    compact_objects = json.dumps(objects, ensure_ascii=False, separators=(",", ":"))
    pretty_objects = json.dumps(objects, ensure_ascii=False, indent=2)

    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                len(objects),
                f"{elapsed:.6f}",
                sentence,
                compact_objects,
            ]
        )

    with txt_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[run {run_id:03d} / sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {len(objects)}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"cleaning_sequence_sentence: {sentence}\n"
            f"objects:\n{pretty_objects}\n\n"
        )


def draw_object_boxes(frame, inputs):
    import cv2

    display = frame.copy()
    for inp in inputs:
        if not inp.bbox_2d:
            continue
        x1, y1, x2, y2 = map(int, inp.bbox_2d)
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Obj {inp.object_id}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return display


def normalize_one_sentence(text: str) -> str:
    sentence = " ".join(str(text).strip().split())
    if not sentence:
        return ""
    sentence = sentence.replace("```", "").strip()
    if "\n" in sentence:
        sentence = sentence.splitlines()[0].strip()
    return sentence


def request_cleaning_sequence(objects: list[dict], image) -> str:
    import llm.interpreter as interpreter
    from llm.config import OPENAI_MODEL

    payload = {
        "task": "어지럽혀진 책상을 좌표 기반으로 어떻게 청소할지 한 문장으로 제안하라.",
        "coordinate_system": "camera_opencv_meters: +X는 화면 오른쪽, +Y는 화면 아래, +Z/target_z는 카메라 전방 거리",
        "objects": objects,
    }

    user_content = []
    original_size = image.size
    vlm_image = interpreter.prepare_vlm_image(image)
    base64_image, jpeg_size = interpreter.encode_image_to_base64(vlm_image)
    print(
        f"[VLM 이미지] {original_size[0]}x{original_size[1]} → "
        f"{vlm_image.width}x{vlm_image.height}, JPEG {jpeg_size / 1024:.1f}KB"
    )
    user_content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
    )
    user_content.append(
        {
            "type": "text",
            "text": json.dumps(payload, ensure_ascii=False),
        }
    )

    response = interpreter.client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": CLEAN_DESK_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=180,
    )
    return normalize_one_sentence(response.choices[0].message.content or "")


def main() -> int:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2
    from PIL import Image

    import llm.server_websocket as server
    from llm.feature_extractor import build_inputs_from_scene
    from vision.stream import WebcamStream

    run_id, csv_result, txt_result = allocate_result_files()
    initialise_result_files(run_id, csv_result, txt_result)

    print("[clean_desk_plan] 좌표 기반 책상 정리 시퀀스 테스트 시작")
    print(f"[clean_desk_plan] run_id: {run_id:03d}")
    print(f"[clean_desk_plan] CSV 저장: {csv_result}")
    print(f"[clean_desk_plan] TXT 저장: {txt_result}")
    print(f"[clean_desk_plan] 청소 시퀀스 {TARGET_SAMPLES}개 수집 후 자동 종료합니다. q를 누르면 중단합니다.")

    server.init_vision_modules()
    sam_cache = {
        "last_labels": [],
        "last_bboxes": [],
        "last_masks_list": [],
        "last_sam_data": [],
        "cached_depth_map": None,
    }

    stream = WebcamStream()
    frame_count = 0
    sam_frame_count = 0
    sample = 0

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.imshow("clean desk plan test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[clean_desk_plan] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            sam_frame_count += 1
            scene_data = server.build_scene_graph_for_frame(
                frame,
                frame_count,
                sam_frame_count,
                sam_cache,
            )
            inputs = build_inputs_from_scene(scene_data)
            if not inputs:
                print("[clean_desk_plan] 입력 객체가 없어 VLM 호출을 건너뜁니다.")
                cv2.imshow("clean desk plan test", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[clean_desk_plan] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            objects = [input_to_cleaning_object(inp) for inp in inputs]
            vlm_frame = draw_object_boxes(frame, inputs)
            pil_image = Image.fromarray(cv2.cvtColor(vlm_frame, cv2.COLOR_BGR2RGB))

            next_sample = sample + 1
            print(f"[clean_desk_plan] sample {next_sample}/{TARGET_SAMPLES} VLM 호출 (객체 {len(inputs)}개)")
            started_at = time.perf_counter()
            try:
                sentence = request_cleaning_sequence(objects, pil_image)
            except Exception as exc:
                print(f"[clean_desk_plan] VLM 호출 실패, 샘플로 집계하지 않습니다: {exc}")
                cv2.imshow("clean desk plan test", vlm_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[clean_desk_plan] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            elapsed = time.perf_counter() - started_at
            append_result(
                run_id=run_id,
                csv_result=csv_result,
                txt_result=txt_result,
                sample=next_sample,
                frame_count=frame_count,
                elapsed=elapsed,
                sentence=sentence,
                objects=objects,
            )
            sample = next_sample
            print(f"[clean_desk_plan] sample {sample}/{TARGET_SAMPLES} 저장 완료 ({elapsed:.3f}초)")
            print(f"[clean_desk_plan] {sentence}")

            cv2.imshow("clean desk plan test", vlm_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[clean_desk_plan] 사용자 입력 q로 중단합니다.")
                return 1

    except KeyboardInterrupt:
        print("\n[clean_desk_plan] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print("[clean_desk_plan] 30개 청소 시퀀스 수집 완료")
    print(f"[clean_desk_plan] CSV: {csv_result}")
    print(f"[clean_desk_plan] TXT: {txt_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
