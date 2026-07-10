"""정량평가 test3: 전체 파이프라인 VLM 판단 결과 수집.

목적:
- 전체 Vision/Geometry/VLM 파이프라인을 실행한다.
- VLM 추론 결과 30개를 CSV/TXT로 저장한다.
- 결과 파일은 test_QE/test3_res_001.csv/txt부터 매번 새로 생성한다.
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
RESULT_PREFIX = "test3_res"
TARGET_SAMPLES = 30
PROCESS_EVERY_N_FRAMES = 5


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
        "test3: 전체 파이프라인 VLM 판단 결과 수집\n"
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
                "vlm_object_count",
                "vlm_inference_seconds",
                "json",
            ]
        )


def append_result(
    run_id: int,
    csv_result: Path,
    txt_result: Path,
    sample: int,
    frame_count: int,
    object_count: int,
    elapsed: float,
    result: dict,
) -> None:
    timestamp = now_text()
    compact_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    vlm_object_count = len(result.get("results", []))

    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                object_count,
                vlm_object_count,
                f"{elapsed:.6f}",
                compact_json,
            ]
        )

    with txt_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[run {run_id:03d} / sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {object_count}\n"
            f"vlm_object_count: {vlm_object_count}\n"
            f"vlm_inference_seconds: {elapsed:.6f}\n"
            f"json:\n{pretty_json}\n\n"
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


def main() -> int:
    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2
    from PIL import Image

    import llm.interpreter as interpreter
    import llm.server_websocket as server
    from llm.feature_extractor import DEFAULT_CONTEXT, build_inputs_from_scene
    from llm.schemas import SemanticInterpretationBatchInput
    from vision.stream import WebcamStream

    run_id, csv_result, txt_result = allocate_result_files()
    initialise_result_files(run_id, csv_result, txt_result)

    print("[test3] 전체 파이프라인 VLM 판단 결과 수집 시작")
    print(f"[test3] run_id: {run_id:03d}")
    print(f"[test3] CSV 저장: {csv_result}")
    print(f"[test3] TXT 저장: {txt_result}")
    print(f"[test3] VLM 추론 결과 {TARGET_SAMPLES}개 수집 후 자동 종료합니다. q를 누르면 중단합니다.")

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
                cv2.imshow("test3 - full pipeline VLM collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test3] 사용자 입력 q로 중단합니다.")
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
                cv2.imshow("test3 - full pipeline VLM collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test3] 사용자 입력 q로 중단합니다.")
                    return 1
                print("[test3] 입력 객체가 없어 VLM 호출을 건너뜁니다.")
                continue

            vlm_frame = draw_object_boxes(frame, inputs)
            pil_image = Image.fromarray(cv2.cvtColor(vlm_frame, cv2.COLOR_BGR2RGB))
            batch_input = SemanticInterpretationBatchInput(
                context=DEFAULT_CONTEXT,
                objects=inputs,
            )

            next_sample = sample + 1
            print(f"[test3] sample {next_sample}/{TARGET_SAMPLES} VLM 호출 (객체 {len(inputs)}개)")
            started_at = time.perf_counter()
            try:
                batch_output = interpreter.interpret_batch(batch_input, image=pil_image)
            except Exception as exc:
                print(f"[test3] VLM 호출 실패, 샘플로 집계하지 않습니다: {exc}")
                cv2.imshow("test3 - full pipeline VLM collection", vlm_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test3] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            elapsed = time.perf_counter() - started_at
            result = batch_output.model_dump(mode="json")
            append_result(
                run_id=run_id,
                csv_result=csv_result,
                txt_result=txt_result,
                sample=next_sample,
                frame_count=frame_count,
                object_count=len(inputs),
                elapsed=elapsed,
                result=result,
            )
            sample = next_sample

            print(
                f"[test3] sample {sample}/{TARGET_SAMPLES} 저장 완료 "
                f"({elapsed:.3f}초, 객체 {len(inputs)}개)"
            )

            cv2.imshow("test3 - full pipeline VLM collection", vlm_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test3] 사용자 입력 q로 중단합니다.")
                return 1

    except KeyboardInterrupt:
        print("\n[test3] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print("[test3] 30개 VLM 추론 결과 수집 완료")
    print(f"[test3] CSV: {csv_result}")
    print(f"[test3] TXT: {txt_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
