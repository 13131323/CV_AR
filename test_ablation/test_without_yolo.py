"""Ablation: YOLO 없는 전체 파이프라인.

YOLO만 제거하고, YOLO bbox 대신 full-frame pseudo bbox를 사용한다.

파이프라인:
webcam frame
-> full-frame pseudo bbox scene
-> SAM
-> Depth Anything
-> Geometry Layer
-> FAST_STREAM data 생성
-> VLM Semantic Layer

생성 파일 3쌍:
1. without_yolo_time_res_XXX.csv/txt
   - YOLO 없는 전체 파이프라인 시간 30개 + 통계
2. without_yolo_fast_stream_res_XXX.csv/txt
   - FAST_STREAM 데이터 30개
3. without_yolo_vlm_res_XXX.csv/txt
   - VLM JSON 데이터 30개
"""

from __future__ import annotations

import csv
import json
import math
import statistics
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
RESULT_DIR = PROJECT_ROOT / "test_ablation" / "without_yolo"
TIME_PREFIX = "without_yolo_time_res"
FAST_PREFIX = "without_yolo_fast_stream_res"
VLM_PREFIX = "without_yolo_vlm_res"
TARGET_SAMPLES = 30
PROCESS_EVERY_N_FRAMES = 5


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path, Path, Path, Path, Path]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    max_run_id = 0
    for path in RESULT_DIR.glob(f"{TIME_PREFIX}_*.csv"):
        suffix = path.stem.removeprefix(f"{TIME_PREFIX}_")
        if suffix.isdigit():
            max_run_id = max(max_run_id, int(suffix))
    run_id = max_run_id + 1
    return (
        run_id,
        RESULT_DIR / f"{TIME_PREFIX}_{run_id:03d}.csv",
        RESULT_DIR / f"{TIME_PREFIX}_{run_id:03d}.txt",
        RESULT_DIR / f"{FAST_PREFIX}_{run_id:03d}.csv",
        RESULT_DIR / f"{FAST_PREFIX}_{run_id:03d}.txt",
        RESULT_DIR / f"{VLM_PREFIX}_{run_id:03d}.csv",
        RESULT_DIR / f"{VLM_PREFIX}_{run_id:03d}.txt",
    )


def initialise_files(
    run_id: int,
    time_csv: Path,
    time_txt: Path,
    fast_csv: Path,
    fast_txt: Path,
    vlm_csv: Path,
    vlm_txt: Path,
) -> None:
    started_at = now_text()
    header = (
        "Ablation: without YOLO full pipeline\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {started_at}\n"
        f"target_samples: {TARGET_SAMPLES}\n"
        "condition: YOLO removed. Full-frame pseudo bbox is used instead.\n\n"
    )
    time_txt.write_text(header + "[time results]\n", encoding="utf-8")
    fast_txt.write_text(header + "[fast stream results]\n", encoding="utf-8")
    vlm_txt.write_text(header + "[vlm results]\n", encoding="utf-8")

    with time_csv.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "row_type",
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "object_count",
                "total_pipeline_seconds",
                "mean",
                "std",
                "median",
                "q1",
                "q3",
                "iqr",
                "outlier_lower_bound",
                "outlier_upper_bound",
                "outlier_count",
                "outliers",
            ]
        )

    with fast_csv.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "object_count",
                "json",
            ]
        )

    with vlm_csv.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "object_count",
                "vlm_object_count",
                "json",
            ]
        )


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] * (high - pos) + ordered[high] * (pos - low)


def compute_stats(sample_rows: list[dict]) -> dict:
    values = [float(row["total_pipeline_seconds"]) for row in sample_rows]
    q1 = percentile(values, 25)
    q3 = percentile(values, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = [
        f"{row['sample']}:{float(row['total_pipeline_seconds']):.6f}"
        for row in sample_rows
        if float(row["total_pipeline_seconds"]) < lower
        or float(row["total_pipeline_seconds"]) > upper
    ]
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "outlier_lower_bound": lower,
        "outlier_upper_bound": upper,
        "outlier_count": len(outliers),
        "outliers": "; ".join(outliers) if outliers else "None",
    }


def build_full_frame_scene(frame, frame_count: int) -> dict:
    height, width = frame.shape[:2]
    return {
        "frame_metadata": {
            "frame_id": frame_count,
            "timestamp": time.time(),
            "camera_resolution": [width, height],
        },
        "scene": {
            "floor_detected": False,
            "floor_normal": [0, 1, 0],
            "camera_height": 0.0,
            "scene_summary": "YOLO removed: full-frame pseudo object",
        },
        "objects": [
            {
                "id": 0,
                "label": "unknown_full_frame",
                "yolo": {
                    "confidence": 1.0,
                    "bbox_2d": [0, 0, width - 1, height - 1],
                },
                "sam": None,
                "depth": None,
                "spatial_3d": None,
                "affordance": None,
                "description": None,
            }
        ],
    }


def fast_stream_from_inputs(inputs) -> list[dict]:
    return [
        {
            "object_id": inp.object_id,
            "target_z": inp.target_z,
            "centroid_y": inp.centroid_y,
            "bbox_2d": inp.bbox_2d,
        }
        for inp in inputs
    ]


def append_time_sample(
    time_csv: Path,
    time_txt: Path,
    run_id: int,
    sample: int,
    frame_count: int,
    object_count: int,
    total_pipeline_seconds: float,
) -> dict:
    timestamp = now_text()
    with time_csv.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "sample",
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                object_count,
                f"{total_pipeline_seconds:.6f}",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        )
    with time_txt.open("a", encoding="utf-8") as file:
        file.write(
            f"[sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {object_count}\n"
            f"total_pipeline_seconds: {total_pipeline_seconds:.6f}\n\n"
        )
    return {"sample": sample, "total_pipeline_seconds": total_pipeline_seconds}


def append_time_summary(time_csv: Path, time_txt: Path, run_id: int, sample_rows: list[dict]) -> None:
    stats = compute_stats(sample_rows)
    with time_csv.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "summary",
                f"{run_id:03d}",
                "",
                now_text(),
                "",
                "",
                "",
                f"{stats['mean']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['q1']:.6f}",
                f"{stats['q3']:.6f}",
                f"{stats['iqr']:.6f}",
                f"{stats['outlier_lower_bound']:.6f}",
                f"{stats['outlier_upper_bound']:.6f}",
                stats["outlier_count"],
                stats["outliers"],
            ]
        )
    with time_txt.open("a", encoding="utf-8") as file:
        file.write(
            "[summary]\n"
            f"mean: {stats['mean']:.6f}\n"
            f"std: {stats['std']:.6f}\n"
            f"median: {stats['median']:.6f}\n"
            f"q1: {stats['q1']:.6f}\n"
            f"q3: {stats['q3']:.6f}\n"
            f"iqr: {stats['iqr']:.6f}\n"
            f"outlier_lower_bound: {stats['outlier_lower_bound']:.6f}\n"
            f"outlier_upper_bound: {stats['outlier_upper_bound']:.6f}\n"
            f"outlier_count: {stats['outlier_count']}\n"
            f"outliers: {stats['outliers']}\n"
        )


def append_fast_stream(fast_csv: Path, fast_txt: Path, run_id: int, sample: int, frame_count: int, data: list[dict]) -> None:
    timestamp = now_text()
    compact_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(data, ensure_ascii=False, indent=2)
    with fast_csv.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [f"{run_id:03d}", sample, timestamp, frame_count, len(data), compact_json]
        )
    with fast_txt.open("a", encoding="utf-8") as file:
        file.write(
            f"[sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {len(data)}\n"
            f"json:\n{pretty_json}\n\n"
        )


def append_vlm(vlm_csv: Path, vlm_txt: Path, run_id: int, sample: int, frame_count: int, object_count: int, result: dict) -> None:
    timestamp = now_text()
    compact_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
    vlm_object_count = len(result.get("results", []))
    with vlm_csv.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [f"{run_id:03d}", sample, timestamp, frame_count, object_count, vlm_object_count, compact_json]
        )
    with vlm_txt.open("a", encoding="utf-8") as file:
        file.write(
            f"[sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {object_count}\n"
            f"vlm_object_count: {vlm_object_count}\n"
            f"json:\n{pretty_json}\n\n"
        )


def process_without_yolo_frame(frame, frame_count: int, modules: dict):
    scene_data = build_full_frame_scene(frame, frame_count)
    annotated_frame, scene_data, masks_list = modules["segmenter"].segment_objects(frame, scene_data)
    depth_map = modules["depth_estimator"].get_depth_map(frame)
    scene_data = modules["depth_attacher"].attach_depth(scene_data, masks_list, depth_map)
    scene_data = modules["spatial_converter"].process_scene_3d(scene_data)
    scene_data = modules["stabilizer"].process_scene(scene_data)
    scene_data = modules["floor_detector"].update_scene_with_floor(scene_data, depth_map)
    scene_data = modules["relation_graph"].process_scene_relations(scene_data)
    scene_data = modules["affordance_engine"].infer_affordances(scene_data)
    return scene_data, annotated_frame


def draw_overlay(frame, sample: int):
    import cv2

    display = frame.copy()
    cv2.putText(
        display,
        "Ablation: without YOLO / full-frame bbox",
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        display,
        f"sample {sample}/{TARGET_SAMPLES}",
        (24, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
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
    from llm.feature_extractor import DEFAULT_CONTEXT, build_inputs_from_scene
    from llm.schemas import SemanticInterpretationBatchInput
    from vision.depth.depth_estimator import DepthEstimator
    from vision.reasoning.affordance_engine import AffordanceEngine
    from vision.reasoning.relation_graph import SpatialRelationGraph
    from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
    from vision.spatial.floor_detector import FloorPlaneDetector
    from vision.spatial.stabilizer import CoordinateStabilizer
    from vision.spatial.transformer import Spatial3DConverter
    from vision.stream import CAMERA_MATRIX, WebcamStream

    (
        run_id,
        time_csv,
        time_txt,
        fast_csv,
        fast_txt,
        vlm_csv,
        vlm_txt,
    ) = allocate_result_files()
    initialise_files(run_id, time_csv, time_txt, fast_csv, fast_txt, vlm_csv, vlm_txt)

    print("[without_yolo] YOLO 없는 전체 파이프라인 시작")
    print(f"[without_yolo] run_id: {run_id:03d}")
    print(f"[without_yolo] time CSV/TXT: {time_csv} | {time_txt}")
    print(f"[without_yolo] fast CSV/TXT: {fast_csv} | {fast_txt}")
    print(f"[without_yolo] VLM CSV/TXT: {vlm_csv} | {vlm_txt}")

    modules = {
        "segmenter": ObjectSegmenter(),
        "depth_estimator": DepthEstimator(),
        "depth_attacher": SceneDepthAttacher(),
        "spatial_converter": Spatial3DConverter(camera_matrix=CAMERA_MATRIX),
        "stabilizer": CoordinateStabilizer(),
        "floor_detector": FloorPlaneDetector(),
        "relation_graph": SpatialRelationGraph(),
        "affordance_engine": AffordanceEngine(),
    }

    stream = WebcamStream()
    frame_count = 0
    sample = 0
    sample_rows: list[dict] = []

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.imshow("ablation - without YOLO", draw_overlay(frame, sample))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[without_yolo] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            total_started_at = time.perf_counter()
            scene_data, annotated_frame = process_without_yolo_frame(frame, frame_count, modules)
            inputs = build_inputs_from_scene(scene_data)
            if not inputs:
                print("[without_yolo] 입력 객체가 없어 샘플로 집계하지 않습니다.")
                continue

            fast_data = fast_stream_from_inputs(inputs)
            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            batch_input = SemanticInterpretationBatchInput(
                context=(
                    f"{DEFAULT_CONTEXT}. "
                    "Ablation condition: YOLO is removed. "
                    "The single input object was generated from a full-frame pseudo bounding box."
                ),
                objects=inputs,
            )

            next_sample = sample + 1
            print(f"[without_yolo] sample {next_sample}/{TARGET_SAMPLES} VLM 호출")
            try:
                batch_output = interpreter.interpret_batch(batch_input, image=pil_image)
            except Exception as exc:
                print(f"[without_yolo] VLM 호출 실패, 샘플로 집계하지 않습니다: {exc}")
                cv2.imshow("ablation - without YOLO", draw_overlay(annotated_frame, sample))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[without_yolo] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            total_pipeline_seconds = time.perf_counter() - total_started_at
            vlm_result = batch_output.model_dump(mode="json")

            append_fast_stream(fast_csv, fast_txt, run_id, next_sample, frame_count, fast_data)
            append_vlm(vlm_csv, vlm_txt, run_id, next_sample, frame_count, len(inputs), vlm_result)
            sample_row = append_time_sample(
                time_csv,
                time_txt,
                run_id,
                next_sample,
                frame_count,
                len(inputs),
                total_pipeline_seconds,
            )
            sample_rows.append(sample_row)
            sample = next_sample

            print(
                f"[without_yolo] sample {sample}/{TARGET_SAMPLES} 저장 완료 | "
                f"total_pipeline={total_pipeline_seconds:.3f}s, objects={len(inputs)}"
            )

            cv2.imshow("ablation - without YOLO", draw_overlay(annotated_frame, sample))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[without_yolo] 사용자 입력 q로 중단합니다.")
                return 1

    except KeyboardInterrupt:
        print("\n[without_yolo] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    append_time_summary(time_csv, time_txt, run_id, sample_rows)
    print("[without_yolo] 30개 샘플 수집 완료")
    print(f"[without_yolo] time CSV/TXT: {time_csv} | {time_txt}")
    print(f"[without_yolo] fast CSV/TXT: {fast_csv} | {fast_txt}")
    print(f"[without_yolo] VLM CSV/TXT: {vlm_csv} | {vlm_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
