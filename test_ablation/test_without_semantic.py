"""Ablation: Semantic Layer 없는 전체 파이프라인.

VLM/Semantic Layer를 제거하고 FAST_STREAM 데이터 생성까지의 파이프라인만 실행한다.

생성 파일 2쌍:
1. Semantic Layer 없는 파이프라인 시간
   - without_semantic_time_res_XXX.csv/txt
   - FAST_STREAM 데이터 생성까지의 파이프라인 시간 30개 + 통계(mean/std/median/Q1/Q3/IQR/outlier)

2. FAST_STREAM 데이터
   - without_semantic_fast_stream_res_XXX.csv/txt
   - Vision/Geometry 이후 Unity로 빠르게 보내는 좌표 데이터 30개
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
RESULT_DIR = PROJECT_ROOT / "test_ablation" / "without_semantic"
TIME_PREFIX = "without_semantic_time_res"
FAST_PREFIX = "without_semantic_fast_stream_res"
TARGET_SAMPLES = 30
PROCESS_EVERY_N_FRAMES = 5


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path, Path, Path]:
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
    )


def initialise_files(
    run_id: int,
    time_csv: Path,
    time_txt: Path,
    fast_csv: Path,
    fast_txt: Path,
) -> None:
    started_at = now_text()
    header = (
        "Ablation: without Semantic Layer\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {started_at}\n"
        f"target_samples: {TARGET_SAMPLES}\n"
        "condition: VLM/Semantic Layer removed. Pipeline stops after FAST_STREAM data generation.\n\n"
    )
    time_txt.write_text(header + "[time results]\n", encoding="utf-8")
    fast_txt.write_text(header + "[fast stream results]\n", encoding="utf-8")

    with time_csv.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "row_type",
                "run_id",
                "sample",
                "timestamp",
                "frame_count",
                "sam_frame_count",
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
                "sam_frame_count",
                "object_count",
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
    sam_frame_count: int,
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
                sam_frame_count,
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
            f"sam_frame_count: {sam_frame_count}\n"
            f"object_count: {object_count}\n"
            f"total_pipeline_seconds: {total_pipeline_seconds:.6f}\n\n"
        )
    return {
        "sample": sample,
        "total_pipeline_seconds": total_pipeline_seconds,
    }


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


def append_fast_stream(
    fast_csv: Path,
    fast_txt: Path,
    run_id: int,
    sample: int,
    frame_count: int,
    sam_frame_count: int,
    data: list[dict],
) -> None:
    timestamp = now_text()
    compact_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(data, ensure_ascii=False, indent=2)
    with fast_csv.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                sam_frame_count,
                len(data),
                compact_json,
            ]
        )
    with fast_txt.open("a", encoding="utf-8") as file:
        file.write(
            f"[sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"sam_frame_count: {sam_frame_count}\n"
            f"object_count: {len(data)}\n"
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

    import llm.server_websocket as server
    from llm.feature_extractor import build_inputs_from_scene
    from vision.stream import WebcamStream

    (
        run_id,
        time_csv,
        time_txt,
        fast_csv,
        fast_txt,
    ) = allocate_result_files()
    initialise_files(run_id, time_csv, time_txt, fast_csv, fast_txt)

    print("[without_semantic] Semantic Layer 없는 파이프라인 시작")
    print(f"[without_semantic] run_id: {run_id:03d}")
    print(f"[without_semantic] time CSV/TXT: {time_csv} | {time_txt}")
    print(f"[without_semantic] fast CSV/TXT: {fast_csv} | {fast_txt}")
    print("[without_semantic] VLM/Semantic Layer는 호출하지 않고 FAST_STREAM 데이터 생성 후 저장합니다.")

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
    sample_rows: list[dict] = []

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.imshow("ablation - without Semantic Layer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[without_semantic] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            total_started_at = time.perf_counter()
            sam_frame_count += 1

            scene_data = server.build_scene_graph_for_frame(
                frame,
                frame_count,
                sam_frame_count,
                sam_cache,
            )
            inputs = build_inputs_from_scene(scene_data)
            if not inputs:
                print("[without_semantic] 입력 객체가 없어 샘플로 집계하지 않습니다.")
                cv2.imshow("ablation - without Semantic Layer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[without_semantic] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            fast_data = fast_stream_from_inputs(inputs)
            next_sample = sample + 1
            total_pipeline_seconds = time.perf_counter() - total_started_at

            append_fast_stream(
                fast_csv,
                fast_txt,
                run_id,
                next_sample,
                frame_count,
                sam_frame_count,
                fast_data,
            )
            sample_row = append_time_sample(
                time_csv,
                time_txt,
                run_id,
                next_sample,
                frame_count,
                sam_frame_count,
                len(inputs),
                total_pipeline_seconds,
            )
            sample_rows.append(sample_row)
            sample = next_sample

            print(
                f"[without_semantic] sample {sample}/{TARGET_SAMPLES} 저장 완료 | "
                f"pipeline_to_fast_stream={total_pipeline_seconds:.3f}s, objects={len(inputs)}"
            )

            display_frame = draw_object_boxes(frame, inputs)
            cv2.imshow("ablation - without Semantic Layer", display_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[without_semantic] 사용자 입력 q로 중단합니다.")
                return 1

    except KeyboardInterrupt:
        print("\n[without_semantic] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    append_time_summary(time_csv, time_txt, run_id, sample_rows)
    print("[without_semantic] 30개 샘플 수집 완료")
    print(f"[without_semantic] time CSV/TXT: {time_csv} | {time_txt}")
    print(f"[without_semantic] fast CSV/TXT: {fast_csv} | {fast_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
