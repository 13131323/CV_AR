"""정량평가 test2: bottle 3D 위치 정보 정확도 측정.

측정 대상:
- 실제 Z 거리
- 예측 target_z
- Z축 평균 절대 오차(MAE)
- Z축 RMSE
- 프레임별 좌표 표준편차
- 정지 상태에서의 좌표 지터

실험 방식:
1. 실제 Z 거리 목록을 입력한다. 예: 0.4,0.6,0.8,1.0,1.2
2. 각 위치마다 bottle을 해당 거리에 놓고 Enter를 누른다.
3. 위치당 bottle 샘플 30개를 자동 수집한다.
4. 결과는 test_QE/test2_res_001.csv/txt부터 새 파일로 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
RESULT_PREFIX = "test2_res"
TARGET_SAMPLES_PER_POSITION = 30
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


def parse_distances(raw: str | None) -> list[float]:
    if raw:
        values = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        entered = input(
            "측정할 실제 Z 거리(m)를 쉼표로 입력하세요. 예: 0.4,0.6,0.8,1.0,1.2\n"
            "입력: "
        ).strip()
        values = [item.strip() for item in entered.split(",") if item.strip()]

    distances = []
    for value in values:
        try:
            distances.append(float(value))
        except ValueError as exc:
            raise ValueError(f"거리 값이 숫자가 아닙니다: {value}") from exc

    if not distances:
        raise ValueError("최소 1개 이상의 실제 Z 거리가 필요합니다.")
    return distances


def initialise_result_files(run_id: int, csv_result: Path, txt_result: Path, distances: list[float]) -> None:
    txt_result.write_text(
        "test2: bottle 3D 위치 정보 정확도 측정\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {now_text()}\n"
        f"samples_per_position: {TARGET_SAMPLES_PER_POSITION}\n"
        f"true_z_distances_m: {distances}\n\n",
        encoding="utf-8",
    )
    with csv_result.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "row_type",
                "run_id",
                "position_index",
                "sample",
                "timestamp",
                "frame_count",
                "true_z_m",
                "pred_target_z_m",
                "abs_z_error_m",
                "squared_z_error_m2",
                "object_x_m",
                "object_y_m",
                "object_z_m",
                "bbox_2d",
                "yolo_confidence",
                "mask_area",
                "centroid_x",
                "centroid_y",
                "n_samples",
                "mean_target_z_m",
                "z_mae_m",
                "z_rmse_m",
                "object_x_std_m",
                "object_y_std_m",
                "target_z_std_m",
                "mean_3d_jitter_m",
                "max_3d_jitter_m",
                "mean_z_jitter_m",
                "max_z_jitter_m",
            ]
        )


def write_sample_row(
    csv_result: Path,
    run_id: int,
    position_index: int,
    sample: int,
    frame_count: int,
    true_z: float,
    bottle: dict,
) -> None:
    spatial = bottle["spatial_3d"]
    sam = bottle["sam"]
    yolo = bottle["yolo"]
    pred_z = float(spatial["z"])
    abs_error = abs(pred_z - true_z)
    centroid = sam.get("centroid_2d", [None, None])
    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "sample",
                f"{run_id:03d}",
                position_index,
                sample,
                now_text(),
                frame_count,
                f"{true_z:.6f}",
                f"{pred_z:.6f}",
                f"{abs_error:.6f}",
                f"{abs_error ** 2:.9f}",
                f"{float(spatial['x']):.6f}",
                f"{float(spatial['y']):.6f}",
                f"{pred_z:.6f}",
                json.dumps(yolo.get("bbox_2d"), ensure_ascii=False),
                yolo.get("confidence"),
                sam.get("mask_area"),
                centroid[0],
                centroid[1],
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
                "",
            ]
        )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / (len(values) - 1))


def summarize_position(samples: list[dict], true_z: float) -> dict[str, float]:
    xs = [float(item["spatial_3d"]["x"]) for item in samples]
    ys = [float(item["spatial_3d"]["y"]) for item in samples]
    zs = [float(item["spatial_3d"]["z"]) for item in samples]
    abs_errors = [abs(z - true_z) for z in zs]
    squared_errors = [(z - true_z) ** 2 for z in zs]

    jitter_3d = []
    jitter_z = []
    for prev, curr in zip(samples, samples[1:]):
        px, py, pz = (
            float(prev["spatial_3d"]["x"]),
            float(prev["spatial_3d"]["y"]),
            float(prev["spatial_3d"]["z"]),
        )
        cx, cy, cz = (
            float(curr["spatial_3d"]["x"]),
            float(curr["spatial_3d"]["y"]),
            float(curr["spatial_3d"]["z"]),
        )
        jitter_3d.append(math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cz - pz) ** 2))
        jitter_z.append(abs(cz - pz))

    return {
        "n_samples": float(len(samples)),
        "mean_target_z_m": mean(zs),
        "z_mae_m": mean(abs_errors),
        "z_rmse_m": math.sqrt(mean(squared_errors)),
        "object_x_std_m": sample_std(xs),
        "object_y_std_m": sample_std(ys),
        "target_z_std_m": sample_std(zs),
        "mean_3d_jitter_m": mean(jitter_3d),
        "max_3d_jitter_m": max(jitter_3d) if jitter_3d else 0.0,
        "mean_z_jitter_m": mean(jitter_z),
        "max_z_jitter_m": max(jitter_z) if jitter_z else 0.0,
    }


def write_summary_row(
    csv_result: Path,
    txt_result: Path,
    run_id: int,
    position_index: int,
    true_z: float,
    summary: dict[str, float],
) -> None:
    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                "summary",
                f"{run_id:03d}",
                position_index,
                "",
                now_text(),
                "",
                f"{true_z:.6f}",
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
                "",
                int(summary["n_samples"]),
                f"{summary['mean_target_z_m']:.6f}",
                f"{summary['z_mae_m']:.6f}",
                f"{summary['z_rmse_m']:.6f}",
                f"{summary['object_x_std_m']:.6f}",
                f"{summary['object_y_std_m']:.6f}",
                f"{summary['target_z_std_m']:.6f}",
                f"{summary['mean_3d_jitter_m']:.6f}",
                f"{summary['max_3d_jitter_m']:.6f}",
                f"{summary['mean_z_jitter_m']:.6f}",
                f"{summary['max_z_jitter_m']:.6f}",
            ]
        )

    with txt_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[position {position_index}] true_z={true_z:.3f}m summary\n"
            f"n_samples: {int(summary['n_samples'])}\n"
            f"mean_target_z_m: {summary['mean_target_z_m']:.6f}\n"
            f"z_mae_m: {summary['z_mae_m']:.6f}\n"
            f"z_rmse_m: {summary['z_rmse_m']:.6f}\n"
            f"object_x_std_m: {summary['object_x_std_m']:.6f}\n"
            f"object_y_std_m: {summary['object_y_std_m']:.6f}\n"
            f"target_z_std_m: {summary['target_z_std_m']:.6f}\n"
            f"mean_3d_jitter_m: {summary['mean_3d_jitter_m']:.6f}\n"
            f"max_3d_jitter_m: {summary['max_3d_jitter_m']:.6f}\n"
            f"mean_z_jitter_m: {summary['mean_z_jitter_m']:.6f}\n"
            f"max_z_jitter_m: {summary['max_z_jitter_m']:.6f}\n\n"
        )


def select_bottle(scene_data: dict) -> dict | None:
    bottles = [
        obj
        for obj in scene_data.get("objects", [])
        if str(obj.get("label", "")).strip().lower() == "bottle"
        and obj.get("sam") is not None
        and obj.get("depth") is not None
        and obj.get("spatial_3d") is not None
    ]
    if not bottles:
        return None
    return max(bottles, key=lambda obj: float(obj.get("yolo", {}).get("confidence", 0.0)))


def draw_guidance(frame, position_index: int, true_z: float, sample: int, bottle: dict | None):
    import cv2

    display = frame.copy()
    lines = [
        f"test2 bottle position accuracy",
        f"position {position_index} | true Z = {true_z:.3f} m",
        f"samples {sample}/{TARGET_SAMPLES_PER_POSITION}",
        "Keep bottle still. Press q to stop.",
    ]
    for idx, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (24, 36 + idx * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 255),
            2,
        )

    if bottle is not None and bottle.get("yolo"):
        x1, y1, x2, y2 = map(int, bottle["yolo"]["bbox_2d"])
        pred_z = float(bottle["spatial_3d"]["z"])
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"bottle target_z={pred_z:.3f}m",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return display


def build_geometry_scene(frame, frame_count: int, modules: dict) -> tuple[dict, object, list]:
    detector = modules["detector"]
    segmenter = modules["segmenter"]
    depth_estimator = modules["depth_estimator"]
    depth_attacher = modules["depth_attacher"]
    spatial_converter = modules["spatial_converter"]
    stabilizer = modules["stabilizer"]
    floor_detector = modules["floor_detector"]
    relation_graph = modules["relation_graph"]
    affordance_engine = modules["affordance_engine"]

    yolo_result = detector.detect(frame)
    scene_data = detector.build_scene(yolo_result, frame, frame_count)
    annotated_frame, scene_data, masks_list = segmenter.segment_objects(frame, scene_data)
    depth_map = depth_estimator.get_depth_map(frame)
    scene_data = depth_attacher.attach_depth(scene_data, masks_list, depth_map)
    scene_data = spatial_converter.process_scene_3d(scene_data)
    scene_data = stabilizer.process_scene(scene_data)
    scene_data = floor_detector.update_scene_with_floor(scene_data, depth_map)
    scene_data = relation_graph.process_scene_relations(scene_data)
    scene_data = affordance_engine.infer_affordances(scene_data)
    return scene_data, annotated_frame, masks_list


def main() -> int:
    parser = argparse.ArgumentParser(description="bottle 3D 위치 정확도 정량평가")
    parser.add_argument(
        "--distances",
        help="쉼표로 구분한 실제 Z 거리(m). 예: 0.4,0.6,0.8,1.0,1.2",
    )
    args = parser.parse_args()

    project_root = str(PROJECT_ROOT)
    if project_root in sys.path:
        sys.path.remove(project_root)
    sys.path.insert(0, project_root)

    import cv2

    from vision.depth.depth_estimator import DepthEstimator
    from vision.detector import ObjectDetector
    from vision.reasoning.affordance_engine import AffordanceEngine
    from vision.reasoning.relation_graph import SpatialRelationGraph
    from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
    from vision.spatial.floor_detector import FloorPlaneDetector
    from vision.spatial.stabilizer import CoordinateStabilizer
    from vision.spatial.transformer import Spatial3DConverter
    from vision.stream import CAMERA_MATRIX, WebcamStream

    distances = parse_distances(args.distances)
    run_id, csv_result, txt_result = allocate_result_files()
    initialise_result_files(run_id, csv_result, txt_result, distances)

    print("[test2] bottle 3D 위치 정보 정확도 측정 시작")
    print(f"[test2] run_id: {run_id:03d}")
    print(f"[test2] CSV 저장: {csv_result}")
    print(f"[test2] TXT 저장: {txt_result}")
    print(f"[test2] 위치당 샘플 수: {TARGET_SAMPLES_PER_POSITION}")

    modules = {
        "detector": ObjectDetector(),
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

    try:
        for position_index, true_z in enumerate(distances, start=1):
            print("\n" + "=" * 72)
            print(f"[test2] 위치 {position_index}/{len(distances)} 준비")
            print(f"[test2] bottle을 실제 Z={true_z:.3f}m 위치에 놓고 최대한 정지시켜 주세요.")
            print("[test2] 카메라와 bottle 위치가 안정되면 Enter를 누르면 30개 샘플 수집을 시작합니다.")
            input("[test2] 준비 완료 후 Enter...")

            samples: list[dict] = []
            while len(samples) < TARGET_SAMPLES_PER_POSITION:
                ret, frame = stream.get_frame()
                if not ret:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    preview = draw_guidance(frame, position_index, true_z, len(samples), None)
                    cv2.imshow("test2 - bottle position accuracy", preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[test2] 사용자 입력 q로 중단합니다.")
                        return 1
                    continue

                scene_data, annotated_frame, _ = build_geometry_scene(frame, frame_count, modules)
                bottle = select_bottle(scene_data)
                preview = draw_guidance(
                    annotated_frame,
                    position_index,
                    true_z,
                    len(samples),
                    bottle,
                )
                cv2.imshow("test2 - bottle position accuracy", preview)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test2] 사용자 입력 q로 중단합니다.")
                    return 1

                if bottle is None:
                    print("[test2] bottle 미검출: 샘플로 집계하지 않습니다.")
                    continue

                samples.append(json.loads(json.dumps(bottle, ensure_ascii=False)))
                sample_number = len(samples)
                write_sample_row(
                    csv_result,
                    run_id,
                    position_index,
                    sample_number,
                    frame_count,
                    true_z,
                    bottle,
                )
                pred_z = float(bottle["spatial_3d"]["z"])
                print(
                    f"[test2] position {position_index} sample "
                    f"{sample_number}/{TARGET_SAMPLES_PER_POSITION} | "
                    f"true_z={true_z:.3f}m pred_z={pred_z:.3f}m "
                    f"abs_error={abs(pred_z - true_z):.3f}m"
                )

            summary = summarize_position(samples, true_z)
            write_summary_row(csv_result, txt_result, run_id, position_index, true_z, summary)
            print(
                f"[test2] 위치 {position_index} 완료 | "
                f"MAE={summary['z_mae_m']:.3f}m, "
                f"RMSE={summary['z_rmse_m']:.3f}m, "
                f"target_z_std={summary['target_z_std_m']:.3f}m, "
                f"mean_3d_jitter={summary['mean_3d_jitter_m']:.3f}m"
            )

    except KeyboardInterrupt:
        print("\n[test2] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print("\n[test2] 모든 위치 측정 완료")
    print(f"[test2] CSV: {csv_result}")
    print(f"[test2] TXT: {txt_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
