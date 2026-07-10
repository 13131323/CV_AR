"""정량평가 test1-1: 전체 파이프라인 레이어별 처리 시간 측정.

측정 항목:
- Vision Layer 처리 시간: YOLO + SAM + Depth Anything
- Geometry Layer 처리 시간: depth attach + 3D 변환 + 안정화 + floor/relation/affordance
- Semantic Layer 처리 시간: VLM 입력 구성 + VLM 추론
- 프레임 단위 파이프라인 처리 시간: 위 세 레이어를 포함한 전체 시간

각 항목은 VLM까지 성공한 프레임 30개를 기준으로 수집한다.
결과는 실행할 때마다 test_QE 바로 아래에 test1_res_004.csv/txt, test1_res_005.csv/txt ... 형태로 새로 생성한다.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def find_project_root() -> Path:
    """파일 위치가 바뀌어도 CV-AR 프로젝트 루트를 안정적으로 찾는다."""
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "llm").is_dir() and (candidate / "vision").is_dir():
            return candidate
    raise RuntimeError("프로젝트 루트를 찾을 수 없습니다. llm/vision 폴더 위치를 확인하세요.")


PROJECT_ROOT = find_project_root()
RESULT_DIR = PROJECT_ROOT / "test_QE"
RESULT_PREFIX = "test1_res"
MIN_RUN_ID = 4
TARGET_SAMPLES = 30
PROCESS_EVERY_N_FRAMES = 5


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def allocate_result_files() -> tuple[int, Path, Path]:
    """기존 결과를 덮어쓰지 않도록 다음 run_id의 CSV/TXT 경로를 만든다."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    max_run_id = MIN_RUN_ID - 1
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
        "test1: 전체 파이프라인 레이어별 처리 시간 측정\n"
        f"run_id: {run_id:03d}\n"
        f"started_at: {now_text()}\n"
        f"target_samples: {TARGET_SAMPLES}\n"
        "metrics: vision_layer_seconds, geometry_layer_seconds, "
        "semantic_layer_seconds, frame_pipeline_seconds\n\n",
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
                "vision_layer_seconds",
                "geometry_layer_seconds",
                "semantic_layer_seconds",
                "frame_pipeline_seconds",
                "yolo_seconds",
                "sam_seconds",
                "depth_seconds",
                "depth_attach_seconds",
                "spatial_3d_seconds",
                "stabilizer_seconds",
                "floor_relation_affordance_seconds",
                "vlm_object_count",
                "vlm_json",
            ]
        )


def append_csv_row(
    csv_result: Path,
    txt_result: Path,
    run_id: int,
    sample: int,
    frame_count: int,
    timings: dict[str, float],
    object_count: int,
    vlm_result: dict,
) -> None:
    timestamp = now_text()
    compact_json = json.dumps(vlm_result, ensure_ascii=False, separators=(",", ":"))
    pretty_json = json.dumps(vlm_result, ensure_ascii=False, indent=2)
    vlm_object_count = len(vlm_result.get("results", []))

    with csv_result.open("a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(
            [
                f"{run_id:03d}",
                sample,
                timestamp,
                frame_count,
                object_count,
                f"{timings['vision_layer_seconds']:.6f}",
                f"{timings['geometry_layer_seconds']:.6f}",
                f"{timings['semantic_layer_seconds']:.6f}",
                f"{timings['frame_pipeline_seconds']:.6f}",
                f"{timings['yolo_seconds']:.6f}",
                f"{timings['sam_seconds']:.6f}",
                f"{timings['depth_seconds']:.6f}",
                f"{timings['depth_attach_seconds']:.6f}",
                f"{timings['spatial_3d_seconds']:.6f}",
                f"{timings['stabilizer_seconds']:.6f}",
                f"{timings['floor_relation_affordance_seconds']:.6f}",
                vlm_object_count,
                compact_json,
            ]
        )

    with txt_result.open("a", encoding="utf-8") as file:
        file.write(
            f"[run {run_id:03d} / sample {sample:02d}]\n"
            f"timestamp: {timestamp}\n"
            f"frame_count: {frame_count}\n"
            f"object_count: {object_count}\n"
            f"vision_layer_seconds: {timings['vision_layer_seconds']:.6f}\n"
            f"geometry_layer_seconds: {timings['geometry_layer_seconds']:.6f}\n"
            f"semantic_layer_seconds: {timings['semantic_layer_seconds']:.6f}\n"
            f"frame_pipeline_seconds: {timings['frame_pipeline_seconds']:.6f}\n"
            f"yolo_seconds: {timings['yolo_seconds']:.6f}\n"
            f"sam_seconds: {timings['sam_seconds']:.6f}\n"
            f"depth_seconds: {timings['depth_seconds']:.6f}\n"
            f"depth_attach_seconds: {timings['depth_attach_seconds']:.6f}\n"
            f"spatial_3d_seconds: {timings['spatial_3d_seconds']:.6f}\n"
            f"stabilizer_seconds: {timings['stabilizer_seconds']:.6f}\n"
            f"floor_relation_affordance_seconds: {timings['floor_relation_affordance_seconds']:.6f}\n"
            f"vlm_object_count: {vlm_object_count}\n"
            f"vlm_json:\n{pretty_json}\n\n"
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
    from llm.feature_extractor import DEFAULT_CONTEXT, build_inputs_from_scene
    from llm.schemas import SemanticInterpretationBatchInput
    from vision.depth.depth_estimator import DepthEstimator
    from vision.detector import ObjectDetector
    from vision.reasoning.affordance_engine import AffordanceEngine
    from vision.reasoning.relation_graph import SpatialRelationGraph
    from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
    from vision.spatial.floor_detector import FloorPlaneDetector
    from vision.spatial.stabilizer import CoordinateStabilizer
    from vision.spatial.transformer import Spatial3DConverter
    from vision.stream import CAMERA_MATRIX, WebcamStream

    run_id, csv_result, txt_result = allocate_result_files()
    initialise_result_files(run_id, csv_result, txt_result)

    print("[test1-1] 전체 파이프라인 레이어별 처리 시간 측정 시작")
    print(f"[test1-1] run_id: {run_id:03d}")
    print(f"[test1-1] CSV 저장: {csv_result}")
    print(f"[test1-1] TXT 저장: {txt_result}")
    print("[test1-1] VLM까지 성공한 샘플 30개 수집 후 자동 종료합니다. q를 누르면 중단합니다.")

    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_estimator = DepthEstimator()
    depth_attacher = SceneDepthAttacher()
    spatial_converter = Spatial3DConverter(camera_matrix=CAMERA_MATRIX)
    stabilizer = CoordinateStabilizer()
    floor_detector = FloorPlaneDetector()
    relation_graph = SpatialRelationGraph()
    affordance_engine = AffordanceEngine()

    stream = WebcamStream()
    frame_count = 0
    sample = 0

    try:
        while sample < TARGET_SAMPLES:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % PROCESS_EVERY_N_FRAMES != 0:
                cv2.imshow("test1-1 pipeline timing", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test1-1] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            timings: dict[str, float] = {}
            pipeline_started_at = time.perf_counter()

            vision_started_at = time.perf_counter()

            yolo_started_at = time.perf_counter()
            yolo_result = detector.detect(frame)
            scene_data = detector.build_scene(yolo_result, frame, frame_count)
            timings["yolo_seconds"] = time.perf_counter() - yolo_started_at

            sam_started_at = time.perf_counter()
            annotated_frame, scene_data, masks_list = segmenter.segment_objects(frame, scene_data)
            timings["sam_seconds"] = time.perf_counter() - sam_started_at

            depth_started_at = time.perf_counter()
            depth_map = depth_estimator.get_depth_map(frame)
            timings["depth_seconds"] = time.perf_counter() - depth_started_at

            timings["vision_layer_seconds"] = time.perf_counter() - vision_started_at

            geometry_started_at = time.perf_counter()

            depth_attach_started_at = time.perf_counter()
            scene_data = depth_attacher.attach_depth(scene_data, masks_list, depth_map)
            timings["depth_attach_seconds"] = time.perf_counter() - depth_attach_started_at

            spatial_started_at = time.perf_counter()
            scene_data = spatial_converter.process_scene_3d(scene_data)
            timings["spatial_3d_seconds"] = time.perf_counter() - spatial_started_at

            stabilizer_started_at = time.perf_counter()
            scene_data = stabilizer.process_scene(scene_data)
            timings["stabilizer_seconds"] = time.perf_counter() - stabilizer_started_at

            floor_relation_started_at = time.perf_counter()
            scene_data = floor_detector.update_scene_with_floor(scene_data, depth_map)
            scene_data = relation_graph.process_scene_relations(scene_data)
            scene_data = affordance_engine.infer_affordances(scene_data)
            timings["floor_relation_affordance_seconds"] = (
                time.perf_counter() - floor_relation_started_at
            )

            timings["geometry_layer_seconds"] = time.perf_counter() - geometry_started_at

            inputs = build_inputs_from_scene(scene_data)
            if not inputs:
                cv2.imshow("test1-1 pipeline timing", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test1-1] 사용자 입력 q로 중단합니다.")
                    return 1
                print("[test1-1] 입력 객체가 없어 샘플로 집계하지 않습니다.")
                continue

            semantic_started_at = time.perf_counter()
            vlm_frame = draw_object_boxes(annotated_frame, inputs)
            pil_image = Image.fromarray(cv2.cvtColor(vlm_frame, cv2.COLOR_BGR2RGB))
            batch_input = SemanticInterpretationBatchInput(
                context=DEFAULT_CONTEXT,
                objects=inputs,
            )

            next_sample = sample + 1
            print(
                f"[test1-1] sample {next_sample}/{TARGET_SAMPLES} "
                f"VLM 호출 (객체 {len(inputs)}개)"
            )

            try:
                batch_output = interpreter.interpret_batch(batch_input, image=pil_image)
            except Exception as exc:
                print(f"[test1-1] VLM 호출 실패, 샘플로 집계하지 않습니다: {exc}")
                cv2.imshow("test1-1 pipeline timing", vlm_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[test1-1] 사용자 입력 q로 중단합니다.")
                    return 1
                continue

            timings["semantic_layer_seconds"] = time.perf_counter() - semantic_started_at
            timings["frame_pipeline_seconds"] = time.perf_counter() - pipeline_started_at

            vlm_result = batch_output.model_dump(mode="json")
            append_csv_row(
                csv_result=csv_result,
                txt_result=txt_result,
                run_id=run_id,
                sample=next_sample,
                frame_count=frame_count,
                timings=timings,
                object_count=len(inputs),
                vlm_result=vlm_result,
            )
            sample = next_sample

            print(
                f"[test1-1] sample {sample}/{TARGET_SAMPLES} 저장 완료 | "
                f"vision={timings['vision_layer_seconds']:.3f}s, "
                f"geometry={timings['geometry_layer_seconds']:.3f}s, "
                f"semantic={timings['semantic_layer_seconds']:.3f}s, "
                f"pipeline={timings['frame_pipeline_seconds']:.3f}s"
            )

            cv2.imshow("test1-1 pipeline timing", vlm_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[test1-1] 사용자 입력 q로 중단합니다.")
                return 1

    except KeyboardInterrupt:
        print("\n[test1-1] KeyboardInterrupt로 중단합니다.")
        return 1
    finally:
        stream.release()
        cv2.destroyAllWindows()

    print("[test1-1] 30개 샘플 수집 완료")
    print(f"[test1-1] CSV: {csv_result}")
    print(f"[test1-1] TXT: {txt_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
