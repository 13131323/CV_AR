"""
실시간 관통 데모: Webcam -> Geometry Layer(scene_data) -> Semantic Interpretation Layer(Gemini)

⚠️ 통합 지점 안내
    이 스크립트는 기존 vision/ 모듈(WebcamStream, ObjectDetector, ObjectSegmenter,
    DepthEstimator, Spatial3DConverter, SceneDepthAttacher, SpatialRelationGraph)이
    실험 로그에 기록된 scene_data 스키마를 그대로 산출한다고 가정합니다.

    아래 build_scene_graph_for_frame()의 본문을, 실제 vision/ 모듈 구성에 맞춰
    한 번만 연결해 주세요. (클래스/메서드 이름이 버전마다 약간 다를 수 있어
    여기서는 표준 인터페이스만 정의하고 호출부는 비워둡니다.)

실행:
    export GEMINI_API_KEY="..."
    python -m llm.realtime_demo
"""

import time

from vision.stream import WebcamStream
from vision.detector import ObjectDetector
from vision.segmenter import ObjectSegmenter
from vision.depth import DepthEstimator
from vision.spatial import Spatial3DConverter, SceneDepthAttacher
from vision.relations import SpatialRelationGraph

from llm.feature_extractor import build_inputs_from_scene, DEFAULT_CONTEXT
from llm.interpreter import interpret_batch
from llm.schemas import SemanticInterpretationBatchInput


def build_scene_graph_for_frame(frame, frame_count: int) -> dict:
    """
    한 프레임에 대해 Geometry Layer(YOLO -> SAM -> Depth -> 3D 변환 -> 관계 그래프)를
    실행하여 scene_data를 반환한다. (실시간 측정값, 하드코딩 없음)

    실험 로그의 기존 모듈 조합을 그대로 사용합니다:
    Detector.detect -> Detector.build_scene -> Segmenter.segment_objects
    -> DepthEstimator.get_depth_map -> SceneDepthAttacher.attach_depth
    -> Spatial3DConverter.process_scene_3d -> SpatialRelationGraph.process_scene_relations
    """
    result = detector.detect(frame)
    scene_data = detector.build_scene(result, frame, frame_count)

    scene_data = segmenter.segment_objects(frame, scene_data)

    depth_map = depth_estimator.get_depth_map(frame)
    scene_data = depth_attacher.attach_depth(scene_data, scene_data["objects"], depth_map)

    scene_data = spatial_converter.process_scene_3d(scene_data)
    scene_data = relation_graph.process_scene_relations(scene_data)

    return scene_data


def main():
    global detector, segmenter, depth_estimator, depth_attacher, spatial_converter, relation_graph

    stream = WebcamStream()
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_estimator = DepthEstimator()
    depth_attacher = SceneDepthAttacher()
    spatial_converter = Spatial3DConverter()
    relation_graph = SpatialRelationGraph()

    frame_count = 0

    print("=" * 60)
    print("Layer 4(Geometry) -> Layer 5(Semantic Interpretation) 실시간 관통 데모")
    print("'q'를 누르면 종료됩니다.")
    print("=" * 60)

    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret:
                print("[에러] 카메라 프레임을 읽어올 수 없습니다.")
                break

            frame_count += 1

            # 5프레임마다 한 번씩만 LLM 호출 (실시간 부하/비용 절감)
            if frame_count % 5 != 0:
                continue

            scene_data = build_scene_graph_for_frame(frame, frame_count)
            inputs = build_inputs_from_scene(scene_data)

            if not inputs:
                continue

            # 객체 수만큼 API를 따로 호출하면 지연/Rate Limit 위험이 있으므로
            # 한 프레임의 모든 객체를 단 1회의 배치 호출로 처리한다.
            batch_input = SemanticInterpretationBatchInput(context=DEFAULT_CONTEXT, objects=inputs)

            print(f"\n--- [FRAME {frame_count}] 탐지된 객체 {len(inputs)}개 일괄 추론 ---")
            t0 = time.time()
            try:
                batch_output = interpret_batch(batch_input)
            except RuntimeError as e:
                print(f"[Gemini 배치 호출 실패] {e}")
                continue
            elapsed = time.time() - t0
            print(f"-> 추론 완료 ({elapsed:.2f}s)")

            for input_data, output_data in zip(inputs, batch_output.results):
                print(
                    f"  class={input_data.detected_class:<10} "
                    f"mask_area={input_data.mask_area:<8} "
                    f"target_z={input_data.target_z:<6} "
                    f"-> identity={output_data.object_identity:<12} "
                    f"state={output_data.object_state:<10} "
                    f"interaction={output_data.interaction_state:<16} "
                    f"conf={output_data.confidence:.2f}"
                )
                if output_data.reasoning:
                    print(f"      이유: {output_data.reasoning}")

    finally:
        stream.release()


if __name__ == "__main__":
    main()