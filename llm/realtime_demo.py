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
import cv2
from PIL import Image

from vision.stream import WebcamStream
from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
from vision.depth.depth_estimator import DepthEstimator
from vision.spatial.transformer import Spatial3DConverter
from vision.reasoning.relation_graph import SpatialRelationGraph

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

    annotated_frame, scene_data, masks_list = segmenter.segment_objects(frame, scene_data)

    depth_map = depth_estimator.get_depth_map(frame)
    scene_data = depth_attacher.attach_depth(scene_data, masks_list, depth_map)

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
    last_sent_inputs = None

    print("=" * 60)
    print("Layer 4(Geometry) -> Layer 5(Semantic Interpretation) VLM 실시간 데모")
    print("'q'를 누르면 종료됩니다.")
    print("=" * 60)

    def is_significant_change(prev_inputs, curr_inputs):
        if prev_inputs is None:
            return True
        if len(prev_inputs) != len(curr_inputs):
            return True
        for p_obj, c_obj in zip(prev_inputs, curr_inputs):
            if p_obj.detected_class != c_obj.detected_class:
                return True
            # Check mask area difference > 20%
            if abs(p_obj.mask_area - c_obj.mask_area) / max(p_obj.mask_area, 1) > 0.2:
                return True
            # Check target_z difference > 0.5m
            if abs(p_obj.target_z - c_obj.target_z) > 0.5:
                return True
        return False

    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret:
                print("[에러] 카메라 프레임을 읽어올 수 없습니다.")
                break

            frame_count += 1

            # 5프레임마다 연산
            if frame_count % 5 != 0:
                continue

            scene_data = build_scene_graph_for_frame(frame, frame_count)
            inputs = build_inputs_from_scene(scene_data)

            if not inputs:
                continue

            # 수치가 크게 변했을 때만 Gemini 호출
            if not is_significant_change(last_sent_inputs, inputs):
                continue
                
            last_sent_inputs = inputs

            # 시각적 식별을 위해 원본 프레임에 Bounding Box와 ID 그리기
            annotated_frame = frame.copy()
            for inp in inputs:
                if inp.bbox_2d:
                    x1, y1, x2, y2 = map(int, inp.bbox_2d)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Obj {inp.object_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # OpenCV BGR을 PIL RGB로 변환
            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            # 전송량 최적화를 위해 리사이즈 (가로 640 고정)
            wpercent = (640 / float(pil_image.size[0]))
            hsize = int((float(pil_image.size[1]) * float(wpercent)))
            pil_image = pil_image.resize((640, hsize), Image.Resampling.LANCZOS)

            batch_input = SemanticInterpretationBatchInput(context=DEFAULT_CONTEXT, objects=inputs)

            print(f"\n--- [FRAME {frame_count}] 수치 변화 감지! 객체 {len(inputs)}개 이미지 전송 및 일괄 추론 ---")
            t0 = time.time()
            try:
                batch_output = interpret_batch(batch_input, image=pil_image)
            except RuntimeError as e:
                print(f"[Gemini 배치 호출 실패] {e}")
                continue
            elapsed = time.time() - t0
            print(f"-> 추론 완료 ({elapsed:.2f}s)")

            for input_data, output_data in zip(inputs, batch_output.results):
                print(
                    f"  [Obj {input_data.object_id}] YOLO={input_data.detected_class:<10} -> VLM={output_data.object_identity:<12} (conf: {output_data.confidence:.2f})\n"
                    f"      배경 묘사: {output_data.visual_context}\n"
                    f"      사물 상태: {output_data.object_state}\n"
                    f"      어포던스 추론: {output_data.affordance_reasoning}\n"
                    f"      최종 상호작용: {output_data.interaction_state}\n"
                )

    finally:
        stream.release()


if __name__ == "__main__":
    main()