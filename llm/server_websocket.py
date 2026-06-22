"""
유니티 연동을 위한 파이썬 WebSocket 서버
실시간 웹캠 -> 비전 파이프라인 -> Gemini (10초 쿨다운) -> WebSocket 브로드캐스트
"""

import asyncio
import websockets
import json
import time
import cv2
import threading
from PIL import Image

from vision.stream import WebcamStream
from vision.detector import ObjectDetector
from vision.segmentation.segmenter import ObjectSegmenter, SceneDepthAttacher
from vision.depth.depth_estimator import DepthEstimator
from vision.spatial.transformer import Spatial3DConverter
from vision.reasoning.relation_graph import SpatialRelationGraph

from llm.feature_extractor import build_inputs_from_scene, DEFAULT_CONTEXT
from llm.interpreter import interpret_batch
from llm.schemas import SemanticInterpretationBatchInput, SemanticInterpretationBatchOutput, SemanticInterpretationOutput

# [TEST MODE] API 한도 회피용 모의(Mock) LLM 활성화 플래그
MOCK_LLM = False

# 글로벌 변수
connected_clients = set()

# 비전 파이프라인 모듈들
detector = None
segmenter = None
depth_estimator = None
depth_attacher = None
spatial_converter = None
relation_graph = None

def init_vision_modules():
    global detector, segmenter, depth_estimator, depth_attacher, spatial_converter, relation_graph
    detector = ObjectDetector()
    segmenter = ObjectSegmenter()
    depth_estimator = DepthEstimator()
    depth_attacher = SceneDepthAttacher()
    spatial_converter = Spatial3DConverter()
    relation_graph = SpatialRelationGraph()
    print("[서버] 비전 모듈 초기화 완료.")

def build_scene_graph_for_frame(frame, frame_count: int) -> dict:
    result = detector.detect(frame)
    scene_data = detector.build_scene(result, frame, frame_count)
    annotated_frame, scene_data, masks_list = segmenter.segment_objects(frame, scene_data)
    depth_map = depth_estimator.get_depth_map(frame)
    scene_data = depth_attacher.attach_depth(scene_data, masks_list, depth_map)
    scene_data = spatial_converter.process_scene_3d(scene_data)
    scene_data = relation_graph.process_scene_relations(scene_data)
    return scene_data

def is_significant_change(prev_inputs, curr_inputs):
    if prev_inputs is None:
        return True
    if len(prev_inputs) != len(curr_inputs):
        return True
    for p_obj, c_obj in zip(prev_inputs, curr_inputs):
        if p_obj.detected_class != c_obj.detected_class:
            return True
        if abs(p_obj.mask_area - c_obj.mask_area) / max(p_obj.mask_area, 1) > 0.5:
            return True
        if abs(p_obj.target_z - c_obj.target_z) > 2.0:
            return True
    return False

def broadcast_message(msg_dict: dict):
    if not connected_clients or not ws_loop:
        return
    msg_str = json.dumps(msg_dict, ensure_ascii=False)
    for ws in list(connected_clients):
        asyncio.run_coroutine_threadsafe(ws.send(msg_str), ws_loop)

def vision_thread_func():
    init_vision_modules()
    stream = WebcamStream()
    
    frame_count = 0
    last_sent_inputs = None
    last_api_call_time = 0.0

    print("[서버] 카메라 스트림 처리를 시작합니다...")
    try:
        while True:
            ret, frame = stream.get_frame()
            if not ret:
                time.sleep(0.1)
                continue

            frame_count += 1

            if frame_count % 5 != 0:
                continue

            scene_data = build_scene_graph_for_frame(frame, frame_count)
            inputs = build_inputs_from_scene(scene_data)

            if not inputs:
                continue

            if not is_significant_change(last_sent_inputs, inputs):
                continue

            current_time = time.time()
            if current_time - last_api_call_time < 10.0:
                continue

            last_sent_inputs = inputs
            last_api_call_time = current_time

            # 시각적 피드백용 전처리
            annotated_frame = frame.copy()
            for inp in inputs:
                if inp.bbox_2d:
                    x1, y1, x2, y2 = map(int, inp.bbox_2d)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"Obj {inp.object_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            pil_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            wpercent = (448 / float(pil_image.size[0]))
            hsize = int((float(pil_image.size[1]) * float(wpercent)))
            pil_image = pil_image.resize((448, hsize), Image.Resampling.LANCZOS)

            batch_input = SemanticInterpretationBatchInput(context=DEFAULT_CONTEXT, objects=inputs)
            
            print(f"--- [서버] VLM 호출 (객체 {len(inputs)}개) ---")
            
            try:
                if MOCK_LLM:
                    # API 한도를 피하기 위해 실시간 좌표(입력)는 그대로 유지하되, LLM 응답만 강제로 조작(Mocking)
                    mock_results = []
                    for inp in inputs:
                        is_person = (inp.detected_class == "person")
                        mock_obj = SemanticInterpretationOutput(
                            visual_context="mocked context",
                            object_identity=f"가짜_사물_{inp.detected_class}",
                            object_state="on_surface",
                            affordance_reasoning="mocked reasoning",
                            interaction_state="not_interactable" if is_person else "available",
                            is_interactable=not is_person,
                            affordances=[] if is_person else ["grasp", "push"],
                            action_policy="ignore_for_avatar" if is_person else "approach_and_interact",
                            confidence=0.99
                        )
                        mock_results.append(mock_obj)
                    batch_output = SemanticInterpretationBatchOutput(results=mock_results)
                    # 모의 처리에 걸리는 시간(딜레이) 모사
                    time.sleep(1.5)
                else:
                    batch_output = interpret_batch(batch_input, image=pil_image)
                    
                # 성공 시 유니티로 전송
                broadcast_message({
                    "status": "SUCCESS",
                    "data": batch_output.model_dump()
                })
                print("-> 유니티로 실시간 좌표+Mock 데이터 전송 완료!")
            except Exception as e:
                print(f"[OpenAI GPT 호출 에러] {e}")
                # 에러 발생 시 유니티에 상태 알림 및 60초 쿨다운
                broadcast_message({
                    "status": "API_LIMIT_EXCEEDED",
                    "data": None
                })
                
            # OpenCV 윈도우에 실시간 카메라 화면 출력 (필수)
            cv2.imshow("Layer 6 - WebSocket Server (Mock Mode)", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[서버] 'q' 입력 감지. 서버 종료를 예약합니다.")
                break
                
    finally:
        stream.release()
        cv2.destroyAllWindows()

async def ws_handler(websocket):
    connected_clients.add(websocket)
    remote_ip = websocket.remote_address
    print(f"[WebSocket] 유니티 클라이언트 접속됨: {remote_ip}")
    try:
        async for message in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.remove(websocket)
        print(f"[WebSocket] 유니티 클라이언트 접속 종료: {remote_ip}")

ws_loop = None

async def run_ws_server():
    print("[서버] WebSocket 서버 시작 대기: ws://127.0.0.1:8765")
    async with websockets.serve(ws_handler, "127.0.0.1", 8765):
        print("[서버] 클라이언트 접속 대기 중...")
        await asyncio.Future()  # 무한 대기

def start_websocket_server():
    global ws_loop
    ws_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(ws_loop)
    ws_loop.run_until_complete(run_ws_server())

if __name__ == "__main__":
    # 1. 백그라운드 쓰레드에서 WebSocket 서버 실행 (비동기)
    ws_thread = threading.Thread(target=start_websocket_server, daemon=True)
    ws_thread.start()

    # 2. 메인 쓰레드에서 비전 파이프라인(카메라 화면 띄우기) 실행
    # (macOS에서는 cv2.imshow를 반드시 메인 쓰레드에서 호출해야 함)
    vision_thread_func()
