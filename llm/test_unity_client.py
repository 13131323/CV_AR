"""
유니티의 ActionPlanner.cs를 파이썬 터미널에서 완벽하게 모사하는 테스트 클라이언트
"""
import asyncio
import websockets
import json

async def receive_loop():
    uri = "ws://127.0.0.1:8765"
    print(f"[Virtual Client] 서버({uri})에 접속을 시도합니다...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("[Virtual Client] 접속 성공! 유니티의 ActionPlanner 로직 가동 준비 완료.\n")
            
            while True:
                response = await websocket.recv()
                msg = json.loads(response)
                
                print("="*60)
                if msg.get("status") == "API_LIMIT_EXCEEDED":
                    print("[ActionPlanner] 아바타 상태: 대기 모드 (API 쿨다운)")
                    print("[ActionPlanner] (애니메이션: 머리 긁적이기 / 생각 중...)")
                elif msg.get("status") == "SUCCESS":
                    data = msg.get("data", {})
                    results = data.get("results", [])
                    
                    if not results:
                        print("[ActionPlanner] 감지된 객체가 없습니다. 대기(Idle) 상태 전환.")
                        continue
                        
                    print(f"[ActionPlanner] 총 {len(results)}개의 객체 데이터 수신 완료!")
                    
                    # 시연용: 첫 번째 객체에 대해서만 행동 결정
                    target_obj = results[0]
                    
                    identity = target_obj.get("identity", {})
                    semantic_state = target_obj.get("semantic_state", {})
                    planner = target_obj.get("planner_directives", {})
                    
                    obj_identity = identity.get("class_name", "Unknown")
                    is_person = identity.get("is_person", False)
                    social_state = semantic_state.get("social_state", "Unknown")
                    
                    final_policy = planner.get("action_policy", "Unknown")
                    animation_trigger = planner.get("animation_trigger", "")
                    
                    print(f" -> 타겟 객체: {obj_identity} | 소셜 상태: {social_state}")
                    
                    # 1. 유니티 클라이언트 측 이중 방어 로직 (Override GPT Policy)
                    if social_state in ["held_by_user", "in_use_by_other"]:
                        if final_policy == "APPROACH_AND_INTERACT":
                            print(" -> [안전장치 작동] 사용 중인 객체 접근 권고를 무시(OBSERVE_ONLY)로 강제 변환합니다.")
                            final_policy = "OBSERVE_ONLY"
                            
                    if is_person:
                        final_policy = "IGNORE"
                    
                    # 2. 최종 행동 실행
                    if final_policy == "IGNORE":
                        print(" -> 결정: 무시 (Ignore). 현재 행동 계속 수행.")
                    elif final_policy == "OBSERVE_ONLY":
                        print(" -> 결정: 관찰 (Observe). 아바타의 시선을 객체로 향함 (LookAt 활성화).")
                    elif final_policy == "APPROACH_AND_INTERACT":
                        print(f" -> 결정: 접근 및 상호작용 (Approach & Interact).")
                        print(f" -> (NavMeshAgent를 통해 타겟의 실시간 최신 3D 좌표로 이동 시작!)")
                        print(f" -> (도달 시 애니메이션 재생 트리거: {animation_trigger})")
                    else:
                        print(f" -> 알 수 없는 Policy: {final_policy}")
                        
                    print("="*60 + "\n")
                
                elif msg.get("status") == "FAST_STREAM":
                    # 너무 많이 찍히면 보기 힘드므로 1줄로 작게 표시
                    data = msg.get("data", [])
                    print(f"   [SpatialTracker] 실시간 좌표 수신: {len(data)}개 객체 추적 중...", end="\r")
                
    except ConnectionRefusedError:
        print("[Virtual Client] 에러: 서버를 찾을 수 없습니다. python -m llm.server_websocket 을 먼저 실행해주세요.")
    except websockets.exceptions.ConnectionClosed:
        print("[Virtual Client] 서버와의 연결이 끊어졌습니다.")

if __name__ == "__main__":
    asyncio.run(receive_loop())
