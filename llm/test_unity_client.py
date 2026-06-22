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
                    obj_identity = target_obj.get("object_identity", "Unknown")
                    action_policy = target_obj.get("action_policy", "Unknown")
                    affordances = target_obj.get("affordances", [])
                    
                    print(f" -> 타겟 객체: {obj_identity}")
                    
                    if action_policy == "ignore_for_avatar":
                        print(" -> 결정: 무시 (Ignore). 현재 행동 계속 수행.")
                    elif action_policy == "observe_only":
                        print(" -> 결정: 관찰 (Observe). 아바타의 시선을 객체로 향함 (LookAt 활성화).")
                    elif action_policy == "approach_and_interact":
                        action = affordances[0] if affordances else "조사하기"
                        print(f" -> 결정: 접근 및 상호작용 (Approach & Interact).")
                        print(f" -> (NavMeshAgent를 통해 타겟의 실시간 target_z 좌표로 이동 시작!)")
                        print(f" -> (도달 시 애니메이션 재생 트리거: {action})")
                    else:
                        print(f" -> 알 수 없는 Policy: {action_policy}")
                        
                print("="*60 + "\n")
                
    except ConnectionRefusedError:
        print("[Virtual Client] 에러: 서버를 찾을 수 없습니다. python -m llm.server_websocket 을 먼저 실행해주세요.")
    except websockets.exceptions.ConnectionClosed:
        print("[Virtual Client] 서버와의 연결이 끊어졌습니다.")

if __name__ == "__main__":
    asyncio.run(receive_loop())
