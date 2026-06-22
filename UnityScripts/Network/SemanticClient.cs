using System;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using Newtonsoft.Json; // Unity Package Manager에서 Newtonsoft Json 설치 필요

namespace CV_AR.Semantic
{
    public class SemanticClient : MonoBehaviour
    {
        [Header("Server Settings")]
        public string serverUrl = "ws://127.0.0.1:8765";

        private ClientWebSocket ws;
        private CancellationTokenSource cts;

        // ActionPlanner가 구독할 이벤트
        public event Action<SemanticBatchOutput> OnDataReceived;
        public event Action OnApiLimitExceeded;

        private void Start()
        {
            ConnectToServer();
        }

        private async void ConnectToServer()
        {
            ws = new ClientWebSocket();
            cts = new CancellationTokenSource();

            try
            {
                Debug.Log($"[SemanticClient] Connecting to {serverUrl}...");
                await ws.ConnectAsync(new Uri(serverUrl), cts.Token);
                Debug.Log("[SemanticClient] Connected!");

                // 백그라운드 수신 루프 실행
                _ = ReceiveLoop();
            }
            catch (Exception e)
            {
                Debug.LogError($"[SemanticClient] Connection Failed: {e.Message}");
            }
        }

        private async Task ReceiveLoop()
        {
            var buffer = new byte[8192];

            while (ws.State == WebSocketState.Open && !cts.IsCancellationRequested)
            {
                try
                {
                    WebSocketReceiveResult result = await ws.ReceiveAsync(new ArraySegment<byte>(buffer), cts.Token);

                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, string.Empty, cts.Token);
                        Debug.Log("[SemanticClient] Connection closed by server.");
                    }
                    else
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        ProcessMessage(message);
                    }
                }
                catch (Exception e)
                {
                    Debug.LogError($"[SemanticClient] Receive Error: {e.Message}");
                    break;
                }
            }
        }

        private void ProcessMessage(string jsonMessage)
        {
            try
            {
                // Newtonsoft.Json 사용을 권장합니다 (유니티 내장 JsonUtility는 복잡한 List 처리에 취약함)
                ServerMessage msg = JsonConvert.DeserializeObject<ServerMessage>(jsonMessage);

                if (msg.status == "API_LIMIT_EXCEEDED")
                {
                    Debug.LogWarning("[SemanticClient] ⚠️ API 한도 초과 (60초 대기 중...)");
                    // 메인 스레드에서 이벤트 발생 (Unity 특성 상 Queue를 타야 할 수도 있지만 최신 UniTask 등 사용 무방)
                    OnApiLimitExceeded?.Invoke();
                }
                else if (msg.status == "SUCCESS" && msg.data != null)
                {
                    Debug.Log($"[SemanticClient] 수신 성공! 객체 수: {msg.data.results.Count}");
                    OnDataReceived?.Invoke(msg.data);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[SemanticClient] JSON Parse Error: {e.Message}\nPayload: {jsonMessage}");
            }
        }

        private void OnDestroy()
        {
            if (cts != null)
            {
                cts.Cancel();
                cts.Dispose();
            }
            if (ws != null)
            {
                ws.Dispose();
            }
        }
    }
}
