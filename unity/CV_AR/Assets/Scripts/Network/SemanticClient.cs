using System;
using System.Collections.Generic;
using UnityEngine;
using NativeWebSocket;

namespace CV_AR.Semantic
{
    public class SemanticClient : MonoBehaviour
    {
        private WebSocket websocket;
        
        // 이벤트 선언
        public event Action<SemanticBatchOutput> OnDataReceived;
        public event Action<List<FastStreamObject>> OnFastStreamReceived;
        public event Action OnApiLimitExceeded;

        [Header("Server Settings")]
        public string serverUrl = "ws://127.0.0.1:8765";

        [Serializable]
        private class FastStreamWrapper
        {
            public List<FastStreamObject> data;
        }
        
        [Serializable]
        private class SemanticOutputWrapper
        {
            public SemanticBatchOutput data;
        }

        async void Start()
        {
            websocket = new WebSocket(serverUrl);

            websocket.OnOpen += () => Debug.Log("[SemanticClient] 서버 접속 성공!");
            websocket.OnError += (e) => Debug.LogError($"[SemanticClient] 서버 에러: {e}");
            websocket.OnClose += (e) => Debug.Log("[SemanticClient] 접속 종료");

            websocket.OnMessage += (bytes) =>
            {
                var message = System.Text.Encoding.UTF8.GetString(bytes);
                ParseServerMessage(message);
            };

            await websocket.Connect();
        }

        void Update()
        {
            #if !UNITY_WEBGL || UNITY_EDITOR
            websocket.DispatchMessageQueue();
            #endif
        }

        private void ParseServerMessage(string jsonStr)
        {
            try
            {
                ServerMessage baseMsg = JsonUtility.FromJson<ServerMessage>(jsonStr);

                if (baseMsg.status == "SUCCESS")
                {
                    SemanticOutputWrapper wrapper = JsonUtility.FromJson<SemanticOutputWrapper>(jsonStr);
                    if (wrapper.data != null)
                        OnDataReceived?.Invoke(wrapper.data);
                }
                else if (baseMsg.status == "FAST_STREAM")
                {
                    FastStreamWrapper wrapper = JsonUtility.FromJson<FastStreamWrapper>(jsonStr);
                    if (wrapper.data != null)
                        OnFastStreamReceived?.Invoke(wrapper.data);
                }
                else if (baseMsg.status == "API_LIMIT_EXCEEDED")
                {
                    OnApiLimitExceeded?.Invoke();
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"[SemanticClient] JSON 파싱 에러: {e.Message}");
            }
        }

        private async void OnApplicationQuit()
        {
            if (websocket != null)
                await websocket.Close();
        }
    }
}
