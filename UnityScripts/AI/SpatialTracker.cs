using System.Collections.Concurrent;
using System.Collections.Generic;
using UnityEngine;

namespace CV_AR.Semantic
{
    [RequireComponent(typeof(SemanticClient))]
    public class SpatialTracker : MonoBehaviour
    {
        private SemanticClient client;
        
        // object_id를 키로 가지는 쓰레드 안전한 딕셔너리
        private ConcurrentDictionary<int, FastStreamObject> trackedObjects = new ConcurrentDictionary<int, FastStreamObject>();

        // 월드 좌표계 변환을 위한 카메라 오프셋/스케일 파라미터 (단순 MVP용)
        [Header("Coordinate Mapping")]
        public float depthScaleMultiplier = 1.0f;
        public Vector3 worldOffset = Vector3.zero;

        private void Awake()
        {
            client = GetComponent<SemanticClient>();
        }

        private void OnEnable()
        {
            client.OnFastStreamReceived += HandleFastStream;
        }

        private void OnDisable()
        {
            client.OnFastStreamReceived -= HandleFastStream;
        }

        private void HandleFastStream(List<FastStreamObject> streamData)
        {
            foreach (var obj in streamData)
            {
                trackedObjects[obj.object_id] = obj;
            }
        }

        /// <summary>
        /// 특정 object_id의 최신 3D 월드 좌표를 반환합니다.
        /// </summary>
        public Vector3 GetLatestWorldPosition(int objectId)
        {
            if (trackedObjects.TryGetValue(objectId, out FastStreamObject data))
            {
                // MVP 레벨 매핑: 파이썬의 target_z(m)를 유니티의 Z축에 매핑
                // 추후 AR 카메라의 역행렬을 곱해 정확한 월드 좌표로 변환해야 하지만 MVP에서는 Z깊이만 활용
                
                // 임시: 중심점 x는 화면 중앙(224) 대비 오프셋으로 계산 (x, y는 화면 픽셀 기반이므로 단순 비율 매핑)
                float xPos = 0; 
                if (data.bbox_2d != null && data.bbox_2d.Count == 4)
                {
                    float centerX = (data.bbox_2d[0] + data.bbox_2d[2]) / 2f;
                    xPos = (centerX - 224f) * 0.01f; // 448x448 기준 임시 매핑
                }

                return new Vector3(xPos, 0, data.target_z * depthScaleMultiplier) + worldOffset;
            }
            
            // 추적 실패 시 원점 반환
            return Vector3.zero;
        }
    }
}
