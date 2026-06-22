using System.Collections.Generic;
using UnityEngine;

namespace CV_AR.Semantic
{
    [RequireComponent(typeof(SemanticClient))]
    public class ActionPlanner : MonoBehaviour
    {
        private SemanticClient client;

        // 임시 아바타 상태 (실제로는 Animator나 NavMeshAgent에 연결)
        private bool isWaitingForApi = false;

        private void Awake()
        {
            client = GetComponent<SemanticClient>();
        }

        private void OnEnable()
        {
            client.OnDataReceived += HandleSemanticData;
            client.OnApiLimitExceeded += HandleApiLimit;
        }

        private void OnDisable()
        {
            client.OnDataReceived -= HandleSemanticData;
            client.OnApiLimitExceeded -= HandleApiLimit;
        }

        private void HandleApiLimit()
        {
            isWaitingForApi = true;
            Debug.Log("[ActionPlanner] 아바타 상태: 대기 모드 (API 쿨다운)");
            // TODO: 아바타에게 "머리 긁적이기" 또는 "생각 중" 애니메이션 재생
        }

        private void HandleSemanticData(SemanticBatchOutput output)
        {
            isWaitingForApi = false;

            if (output.results.Count == 0)
            {
                Debug.Log("[ActionPlanner] 감지된 객체가 없습니다. 대기(Idle) 상태 전환.");
                return;
            }

            // 시연용: 첫 번째 객체에 대해서만 행동 결정
            SemanticObject targetObj = output.results[0];
            Debug.Log($"[ActionPlanner] 타겟 객체: {targetObj.object_identity} | 상태: {targetObj.object_state}");

            switch (targetObj.action_policy)
            {
                case "ignore_for_avatar":
                    // 사람이나 상호작용 불가능한 물체
                    Debug.Log("[ActionPlanner] 결정: 무시 (Ignore). 현재 행동 계속 수행.");
                    break;

                case "observe_only":
                    // 스마트폰 등 사용 중인 물체
                    Debug.Log("[ActionPlanner] 결정: 관찰 (Observe). 아바타의 시선을 객체로 향함.");
                    // TODO: Head IK 또는 LookAt 로직 호출
                    break;

                case "approach_and_interact":
                    // 컵, 책 등
                    string action = (targetObj.affordances != null && targetObj.affordances.Count > 0) 
                                    ? targetObj.affordances[0] 
                                    : "조사하기";
                    
                    Debug.Log($"[ActionPlanner] 결정: 접근 및 상호작용 (Approach & Interact). 액션: {action}");
                    // TODO: NavMeshAgent를 통해 객체의 target_z 좌표로 이동
                    // TODO: 도달 시 action(예: grasp)에 맞는 애니메이션 재생
                    break;

                default:
                    Debug.Log($"[ActionPlanner] 알 수 없는 Policy: {targetObj.action_policy}");
                    break;
            }
        }
    }
}
