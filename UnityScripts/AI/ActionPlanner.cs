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

            if (output.results == null || output.results.Count == 0)
            {
                Debug.Log("[ActionPlanner] 감지된 객체가 없습니다. 대기(Idle) 상태 전환.");
                return;
            }

            // 시연용: 첫 번째 객체에 대해서만 행동 결정
            SemanticObject targetObj = output.results[0];
            Debug.Log($"[ActionPlanner] 타겟 객체: {targetObj.identity.class_name} | 소셜 상태: {targetObj.semantic_state.social_state}");

            // ----------------------------------------------------
            // 1. 유니티 클라이언트 측 방어 로직 (Override GPT Policy)
            // ----------------------------------------------------
            string finalPolicy = targetObj.planner_directives.action_policy;

            // GPT가 혹시라도 사람이 들고 있는 물건(held_by_user)에 접근하라고 권고했을 경우 강제로 무시
            if (targetObj.semantic_state.social_state == "held_by_user" || targetObj.semantic_state.social_state == "in_use_by_other")
            {
                if (finalPolicy == "APPROACH_AND_INTERACT")
                {
                    Debug.LogWarning("[ActionPlanner 안전장치 작동] GPT가 사용 중인 객체에 접근을 권고했으나, 이를 무시(OBSERVE_ONLY)로 강제 변환합니다.");
                    finalPolicy = "OBSERVE_ONLY";
                }
            }

            // 사람 자체에 접근하려는 시도 원천 차단
            if (targetObj.identity.is_person)
            {
                finalPolicy = "IGNORE";
            }

            // ----------------------------------------------------
            // 2. 최종 행동 실행 (Action Execution)
            // ----------------------------------------------------
            switch (finalPolicy)
            {
                case "IGNORE":
                    Debug.Log("[ActionPlanner] 결정: 무시 (Ignore). 현재 행동 계속 수행.");
                    break;

                case "OBSERVE_ONLY":
                    Debug.Log("[ActionPlanner] 결정: 관찰 (Observe). 아바타의 시선을 객체로 향함.");
                    // TODO: Head IK 또는 LookAt 로직 호출 (targetObj.object_id 를 기반으로 Geometry 최신 좌표 추적)
                    break;

                case "APPROACH_AND_INTERACT":
                    string action = targetObj.planner_directives.animation_trigger;
                    Debug.Log($"[ActionPlanner] 결정: 접근 및 상호작용 (Approach & Interact). 액션: {action}");
                    
                    // TODO: targetObj.object_id 를 키값으로 사용하여, 파이썬 Geometry Layer에서 초당 30번 쏘아주는 
                    //       최신 3D 좌표(target_z)를 실시간으로 받아와 NavMeshAgent의 목적지로 갱신해야 함!
                    //       (GPT가 응답하는 데 걸린 1.5초 전의 옛날 좌표를 믿으면 안 됨)
                    break;

                default:
                    Debug.Log($"[ActionPlanner] 알 수 없는 Policy: {finalPolicy}");
                    break;
            }
        }
    }
}
