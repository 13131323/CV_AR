using System.Collections.Generic;
using UnityEngine;
using CV_AR.Avatar;

namespace CV_AR.Semantic
{
    [RequireComponent(typeof(SemanticClient))]
    [RequireComponent(typeof(SpatialTracker))]
    public class ActionPlanner : MonoBehaviour
    {
        private SemanticClient client;
        private SpatialTracker spatialTracker;
        
        [Header("References")]
        public AvatarController avatarController;

        private bool isWaitingForApi = false;

        private void Awake()
        {
            client = GetComponent<SemanticClient>();
            spatialTracker = GetComponent<SpatialTracker>();
        }

        private void Start()
        {
            if (avatarController != null)
            {
                avatarController.Initialize(spatialTracker);
            }
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
        }

        private void HandleSemanticData(SemanticBatchOutput output)
        {
            isWaitingForApi = false;

            if (output.results == null || output.results.Count == 0)
            {
                Debug.Log("[ActionPlanner] 감지된 객체가 없습니다. 대기(Idle) 상태 전환.");
                return;
            }

            // 여러 객체 중 상호작용 가능한 첫 번째 객체를 찾습니다.
            SemanticObject bestTarget = null;
            string bestPolicy = "IGNORE";

            foreach (var targetObj in output.results)
            {
                string policy = targetObj.planner_directives.action_policy;

                // 1. 방어 로직 (Override)
                if (targetObj.semantic_state.social_state == "held_by_user" || targetObj.semantic_state.social_state == "in_use_by_other")
                {
                    if (policy == "APPROACH_AND_INTERACT")
                    {
                        policy = "OBSERVE_ONLY";
                    }
                }

                if (targetObj.identity.is_person)
                {
                    policy = "IGNORE";
                }

                if (policy == "APPROACH_AND_INTERACT")
                {
                    bestTarget = targetObj;
                    bestPolicy = policy;
                    break; // 상호작용 가능한 객체를 찾으면 즉시 루프 종료
                }
                else if (policy == "OBSERVE_ONLY" && bestTarget == null)
                {
                    bestTarget = targetObj;
                    bestPolicy = policy;
                }
            }

            if (bestTarget == null)
            {
                // 상호작용 가능한 객체가 없으면 첫 번째 객체를 로깅용으로 사용
                bestTarget = output.results[0];
                bestPolicy = "IGNORE";
            }

            Debug.Log($"[ActionPlanner] 타겟 객체: {bestTarget.identity.class_name} | 결정: {bestPolicy}");

            // 2. 최종 행동 실행
            switch (bestPolicy)
            {
                case "IGNORE":
                    Debug.Log("[ActionPlanner] 무시 (Ignore)");
                    break;

                case "OBSERVE_ONLY":
                    Debug.Log("[ActionPlanner] 관찰 (Observe)");
                    break;

                case "APPROACH_AND_INTERACT":
                    string action = bestTarget.planner_directives.animation_trigger;
                    Debug.Log($"[ActionPlanner] 접근 및 상호작용. ID: {bestTarget.object_id}, 액션: {action}");
                    
                    if (avatarController != null)
                    {
                        avatarController.MoveToAndInteract(bestTarget.object_id, action);
                    }
                    break;

                default:
                    break;
            }
        }
    }
}
