using System.Collections.Generic;
using UnityEngine;
using CV_AR.Avatar;

namespace CV_AR.Semantic
{
    [RequireComponent(typeof(SemanticClient))]
    [RequireComponent(typeof(SpatialTracker))]
    public class ActionPlanner : MonoBehaviour
    {
        private const int MaxActionQueueSize = 5;

        private SemanticClient client;
        private SpatialTracker spatialTracker;

        [Header("References")]
        public AvatarController avatarController;

        private bool isWaitingForApi = false;
        private bool isActionInProgress = false;

        // 현재 실행 중인 항목도 맨 앞에 남겨 두므로 Queue.Count 자체가 전체 작업 수입니다.
        private readonly Queue<QueuedAction> actionQueue = new Queue<QueuedAction>();
        private readonly HashSet<int> queuedObjectIds = new HashSet<int>();

        private class QueuedAction
        {
            public int ObjectId;
            public string ObjectName;
            public string ActionName;
            public string AffordanceLog;
        }

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

            if (avatarController != null)
            {
                avatarController.OnActionCompleted += HandleActionCompleted;
            }
        }

        private void OnDisable()
        {
            client.OnDataReceived -= HandleSemanticData;
            client.OnApiLimitExceeded -= HandleApiLimit;

            if (avatarController != null)
            {
                avatarController.OnActionCompleted -= HandleActionCompleted;
            }
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
                Debug.Log("[ActionPlanner] 감지된 객체가 없습니다.");
                return;
            }

            int addedCount = 0;

            // VLM 결과 배열 순서를 그대로 유지하며 APPROACH_AND_INTERACT 객체만 FIFO 큐에 넣습니다.
            foreach (SemanticObject targetObj in output.results)
            {
                string policy = GetSafePolicy(targetObj);
                if (policy != "APPROACH_AND_INTERACT")
                {
                    continue;
                }

                if (actionQueue.Count >= MaxActionQueueSize)
                {
                    Debug.Log($"[ActionPlanner] 액션 큐가 가득 찼습니다. 최대 {MaxActionQueueSize}개만 유지합니다.");
                    break;
                }

                // 실행 중이거나 대기 중인 같은 객체가 반복 VLM 결과로 중복 등록되는 것을 막습니다.
                if (queuedObjectIds.Contains(targetObj.object_id))
                {
                    continue;
                }

                List<string> affordances = targetObj.semantic_state.affordances ?? new List<string>();
                string actionName = ResolveAction(targetObj, affordances);
                if (actionName == null)
                {
                    continue;
                }

                string affordanceLog = affordances.Count > 0
                    ? string.Join("/", affordances)
                    : "없음";

                actionQueue.Enqueue(new QueuedAction
                {
                    ObjectId = targetObj.object_id,
                    ObjectName = targetObj.identity.class_name,
                    ActionName = actionName,
                    AffordanceLog = affordanceLog,
                });
                queuedObjectIds.Add(targetObj.object_id);
                addedCount++;

                Debug.Log(
                    $"[ActionPlanner] 큐 등록: {targetObj.identity.class_name} " +
                    $"(ID: {targetObj.object_id}) | affordances: {affordanceLog} | " +
                    $"action: {actionName} | 큐: {actionQueue.Count}/{MaxActionQueueSize}"
                );
            }

            Debug.Log($"[ActionPlanner] 이번 VLM 결과에서 {addedCount}개 등록 | 현재 큐: {actionQueue.Count}/{MaxActionQueueSize}");
            TryStartNextAction();
        }

        private string GetSafePolicy(SemanticObject targetObj)
        {
            string policy = targetObj.planner_directives.action_policy;

            // 기존 방어 정책: 점유된 객체에는 접근하지 않습니다.
            if ((targetObj.semantic_state.social_state == "held_by_user" ||
                 targetObj.semantic_state.social_state == "in_use_by_other") &&
                policy == "APPROACH_AND_INTERACT")
            {
                policy = "OBSERVE_ONLY";
            }

            // 사람은 항상 상호작용 큐에서 제외합니다.
            if (targetObj.identity.is_person)
            {
                policy = "IGNORE";
            }

            if (!targetObj.planner_directives.is_safe_to_approach)
            {
                policy = "IGNORE";
            }

            return policy;
        }

        private string ResolveAction(SemanticObject targetObj, List<string> affordances)
        {
            string actionName = targetObj.planner_directives.animation_trigger;

            if (string.IsNullOrWhiteSpace(actionName) || actionName == "None")
            {
                Debug.LogWarning($"[ActionPlanner] ID {targetObj.object_id}: 실행할 animation_trigger가 없습니다.");
                return null;
            }

            if (affordances.Contains(actionName))
            {
                return actionName;
            }

            // VLM trigger가 목록과 다르면 Observe가 아닌 첫 번째 affordance를 우선 사용합니다.
            string fallbackAction = "Observe";
            foreach (string affordance in affordances)
            {
                if (affordance != "Observe")
                {
                    fallbackAction = affordance;
                    break;
                }
            }

            Debug.LogWarning(
                $"[ActionPlanner] ID {targetObj.object_id}: trigger '{actionName}'가 affordances에 없어 " +
                $"fallback '{fallbackAction}'을 사용합니다."
            );
            return fallbackAction;
        }

        private void TryStartNextAction()
        {
            if (isActionInProgress || actionQueue.Count == 0)
            {
                return;
            }

            if (avatarController == null)
            {
                Debug.LogError("[ActionPlanner] AvatarController가 연결되지 않아 큐를 실행할 수 없습니다.");
                return;
            }

            QueuedAction nextAction = actionQueue.Peek();
            isActionInProgress = true;

            Debug.Log(
                $"[ActionPlanner] 큐 실행: {nextAction.ObjectName} (ID: {nextAction.ObjectId}) | " +
                $"affordances: {nextAction.AffordanceLog} | action: {nextAction.ActionName}"
            );
            avatarController.MoveToAndInteract(nextAction.ObjectId, nextAction.ActionName);
        }

        private void HandleActionCompleted(int objectId)
        {
            if (actionQueue.Count == 0)
            {
                isActionInProgress = false;
                return;
            }

            QueuedAction completedAction = actionQueue.Dequeue();
            queuedObjectIds.Remove(completedAction.ObjectId);
            isActionInProgress = false;

            Debug.Log(
                $"[ActionPlanner] 큐 완료: {completedAction.ObjectName} " +
                $"(ID: {completedAction.ObjectId}) | 남은 큐: {actionQueue.Count}/{MaxActionQueueSize}"
            );

            TryStartNextAction();
        }
    }
}
