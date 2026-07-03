using System;
using UnityEngine;
using UnityEngine.AI;

namespace CV_AR.Avatar
{
    [RequireComponent(typeof(NavMeshAgent))]
    [RequireComponent(typeof(Animator))]
    public class AvatarController : MonoBehaviour
    {
        public event Action<int> OnActionCompleted;

        private NavMeshAgent agent;
        private Semantic.SpatialTracker spatialTracker;

        private int currentTargetId = -1;
        private string pendingAnimationTrigger = "";
        private bool isApproaching = false;

        private void Awake()
        {
            agent = GetComponent<NavMeshAgent>();
        }

        public void Initialize(Semantic.SpatialTracker tracker)
        {
            spatialTracker = tracker;
        }

        public void MoveToAndInteract(int objectId, string animationTrigger)
        {
            currentTargetId = objectId;
            pendingAnimationTrigger = animationTrigger;
            isApproaching = true;

            Debug.Log($"[AvatarController] 대상 객체 ID: {objectId} | 받은 animation_trigger: {animationTrigger}");
            Debug.Log($"[AvatarController] {objectId}번 객체를 향해 이동 시작. 도착 후 {animationTrigger} 액션 수행 예정.");
        }

        public void SimulateAction(int objectId, string animationTrigger)
        {
            Debug.Log($"[AvatarController] {objectId}번 객체에 대해 {animationTrigger} 액션 수행 시뮬레이션.");
            Debug.Log($"[AvatarController] 아바타가 {animationTrigger} 액션을 수행했습니다.");
        }

        public void StopAction()
        {
            isApproaching = false;
            currentTargetId = -1;
            agent.ResetPath();
            Debug.Log("[AvatarController] 아바타 액션 중지 / Idle");
        }

        private void Update()
        {
            if (!isApproaching || spatialTracker == null || currentTargetId == -1)
                return;

            // 기존 동작대로 SpatialTracker에서 객체의 최신 좌표를 받아 계속 목적지를 갱신합니다.
            Vector3 latestPos = spatialTracker.GetLatestWorldPosition(currentTargetId);

            if (latestPos != Vector3.negativeInfinity)
            {
                agent.SetDestination(latestPos);
                Debug.Log($"[Avatar 이동 중] 현재 위치: {transform.position}, 목표 위치: {latestPos}, 남은 거리: {agent.remainingDistance}");
            }
            else
            {
                Debug.LogWarning($"[Avatar 오류] 최신 좌표 추적 실패 (Target ID: {currentTargetId})");
            }

            // 목적지에 도착하면 NavMeshAgent가 완료된 경로를 제거하여 hasPath=false가 될 수 있습니다.
            // 따라서 hasPath를 필수로 요구하지 않고, 경로 계산 완료 + 도착 거리 + 정지 상태로 판정합니다.
            bool hasArrived =
                !agent.pathPending &&
                !float.IsInfinity(agent.remainingDistance) &&
                agent.remainingDistance <= agent.stoppingDistance &&
                (!agent.hasPath || agent.velocity.sqrMagnitude < 0.01f);

            if (hasArrived)
            {
                int completedObjectId = currentTargetId;
                Debug.Log($"[AvatarController] {currentTargetId}번 객체에 대해 {pendingAnimationTrigger} 액션 수행 시뮬레이션.");
                Debug.Log($"[AvatarController] 아바타가 {pendingAnimationTrigger} 액션을 수행했습니다.");

                isApproaching = false;
                currentTargetId = -1;
                OnActionCompleted?.Invoke(completedObjectId);
            }
        }
    }
}
