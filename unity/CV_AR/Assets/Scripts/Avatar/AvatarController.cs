using UnityEngine;
using UnityEngine.AI;

namespace CV_AR.Avatar
{
    [RequireComponent(typeof(NavMeshAgent))]
    [RequireComponent(typeof(Animator))]
    public class AvatarController : MonoBehaviour
    {
        private NavMeshAgent agent;
        private Animator animator;

        private Semantic.SpatialTracker spatialTracker;
        
        private int currentTargetId = -1;
        private string pendingAnimationTrigger = "";
        private bool isApproaching = false;

        private void Awake()
        {
            agent = GetComponent<NavMeshAgent>();
            animator = GetComponent<Animator>();
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
            
            Debug.Log($"[AvatarController] {objectId}번 객체를 향해 이동 시작. 도착 시 {animationTrigger} 실행 예정.");
        }

        public void StopAction()
        {
            isApproaching = false;
            currentTargetId = -1;
            agent.ResetPath();
            animator.SetTrigger("idle");
        }

        private void Update()
        {
            if (!isApproaching || spatialTracker == null || currentTargetId == -1)
                return;

            // 매 프레임마다 SpatialTracker에서 최신 좌표를 가져와 덮어씀
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

            // 도착 판정 (경로가 있고, 도착 거리 이내로 들어왔을 때만)
            if (agent.hasPath && !agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
            {
                Debug.Log($"[AvatarController] 목적지 도달! (남은 거리: {agent.remainingDistance}) 애니메이션 실행: {pendingAnimationTrigger}");
                animator.SetTrigger(pendingAnimationTrigger);
                
                isApproaching = false;
                currentTargetId = -1;
            }
        }
    }
}
