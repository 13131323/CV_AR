using System;
using System.Collections.Generic;
using UnityEngine;

namespace CV_AR.Semantic
{
    // Python의 SemanticInterpretationOutput 와 1:1 매칭
    [Serializable]
    public class SemanticObject
    {
        public string visual_context;
        public string object_identity;
        public string object_state;
        public string affordance_reasoning;
        public string interaction_state;
        public bool is_interactable;
        public List<string> affordances;
        public string action_policy;
        public float confidence;
    }

    // Python의 SemanticInterpretationBatchOutput 와 1:1 매칭
    [Serializable]
    public class SemanticBatchOutput
    {
        public List<SemanticObject> results;
    }

    // WebSocket 서버로부터 수신하는 전체 페이로드 스키마
    [Serializable]
    public class ServerMessage
    {
        public string status; // "SUCCESS" 또는 "API_LIMIT_EXCEEDED"
        public SemanticBatchOutput data;
    }
}
