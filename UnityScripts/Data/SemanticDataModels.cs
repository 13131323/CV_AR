using System;
using System.Collections.Generic;
using UnityEngine;

namespace CV_AR.Semantic
{
    [Serializable]
    public class IdentityModel
    {
        public string class_name;
        public bool is_person;
    }

    [Serializable]
    public class SpatialContextModel
    {
        public string camera_relative;
        public string environment_relative;
    }

    [Serializable]
    public class SemanticStateModel
    {
        public string social_state;
        public List<string> affordances;
    }

    [Serializable]
    public class PlannerDirectivesModel
    {
        public string action_policy;
        public string animation_trigger;
        public bool is_safe_to_approach;
    }

    // Python의 SemanticInterpretationOutput 와 1:1 매칭
    [Serializable]
    public class SemanticObject
    {
        public int object_id;
        public IdentityModel identity;
        public SpatialContextModel corrected_spatial_relation;
        public SemanticStateModel semantic_state;
        public PlannerDirectivesModel planner_directives;
        public string reasoning;
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
