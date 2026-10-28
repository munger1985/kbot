"""本平台原始证据复用的隔离字段；无运行时 Schema 探测。"""

RUN_EVIDENCE_SCOPE_FIELDS = ("domain_id", "target_id", "agent_id", "agent_version_id", "actor_id")
TURN_EVIDENCE_SCOPE_FIELDS = ("domain_id", "conversation_id", "created_by")
