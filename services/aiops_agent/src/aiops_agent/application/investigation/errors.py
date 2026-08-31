"""调查规划的安全错误映射。"""

from __future__ import annotations

import re
import traceback

from aiops_agent.application.errors import is_schema_or_integrity_error


class TurnPlanningStageError(RuntimeError):
    """保留规划内部故障的安全阶段信息，不携带用户输入或凭据。"""

    code = "AIOPS_INVESTIGATION_PLAN_INTERNAL_ERROR"

    _STAGE_BY_FUNCTION = {
        "_prepare": "PREPARE_CONTEXT",
        "_persist_raw_input": "PERSIST_RAW_INPUT",
        "_persist_input_extractions": "PERSIST_INPUT_EXTRACTIONS",
        "plan": "MODEL_PLANNING",
        "prepare_dynamic_queries": "VALIDATE_DYNAMIC_SQL",
        "prepare_source_queries": "VALIDATE_MONITORING_QUERY",
        "build_playbook_plan": "BUILD_PLAYBOOK_PLAN",
        "_prepare_monitoring": "PREPARE_MONITORING",
        "compile": "COMPILE_TASK_PLAN",
        "build": "BUILD_EXECUTION_SNAPSHOT",
        "_persist": "PERSIST_INVESTIGATION_PLAN",
    }

    def __init__(self, cause: Exception) -> None:
        frames = traceback.extract_tb(cause.__traceback__)
        frame = frames[-1] if frames else None
        stage = "PROCESS_INVESTIGATION_PLAN"
        for candidate in reversed(frames):
            mapped = self._STAGE_BY_FUNCTION.get(candidate.name)
            if mapped is not None:
                stage = mapped
                break
        location = (
            f"{frame.name}:{frame.lineno}" if frame is not None else "unknown"
        )
        detail = self._safe_detail(cause)
        super().__init__(
            f"stage={stage}; cause={type(cause).__name__}; "
            f"detail={detail}; location={location}"
        )
        self.retryable = not is_schema_or_integrity_error(cause)
        if not self.retryable:
            self.code = "AIOPS_SCHEMA_INTEGRITY_ERROR"
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.safe_detail = detail
        self.location = location

    @staticmethod
    def _safe_detail(cause: Exception) -> str:
        if is_schema_or_integrity_error(cause):
            return "database-contract-violation"
        if not isinstance(cause, KeyError) or not cause.args:
            return "not-recorded"
        key = str(cause.args[0])
        if re.fullmatch(r"[A-Za-z0-9_.:@/-]{1,120}", key):
            return f"missing-key:{key}"
        return "missing-key:<non-identifier>"
