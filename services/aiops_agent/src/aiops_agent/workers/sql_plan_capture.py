"""审批后的独立采集Handler；Worker只签发短Grant，不读取保护材料。"""

import asyncio
from datetime import UTC, datetime, timedelta
import hashlib
from uuid import UUID

from platform_core.contracts.aiops.sql_plan_execution import SqlPlanCaptureExecute, SqlPlanCaptureGrant, SqlPlanCaptureResult
from platform_core.identity import uuid7


class SqlPlanCaptureHandler:
    def __init__(self, *, executor_client, grant_codec, grant_issuer, grant_audience, grant_ttl_seconds, clock=None):
        self._client, self._codec = executor_client, grant_codec
        self._issuer, self._audience, self._ttl = grant_issuer, grant_audience, grant_ttl_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(self, context):
        try:
            info = context.plan_snapshot["sql_plan_capture"]
            if (context.task_key != "plan:capture" or context.attempt != 1 or context.max_attempts != 1
                or info["capture_run_id"] != str(context.run_id) or info["capture_task_id"] != str(context.task_id)
                or not context.lease_token or str(UUID(context.lease_token)) != context.lease_token):
                raise ValueError
            now = self._clock()
            deadline = min(datetime.fromisoformat(context.deadline_at), datetime.fromisoformat(context.lease_until),
                now + timedelta(seconds=min(45, self._ttl)))
            grant = SqlPlanCaptureGrant(issuer=self._issuer, audience=self._audience, grant_id=uuid7(), issued_at=now,
                expires_at=deadline, run_id=context.run_id, task_id=context.task_id,
                source_run_id=info["source_run_id"], request_id=info["request_id"], approval_sha256=info["approval_sha256"],
                lease_token_hash=hashlib.sha256(context.lease_token.encode()).hexdigest(), db_type=info["database_type"], trace_id=context.trace_id)
        except (KeyError, ValueError, TypeError, AttributeError):
            raise ValueError("计划采集任务的冻结引用或租约无效") from None
        def failed(code):
            return SqlPlanCaptureResult(capture_run_id=grant.run_id, request_id=grant.request_id,
                database_type=grant.db_type, status="FAILED", error_code=code, captured_at=self._clock(),
                oracle_parameter_peeking="NOT_PROVEN" if grant.db_type == "ORACLE" else "NOT_APPLICABLE")
        try:
            result = await self._client.execute_plan_capture(
                SqlPlanCaptureExecute(grant=self._codec.issue_plan_capture(grant)), trace_id=grant.trace_id)
            if (result.capture_run_id != grant.run_id or result.request_id != grant.request_id or result.database_type != grant.db_type):
                return failed("PLAN_CAPTURE_RESULT_MISMATCH")
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            # 不重试网络丢失；Executor可能已经消费审批并独立保存安全结果。
            return failed("PLAN_CAPTURE_EXECUTOR_UNAVAILABLE")
