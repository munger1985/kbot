"""步骤 6 确定性数据库诊断 Handler。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from uuid import UUID

from aiops_agent.contracts.artifacts import (
    DatabaseDiagnosticReport,
    DatabaseDiagnosticResult,
    DatabaseObservationAggregate,
    DatabaseScopeResult,
    EvidenceGap,
)
from aiops_agent.contracts.tool_execution import (
    DbaToolResult,
    ToolOutcome,
)
from aiops_agent.diagnostics.grants import (
    DiagnosticGrantCodec,
    canonical_sha256,
)
from aiops_agent.diagnostics.registry import database_major_version
from aiops_agent.ports.db_executor import DatabaseExecutorClientPort
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    DiagnosticExecutionGrant,
    DiagnosticLimits,
    DynamicDiagnosticExecutionGrant,
    DynamicReadDiagnosticRequest,
    OracleDynamicQueryPolicyGrant,
    ReadDiagnosticRequest,
)
from platform_core.identity import uuid7

from .errors import RetryableTaskError
from .handlers import TaskExecutionContext


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        UTC
    )


class DatabaseScopeHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> DatabaseScopeResult:
        snapshot = context.plan_snapshot["database_diagnostics"]
        return DatabaseScopeResult(
            target_id=context.target_id,
            db_type=snapshot["db_type"],
            configured_version=snapshot["configured_version"],
            catalog_hash=snapshot["catalog_hash"],
            capability_snapshot_hash=snapshot[
                "capability_snapshot_hash"
            ],
            selected_tool_count=len(snapshot["tools"]),
            initial_gaps=tuple(snapshot["initial_gaps"]),
        )


class DatabaseDiagnosticHandler:
    def __init__(
        self,
        *,
        executor_client: DatabaseExecutorClientPort,
        grant_codec: DiagnosticGrantCodec,
        grant_issuer: str,
        grant_audience: str,
        grant_ttl_seconds: int,
    ):
        self._client = executor_client
        self._codec = grant_codec
        self._issuer = grant_issuer
        self._audience = grant_audience
        self._ttl = grant_ttl_seconds

    async def execute(
        self, context: TaskExecutionContext
    ) -> DatabaseDiagnosticResult:
        tool_id = context.task_key.removeprefix("diagnostic:")
        snapshot = context.plan_snapshot["database_diagnostics"]
        if not snapshot.get("automatic_access_enabled", True):
            blocking_codes = {
                "DIAGNOSTIC_ACCESS_DENIED",
                "DIAGNOSTIC_POLICY_DENIED",
                "DIAGNOSTIC_SECRET_MISSING",
                "TARGET_ENDPOINT_MISSING",
            }
            code = next(
                (
                    str(item.get("code"))
                    for item in snapshot.get("initial_gaps", ())
                    if str(item.get("code")) in blocking_codes
                ),
                "DATABASE_ACCESS_DISABLED",
            )
            return self._gap(
                context,
                tool_id,
                code,
                retryable=False,
            )
        if tool_id != "db.instance.identity":
            identity_result = next(
                (
                    item["payload"]
                    for item in context.input_artifacts
                    if item["schema_version"]
                    == "DATABASE_DIAGNOSTIC_RESULT.v1"
                ),
                None,
            )
        tool = next(
            item for item in snapshot["tools"] if item["tool_id"] == tool_id
        )
        if tool_id != "db.instance.identity":
            actual_version = snapshot["configured_version"]
            if (
                identity_result is not None
                and identity_result.get("status") == "SUCCEEDED"
                and identity_result.get("observation")
            ):
                actual_version = identity_result["observation"]["db_version"]
            try:
                actual_major = database_major_version(actual_version)
            except ValueError:
                return self._gap(
                    context,
                    tool_id,
                    "VERSION_UNSUPPORTED",
                    retryable=False,
                )
            if not (
                int(tool["supported_version_min"])
                <= actual_major
                < int(tool["supported_version_max_exclusive"])
            ):
                return self._gap(
                    context,
                    tool_id,
                    "VERSION_UNSUPPORTED",
                    retryable=False,
                )
        now = datetime.now(UTC)
        lease_until = _utc(context.lease_until)
        expires_at = min(lease_until, now + timedelta(seconds=self._ttl))
        if expires_at <= now + timedelta(seconds=1):
            return self._finish(
                context,
                self._gap(
                    context,
                    tool_id,
                    "GRANT_EXPIRED",
                    retryable=True,
                ),
            )
        parameters = dict(tool.get("parameters", {}))
        grant = DiagnosticExecutionGrant(
            issuer=self._issuer,
            audience=self._audience,
            grant_id=uuid7(),
            issued_at=now,
            expires_at=expires_at,
            run_id=UUID(context.run_id),
            task_id=UUID(context.task_id),
            lease_token_hash=hashlib.sha256(
                context.lease_token.encode()
            ).hexdigest(),
            target_id=UUID(context.target_id),
            domain_id=int(snapshot["domain_id"]),
            target_row_version=int(snapshot["target_row_version"]),
            db_type=snapshot["db_type"],
            connection_profile=DiagnosticConnectionProfile.model_validate(
                snapshot["connection_profile"]
            ),
            diagnostic_credential_id=UUID(snapshot["diagnostic_credential_id"]),
            tool_id=tool["tool_id"],
            tool_version=tool["tool_version"],
            variant=tool["variant"],
            template_sha256=tool["template_sha256"],
            parameters_sha256=canonical_sha256(parameters),
            capability_snapshot_hash=snapshot[
                "capability_snapshot_hash"
            ],
            limits=DiagnosticLimits.model_validate(tool["limits"]),
            trace_id=context.trace_id,
        )
        request_id = uuid7()
        request = ReadDiagnosticRequest(
            executor_request_id=request_id,
            grant=self._codec.issue(grant),
            parameters=parameters,
            idempotency_key=(
                f"{context.task_id}:{context.attempt}:{tool_id}"
            ),
        )
        try:
            result = await self._client.execute_diagnostic(
                request, trace_id=context.trace_id
            )
        except Exception:
            return self._finish(
                context,
                self._gap(
                    context,
                    tool_id,
                    "TARGET_UNREACHABLE",
                    retryable=True,
                ),
            )
        if result.status == "SUCCEEDED" and result.observation is not None:
            return DatabaseDiagnosticResult(
                target_id=context.target_id,
                tool_id=tool_id,
                status="SUCCEEDED",
                observation=result.observation,
            )
        return self._finish(
            context,
            self._gap(
                context,
                tool_id,
                result.error_code or "EXECUTOR_INTERNAL_ERROR",
                retryable=result.retryable,
            ),
        )

    @staticmethod
    def _finish(
        context: TaskExecutionContext,
        result: DatabaseDiagnosticResult,
    ) -> DatabaseDiagnosticResult:
        if (
            result.gap is not None
            and result.gap.retryable
            and context.attempt < context.max_attempts
        ):
            raise RetryableTaskError(result.gap.code)
        return result

    @staticmethod
    def _gap(
        context: TaskExecutionContext,
        tool_id: str,
        code: str,
        *,
        retryable: bool,
    ) -> DatabaseDiagnosticResult:
        details = {
            "PRIVILEGE_MISSING": "Target 只读凭据缺少该诊断工具所需的对象查询权限",
            "AUTH_FAILED": "Target 只读凭据认证失败",
            "TARGET_UNREACHABLE": "Target 数据库当前无法建立只读连接",
            "TIMEOUT": "受控只读查询执行超时",
            "TARGET_CONNECTION_TIMEOUT": "Target 数据库只读连接建立超时",
            "QUERY_TIMEOUT": "受控只读查询执行超时",
            "OUTPUT_SCHEMA_INVALID": "数据库返回列与受控诊断目录不一致",
            "QUERY_INCOMPATIBLE": "受控查询与当前 Oracle 视图列定义不兼容",
            "VERSION_UNSUPPORTED": "Target 数据库版本不在该诊断工具支持范围内",
            "EXECUTOR_INTERNAL_ERROR": "受控数据库执行器未能完成本次只读查询",
        }
        return DatabaseDiagnosticResult(
            target_id=context.target_id,
            tool_id=tool_id,
            status="GAP",
            gap=EvidenceGap(
                code=code,
                tool_id=tool_id,
                detail=details.get(code, "该数据库诊断证据本次不可用"),
                retryable=retryable,
            ),
        )


class DynamicQueryInvocationHandler:
    """只消费规划时冻结的动态查询，并签发一次性执行 Grant。"""

    def __init__(
        self,
        *,
        executor_client: DatabaseExecutorClientPort,
        grant_codec: DiagnosticGrantCodec,
        grant_issuer: str,
        grant_audience: str,
        grant_ttl_seconds: int,
    ) -> None:
        self._client = executor_client
        self._codec = grant_codec
        self._issuer = grant_issuer
        self._audience = grant_audience
        self._ttl = grant_ttl_seconds

    async def execute(self, context: TaskExecutionContext) -> DbaToolResult:
        execution = context.plan_snapshot["investigation_execution"]
        invocation = dict(
            execution["dynamic_invocations"][context.task_key]
        )
        database = dict(execution["database"])
        validated = dict(invocation["validated_query"])
        tool_id = "db.oracle.readonly_query"
        if not database.get("automatic_access_enabled", True):
            return self._finish(
                context,
                self._result(
                    invocation,
                    gap=self._evidence_gap(
                        database,
                        tool_id,
                        default_code="DATABASE_ACCESS_DISABLED",
                    ),
                ),
            )
        now = datetime.now(UTC)
        expires_at = min(
            _utc(context.lease_until),
            now + timedelta(seconds=self._ttl),
        )
        if expires_at <= now + timedelta(seconds=1):
            return self._finish(
                context,
                self._result(
                    invocation,
                    gap=EvidenceGap(
                        code="GRANT_EXPIRED",
                        tool_id=tool_id,
                        detail="动态只读查询的任务租约已过期",
                        retryable=True,
                    ),
                ),
            )
        parameters = dict(validated["parameters"])
        grant = DynamicDiagnosticExecutionGrant(
            issuer=self._issuer,
            audience=self._audience,
            grant_id=uuid7(),
            issued_at=now,
            expires_at=expires_at,
            run_id=UUID(context.run_id),
            task_id=UUID(context.task_id),
            lease_token_hash=hashlib.sha256(
                context.lease_token.encode()
            ).hexdigest(),
            target_id=UUID(context.target_id),
            domain_id=int(database["domain_id"]),
            target_row_version=int(database["target_row_version"]),
            connection_profile=DiagnosticConnectionProfile.model_validate(
                database["connection_profile"]
            ),
            diagnostic_credential_id=UUID(
                database["diagnostic_credential_id"]
            ),
            query_sha256=validated["query_sha256"],
            policy_sha256=validated["policy_sha256"],
            policy_snapshot=OracleDynamicQueryPolicyGrant.model_validate(
                invocation["policy_snapshot"]
            ),
            parameters_sha256=canonical_sha256(parameters),
            projected_columns=tuple(validated["projected_columns"]),
            capability_snapshot_hash=execution[
                "capability_snapshot_hash"
            ],
            limits=DiagnosticLimits.model_validate(invocation["limits"]),
            trace_id=context.trace_id,
        )
        request = DynamicReadDiagnosticRequest(
            executor_request_id=uuid7(),
            grant=self._codec.issue_dynamic(grant),
            sql=validated["normalized_sql"],
            parameters=parameters,
            idempotency_key=(
                f"{context.task_id}:{context.attempt}:{tool_id}"
            ),
        )
        try:
            result = await self._client.execute_dynamic_diagnostic(
                request, trace_id=context.trace_id
            )
        except Exception:
            return self._finish(
                context,
                self._result(
                    invocation,
                    gap=EvidenceGap(
                        code="TARGET_UNREACHABLE",
                        tool_id=tool_id,
                        detail="动态只读查询执行器当前不可用",
                        retryable=True,
                    ),
                ),
            )
        if result.status == "SUCCEEDED" and result.observation is not None:
            return self._result(invocation, observation=result.observation)
        return self._finish(
            context,
            self._result(
                invocation,
                gap=EvidenceGap(
                    code=result.error_code or "EXECUTOR_INTERNAL_ERROR",
                    tool_id=tool_id,
                    detail="动态只读查询本次未取得可验证结果",
                    retryable=result.retryable,
                ),
            ),
        )

    @staticmethod
    def _finish(
        context: TaskExecutionContext,
        result: DbaToolResult,
    ) -> DbaToolResult:
        retryable_gap = next(
            (
                outcome.gap
                for outcome in result.tool_outcomes
                if outcome.gap is not None and outcome.gap.retryable
            ),
            None,
        )
        if (
            retryable_gap is not None
            and context.attempt < context.max_attempts
        ):
            raise RetryableTaskError(retryable_gap.code)
        return result

    @staticmethod
    def _evidence_gap(database, tool_id: str, *, default_code: str):
        blocking = next(
            (
                item
                for item in database.get("initial_gaps", ())
                if item.get("code")
                in {
                    "DIAGNOSTIC_ACCESS_DENIED",
                    "DIAGNOSTIC_POLICY_DENIED",
                    "DIAGNOSTIC_SECRET_MISSING",
                    "TARGET_ENDPOINT_MISSING",
                    "TARGET_CONNECTIVITY_UNAVAILABLE",
                }
            ),
            {},
        )
        return EvidenceGap(
            code=str(blocking.get("code") or default_code),
            tool_id=tool_id,
            detail=str(
                blocking.get("detail")
                or "Target 当前不允许自动执行动态只读查询"
            ),
            retryable=bool(blocking.get("retryable", False)),
        )

    @staticmethod
    def _result(invocation, *, observation=None, gap=None) -> DbaToolResult:
        succeeded = observation is not None
        return DbaToolResult(
            source_type="TOOL",
            source_id="db.oracle.readonly_query",
            source_version="1.0.0",
            definition_hash=invocation["validated_query"]["policy_sha256"],
            output_schema="DYNAMIC_DATABASE_OBSERVATION.v1",
            measurement_semantics=invocation["measurement_semantics"],
            presentation_kind="TABLE",
            status="SUCCEEDED" if succeeded else "FAILED",
            tool_outcomes=(
                ToolOutcome(
                    step_id=invocation["action_id"],
                    tool_id="db.oracle.readonly_query",
                    tool_version="1.0.0",
                    status="SUCCEEDED" if succeeded else "GAP",
                    observation=observation,
                    gap=gap,
                ),
            ),
        )


class DatabaseAggregateHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> DatabaseObservationAggregate:
        scope = next(
            (
                item["payload"]
                for item in context.input_artifacts
                if item["schema_version"] == "DATABASE_SCOPE_RESULT.v1"
            ),
            {},
        )
        observations = []
        gaps = [
            EvidenceGap.model_validate(item)
            for item in scope.get("initial_gaps", [])
        ]
        for item in context.input_artifacts:
            if item["schema_version"] != "DATABASE_DIAGNOSTIC_RESULT.v1":
                continue
            result = DatabaseDiagnosticResult.model_validate(item["payload"])
            if result.observation is not None:
                observations.append(result.observation)
            if result.gap is not None:
                gaps.append(result.gap)
        return DatabaseObservationAggregate(
            target_id=context.target_id,
            observations=tuple(observations),
            gaps=tuple(gaps),
        )


class DatabaseReportHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> DatabaseDiagnosticReport:
        aggregate = next(
            DatabaseObservationAggregate.model_validate(item["payload"])
            for item in context.input_artifacts
            if item["schema_version"]
            == "DATABASE_OBSERVATION_AGGREGATE.v1"
        )
        tools = tuple(
            item.tool_id for item in aggregate.observations
        )
        return DatabaseDiagnosticReport(
            target_id=context.target_id,
            status="PARTIAL" if aggregate.gaps else "READY",
            observation_count=len(aggregate.observations),
            gap_count=len(aggregate.gaps),
            tools=tools,
            gaps=aggregate.gaps,
            provenance={
                "catalog_hash": context.plan_snapshot[
                    "database_diagnostics"
                ]["catalog_hash"],
                "deterministic": True,
                "llm_used": False,
            },
        )
