"""审批后受限变更的 Claim、单次执行闸门与终态回调。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime

from aiops_agent.actions import (
    ActionRegistry,
    ActionRenderer,
    MutationGrantCodec,
    MutationGrantError,
)
from aiops_agent.adapters.aiops_execution_client import AIOpsExecutionClient
from aiops_agent.executor.drivers import (
    MutationDatabaseDriver,
    MutationDriverError,
)
from aiops_agent.ports.secret_store import SecretStorePort
from platform_core.contracts.aiops.executor import (
    DiagnosticConnectionProfile,
    ExecutionResultRef,
    ExecutionStatusEvent,
    MutationClaimRequest,
    MutationExecutionRequest,
)
from platform_core.identity import uuid7


def _hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class MutationExecutionError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class MutationExecutorService:
    """不接受 SQL，只执行签名 Grant 锁定的本地 Catalog Action。"""

    def __init__(
        self,
        *,
        enabled: bool,
        executor_instance_id: str,
        registry: ActionRegistry,
        grant_codec: MutationGrantCodec,
        secret_store: SecretStorePort,
        control_plane: AIOpsExecutionClient,
        drivers: tuple[MutationDatabaseDriver, ...],
        concurrency: int,
    ):
        self._enabled = enabled
        self._instance_id = executor_instance_id
        self._registry = registry
        self._grant_codec = grant_codec
        self._secret_store = secret_store
        self._control_plane = control_plane
        self._drivers = {driver.db_type: driver for driver in drivers}
        self._renderer = ActionRenderer()
        self._semaphore = asyncio.Semaphore(concurrency)

    async def execute(
        self,
        request: MutationExecutionRequest,
        *,
        trace_id: str,
    ) -> ExecutionResultRef:
        if not self._enabled:
            raise MutationExecutionError("MUTATION_DISABLED")
        try:
            receipt = await self._control_plane.claim_execution(
                request.execution_id,
                MutationClaimRequest(
                    executor_request_id=request.executor_request_id,
                    executor_instance_id=self._instance_id,
                    action_catalog_hash=self._registry.catalog_hash,
                ),
                trace_id=trace_id,
            )
            grant = self._grant_codec.verify(receipt.grant)
            action = self._resolve_action(grant, request)
            profile = DiagnosticConnectionProfile.model_validate(
                grant.connection_profile
            )
            driver = self._drivers.get(grant.db_type)
            if driver is None:
                raise MutationExecutionError("DRIVER_UNAVAILABLE")
            secret = await self._secret_store.resolve(
                grant.execution_secret_ref
            )
            grant_jti_hash = _hash(str(grant.grant_id))
            await self._publish_running(
                request=request,
                grant_jti_hash=grant_jti_hash,
                trace_id=trace_id,
            )
            try:
                async with self._semaphore:
                    result = await driver.execute_action(
                        profile=profile,
                        secret=secret,
                        action=action,
                        trace_id=grant.trace_id,
                    )
                status = "SUCCEEDED"
                error_code = None
                bounded_result = result.bounded_result
            except MutationDriverError as exc:
                status = (
                    "UNKNOWN" if exc.outcome_unknown else "FAILED"
                )
                error_code = exc.code
                bounded_result = {
                    "accepted": False,
                    "action_template_id": action.action_template_id,
                    "outcome_unknown": exc.outcome_unknown,
                }
            result_hash = _hash(bounded_result)
            await self._publish_terminal(
                request=request,
                grant_jti_hash=grant_jti_hash,
                status=status,
                bounded_result=bounded_result,
                result_hash=result_hash,
                error_code=error_code,
                trace_id=trace_id,
            )
            return ExecutionResultRef(
                executor_request_id=request.executor_request_id,
                status=status,
                result_hash=result_hash,
            )
        except MutationGrantError:
            raise
        except MutationExecutionError:
            raise
        except (LookupError, ValueError) as exc:
            raise MutationExecutionError(
                "EXECUTION_GRANT_MISMATCH"
            ) from exc

    def _resolve_action(self, grant, request):
        if (
            grant.execution_id != request.execution_id
            or grant.executor_request_id != request.executor_request_id
            or grant.executor_instance_id != self._instance_id
            or grant.action_catalog_hash != self._registry.catalog_hash
            or grant.max_database_attempts != 1
            or request.idempotency_key
            != f"execution:{request.execution_id}:dispatch"
        ):
            raise MutationExecutionError("EXECUTION_GRANT_MISMATCH")
        template = self._registry.resolve_exact(
            action_template_id=grant.action_template_id,
            version=grant.action_template_version,
            db_type=grant.db_type,
            variant=grant.action_template_variant,
            template_hash=grant.action_template_hash,
        )
        action = self._renderer.render(
            template, dict(grant.typed_parameters)
        )
        if (
            action.execution_capability
            != "EXECUTABLE_AFTER_APPROVAL"
            or action.parameters_hash != grant.parameters_hash
            or action.command_hash != grant.command_hash
            or action.renderer_version != grant.renderer_version
            or action.statement_timeout_seconds
            > grant.statement_timeout_seconds
        ):
            raise MutationExecutionError("EXECUTION_GRANT_MISMATCH")
        return action

    async def _publish_running(
        self,
        *,
        request,
        grant_jti_hash,
        trace_id,
    ) -> None:
        receipt = await self._control_plane.publish_event(
            ExecutionStatusEvent(
                event_id=uuid7(),
                executor_request_id=request.executor_request_id,
                execution_id=request.execution_id,
                executor_instance_id=self._instance_id,
                grant_jti_hash=grant_jti_hash,
                status_version=3,
                status="RUNNING",
                occurred_at=datetime.now(UTC),
            ),
            trace_id=trace_id,
        )
        if not receipt.accepted:
            raise MutationExecutionError("RUNNING_EVENT_REJECTED")

    async def _publish_terminal(
        self,
        *,
        request,
        grant_jti_hash,
        status,
        bounded_result,
        result_hash,
        error_code,
        trace_id,
    ) -> None:
        receipt = await self._control_plane.publish_event(
            ExecutionStatusEvent(
                event_id=uuid7(),
                executor_request_id=request.executor_request_id,
                execution_id=request.execution_id,
                executor_instance_id=self._instance_id,
                grant_jti_hash=grant_jti_hash,
                status_version=4,
                status=status,
                occurred_at=datetime.now(UTC),
                bounded_result=bounded_result,
                result_hash=result_hash,
                error_code=error_code,
            ),
            trace_id=trace_id,
        )
        if not receipt.accepted:
            raise MutationExecutionError("TERMINAL_EVENT_REJECTED")
