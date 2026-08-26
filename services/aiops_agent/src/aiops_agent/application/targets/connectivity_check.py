"""Target 数据库连通性检查用例。"""

from __future__ import annotations

from uuid import UUID

from aiops_agent.application.configuration.connection_test import (
    test_target_connection,
)
from aiops_agent.application.managed_credentials import (
    AIOpsManagedCredentialError,
)
from platform_core.contracts.aiops import TargetConnectionTest


class TargetConnectivityCheckService:
    """消费持久化检查请求，并以版本条件防止旧结果覆盖新配置。"""

    def __init__(self, *, uow_factory, managed_credentials):
        self._uow_factory = uow_factory
        self._managed_credentials = managed_credentials

    async def execute(self, payload: dict) -> None:
        target_id = UUID(payload["aggregate_id"])
        details = payload["details"]
        request_id = UUID(details["connectivity_check_request_id"])
        async with self._uow_factory() as uow:
            target = await uow.targets.get_scoped(
                target_id=target_id,
                domain_id=int(payload["domain_id"]),
            )
            if (
                target is None
                or target.connectivity_check_request_id != request_id
            ):
                return
            snapshot = {
                "domain_id": int(target.domain_id),
                "db_type": target.db_type,
                "endpoint": dict(target.endpoint_json or {}),
                "credential_id": target.diagnostic_credential_id,
                "config_version": int(target.row_version),
                "connectivity_version": int(target.connectivity_version),
            }

        result = None
        error_code = None
        try:
            if not snapshot["endpoint"] or snapshot["credential_id"] is None:
                error_code = "TARGET_CONFIGURATION_INCOMPLETE"
            else:
                credential = await self._managed_credentials.read(
                    domain_id=snapshot["domain_id"],
                    credential_id=snapshot["credential_id"],
                    credential_kind="target_diagnostic",
                    external_key=target_id,
                )
                result = await test_target_connection(
                    TargetConnectionTest.model_validate(
                        {
                            "db_type": snapshot["db_type"],
                            "endpoint": snapshot["endpoint"],
                            "diagnostic_credential": credential,
                        }
                    )
                )
                error_code = result.error_code
        except (AIOpsManagedCredentialError, ValueError):
            error_code = "TARGET_CONFIGURATION_INVALID"
        except Exception:
            error_code = "TARGET_UNREACHABLE"

        connected = result is not None and result.ok
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            changed = await uow.targets.update_connectivity(
                target_id=target_id,
                connectivity_check_request_id=request_id,
                expected_config_version=snapshot["config_version"],
                expected_connectivity_version=snapshot[
                    "connectivity_version"
                ],
                connectivity_status=(
                    "CONNECTED" if connected else "UNREACHABLE"
                ),
                checked_at=now,
                last_error_code=error_code,
            )
            if changed:
                await uow.commit()
