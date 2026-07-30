"""在 Oracle Schema 中验收 Webhook 到只观测报告闭环。"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import tempfile
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiohttp
from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aiops_agent.adapters.monitoring import MonitorProviderRegistry
from aiops_agent.adapters.monitoring.payload_store import (
    LocalMonitorPayloadStore,
)
from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.application.monitoring import MonitorWebhookIntakeService
from aiops_agent.application.runtime import AIOpsRuntimeService
from aiops_agent.config import get_aiops_settings
from aiops_agent.contracts.monitoring import (
    MetricObservation,
    MetricPoint,
    MetricSeries,
)
from aiops_agent.adapters.monitoring.catalog import load_metric_catalog
from aiops_agent.entities import (
    InboxEntity,
    MonitorSourceEntity,
    OpsAlertEntity,
    OpsArtifactEntity,
    OpsEventEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    OutboxEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetMonitorEntity,
)
from aiops_agent.orchestration import create_kernel_blueprint_registry
from aiops_agent.persistence import create_aiops_uow_factory
from aiops_agent.ports.monitor import AlertQueryResult, MetricQueryResult
from aiops_agent.workers import (
    AIOpsDomainOutboxSink,
    AIOpsOutboxDispatcher,
    AIOpsTaskWorker,
    LoggingOutboxSink,
    create_runtime_handler_registry,
)
from main_api.entities import PlatformDomainEntity
from platform_core.contracts.aiops import MonitorWebhookEnvelope
from platform_core.database.oracle import create_database_runtime
from platform_core.identity import uuid7


class FixtureMonitorAdapter:
    """实库 Smoke 只替换外部网络，不替换领域和持久化链路。"""

    async def query_metrics(self, request):
        observations = []
        for definition in request.metric_definitions:
            value = 1.0 if definition.metric_code == "db.availability" else 42.0
            series = (
                MetricSeries(
                    points=(
                        MetricPoint(
                            observed_at=request.window_end,
                            value=value,
                        ),
                    )
                ),
            )
            observations.append(
                MetricObservation(
                    metric_code=definition.metric_code,
                    semantic_version=definition.semantic_version,
                    unit=definition.unit,
                    value_kind=definition.value_kind,
                    window_start=request.window_start,
                    window_end=request.window_end,
                    requested_step_seconds=request.requested_step_seconds,
                    effective_step_seconds=request.requested_step_seconds,
                    source_id="fixture-source",
                    source_type="PROMETHEUS",
                    source_version=1,
                    target_id=request.target_id,
                    binding_id=request.binding_id,
                    external_target_fingerprint=hashlib.sha256(
                        request.external_target_key.encode()
                    ).hexdigest(),
                    series=series,
                    summary={
                        "count": 1,
                        "min": value,
                        "max": value,
                        "avg": value,
                        "p95": None,
                        "last": value,
                    },
                    expected_points=1,
                    actual_points=1,
                    coverage_ratio=1,
                    provenance={
                        "adapter_version": "smoke",
                        "provider_response_hash": "a" * 64,
                    },
                )
            )
        return MetricQueryResult(observations=tuple(observations))

    async def query_alerts(self, request):
        return AlertQueryResult()


class FixtureMonitorRegistry:
    def create(self, context):
        return FixtureMonitorAdapter()


async def main() -> None:
    settings = get_aiops_settings()
    database = create_database_runtime(settings)
    uow_factory = create_aiops_uow_factory(database.session_factory)
    target_id = uuid7()
    source_id = uuid7()
    monitor_id = uuid7()
    inbox_id = alert_id = run_id = None
    created_domain_id = None
    trace_id = str(uuid7())
    webhook_key = f"whk-{uuid7()}-{uuid7()}"
    webhook_secret = f"smoke-secret-{uuid7()}"
    os.environ["AIOPS_MONITORING_SMOKE_WEBHOOK"] = webhook_secret
    client_session = aiohttp.ClientSession()
    payload_directory = tempfile.TemporaryDirectory(
        prefix="kbot-aiops-monitor-smoke-"
    )
    try:
        async with database.session_factory() as session:
            domain = (
                await session.execute(
                    select(PlatformDomainEntity)
                    .where(
                        PlatformDomainEntity.status == "ACTIVE",
                    )
                    .order_by(PlatformDomainEntity.domain_id)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if domain is None:
                domain = PlatformDomainEntity(
                    name=f"monitor-smoke-{target_id}",
                    status="ACTIVE",
                    created_by="monitor-smoke",
                    updated_by="monitor-smoke",
                )
                session.add(domain)
                await session.flush()
                created_domain_id = int(domain.domain_id)
            domain_id = int(domain.domain_id)
            await session.commit()

        async with uow_factory() as uow:
            await uow.targets.add_target(
                TargetEntity(
                    target_id=target_id,
                    domain_id=domain_id,
                    target_key=f"monitor-smoke-{target_id}",
                    display_name="监控闭环 Smoke Target",
                    db_type="ORACLE",
                    environment="DEV",
                    db_role="PRIMARY",
                    security_level=1,
                    status="ACTIVE",
                    health_status="UNKNOWN",
                    created_by="monitor-smoke",
                    updated_by="monitor-smoke",
                )
            )
            await uow.targets.add_binding(
                TargetBindingEntity(
                    target_id=target_id,
                    agent_id=settings.runtime.system_aiops_agent_id,
                    allow_mutation=False,
                    status="ACTIVE",
                    created_by="monitor-smoke",
                    updated_by="monitor-smoke",
                )
            )
            await uow.monitor_sources.add(
                MonitorSourceEntity(
                    monitor_source_id=source_id,
                    domain_id=domain_id,
                    source_key=f"monitor-smoke-{source_id}",
                    display_name="监控闭环 Smoke Prometheus",
                    source_type="PROMETHEUS",
                    endpoint="https://prometheus.invalid",
                    webhook_secret_ref="env://AIOPS_MONITORING_SMOKE_WEBHOOK",
                    webhook_key_hash=hashlib.sha256(
                        webhook_key.encode()
                    ).hexdigest(),
                    capabilities_json={
                        "external_target_label": "instance"
                    },
                    status="ACTIVE",
                    health_status="HEALTHY",
                    created_by="monitor-smoke",
                    updated_by="monitor-smoke",
                )
            )
            await uow.targets.add_monitor(
                TargetMonitorEntity(
                    target_monitor_id=monitor_id,
                    target_id=target_id,
                    monitor_source_id=source_id,
                    external_target_key="db-smoke-1",
                    role="PRIMARY",
                    priority=10,
                    status="ACTIVE",
                    health_status="UNKNOWN",
                    created_by="monitor-smoke",
                    updated_by="monitor-smoke",
                )
            )
            await uow.commit()

        now = datetime.now(UTC).replace(microsecond=0)
        body = json.dumps(
            {
                "status": "firing",
                "alerts": [
                    {
                        "status": "firing",
                        "labels": {
                            "instance": "db-smoke-1",
                            "alertname": "DatabaseDown",
                            "severity": "critical",
                        },
                        "annotations": {"summary": "Smoke 告警"},
                        "startsAt": now.isoformat(),
                        "fingerprint": "smoke-alert",
                    }
                ],
            }
        ).encode()
        timestamp = str(int(now.timestamp()))
        signature = hmac.new(
            webhook_secret.encode(),
            timestamp.encode() + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        secret_store = ConfiguredSecretStore(
            provider="environment",
            allowed_schemes=("env", "vault", "secret-manager"),
        )
        intake = MonitorWebhookIntakeService(
            uow_factory=uow_factory,
            provider_registry=MonitorProviderRegistry(
                session=client_session
            ),
            secret_store=secret_store,
            system_agent_id=settings.runtime.system_aiops_agent_id,
            max_webhook_bytes=settings.monitoring.max_webhook_bytes,
            payload_store=LocalMonitorPayloadStore(
                Path(payload_directory.name)
            ),
        )
        envelope = MonitorWebhookEnvelope(
            request_id=trace_id,
            webhook_key_hash=hashlib.sha256(
                webhook_key.encode()
            ).hexdigest(),
            raw_body_base64=__import__("base64").b64encode(body).decode(),
            raw_body_hash=hashlib.sha256(body).hexdigest(),
            content_type="application/json",
            signature_headers={
                "x-kbot-timestamp": timestamp,
                "x-kbot-signature": f"sha256={signature}",
            },
            received_at=now,
        )
        receipt = await intake.intake(envelope)
        duplicate = await intake.intake(envelope)
        if not receipt.accepted or not duplicate.duplicate:
            raise RuntimeError("Webhook Inbox 未保持幂等")
        inbox_id = receipt.inbox_id
        alert_id = receipt.alert_ids[0]

        handlers = create_runtime_handler_registry(
            monitor_provider_registry=FixtureMonitorRegistry(),
            secret_store=secret_store,
        )
        runtime_service = AIOpsRuntimeService(
            uow_factory=uow_factory,
            blueprint_registry=create_kernel_blueprint_registry(),
            handler_registry=handlers,
            metric_catalog=load_metric_catalog(),
        )
        dispatcher = AIOpsOutboxDispatcher(
            uow_factory=uow_factory,
            sink=AIOpsDomainOutboxSink(
                runtime_service=runtime_service,
                fallback=LoggingOutboxSink(),
            ),
            dispatcher_id="monitor-smoke-dispatcher",
            lease_seconds=120,
            interval_seconds=0.1,
        )
        for _ in range(20):
            await dispatcher.run_once()
            async with database.session_factory() as session:
                run = (
                    await session.execute(
                        select(OpsRunEntity).where(
                            OpsRunEntity.trigger_alert_id == alert_id
                        )
                    )
                ).scalar_one_or_none()
            if run is not None:
                run_id = run.ops_run_id
                break
        if run_id is None:
            raise RuntimeError("Critical Alert 未可靠创建 Observe Run")

        worker = AIOpsTaskWorker(
            runtime_service=runtime_service,
            handler_registry=handlers,
            worker_id="monitor-smoke-worker",
            lease_seconds=120,
            heartbeat_seconds=30,
            poll_interval_seconds=0.1,
        )
        for _ in range(3):
            if not await worker.run_once():
                raise RuntimeError("Observe Blueprint Task 未全部执行")
        summary = await runtime_service.get_run(
            ops_run_id=run_id,
            domain_id=domain_id,
        )
        async with database.session_factory() as session:
            artifact = await session.get(
                OpsArtifactEntity, summary.final_artifact.artifact_id
            )
        if (
            summary.status != "COMPLETED"
            or summary.root_cause_grade != "INCONCLUSIVE"
            or artifact.schema_version != "OBSERVE_REPORT.v1"
            or artifact.payload_json["status"] != "READY"
        ):
            raise RuntimeError("只观测报告未按契约完成")
        print(
            "AIOps 监控闭环 Smoke 成功："
            f"inbox={inbox_id} alert={alert_id} run={run_id}"
        )
    finally:
        async with database.session_factory() as session:
            await session.execute(
                delete(OutboxEntity).where(
                    OutboxEntity.trace_id == trace_id
                )
            )
            if run_id is not None:
                await session.execute(
                    delete(OpsRunEventEntity).where(
                        OpsRunEventEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsArtifactEntity).where(
                        OpsArtifactEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsTaskEntity).where(
                        OpsTaskEntity.ops_run_id == run_id
                    )
                )
                await session.execute(
                    delete(OpsRunEntity).where(
                        OpsRunEntity.ops_run_id == run_id
                    )
                )
            if inbox_id is not None:
                await session.execute(
                    delete(OpsEventEntity).where(
                        OpsEventEntity.source_inbox_id == inbox_id
                    )
                )
            if alert_id is not None:
                await session.execute(
                    delete(OpsAlertEntity).where(
                        OpsAlertEntity.alert_id == alert_id
                    )
                )
            if inbox_id is not None:
                await session.execute(
                    delete(InboxEntity).where(
                        InboxEntity.inbox_id == inbox_id
                    )
                )
            await session.execute(
                delete(TargetMonitorEntity).where(
                    TargetMonitorEntity.target_monitor_id == monitor_id
                )
            )
            await session.execute(
                delete(MonitorSourceEntity).where(
                    MonitorSourceEntity.monitor_source_id == source_id
                )
            )
            await session.execute(
                delete(TargetBindingEntity).where(
                    TargetBindingEntity.target_id == target_id
                )
            )
            await session.execute(
                delete(TargetEntity).where(
                    TargetEntity.target_id == target_id
                )
            )
            if created_domain_id is not None:
                await session.execute(
                    delete(PlatformDomainEntity).where(
                        PlatformDomainEntity.domain_id
                        == created_domain_id
                    )
                )
            await session.commit()
        os.environ.pop("AIOPS_MONITORING_SMOKE_WEBHOOK", None)
        await client_session.close()
        await database.close()
        payload_directory.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
