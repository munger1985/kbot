"""Webhook 验签、去重、Target 映射与 Alert 关联。"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from typing import Any

from sqlalchemy.exc import IntegrityError

from aiops_agent.adapters.monitoring.base import MonitorAdapterError
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    dependency_unavailable,
    validation_failed,
)
from aiops_agent.application.runtime.service import (
    canonical_bytes,
    sha256_json,
)
from aiops_agent.contracts.monitoring import NormalizedMonitorEvent
from aiops_agent.entities import (
    InboxEntity,
    OpsAlertEntity,
    OpsEventEntity,
    OutboxEntity,
)
from aiops_agent.ports.monitor import (
    MonitorProviderContext,
    RawWebhookRequest,
)
from platform_core.contracts.aiops import (
    MonitorWebhookEnvelope,
    MonitorWebhookReceipt,
)
from platform_core.identity import uuid7


_SEVERITY_RANK = {
    "INFO": 0,
    "WARNING": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


class MonitorWebhookIntakeService:
    def __init__(
        self,
        *,
        uow_factory,
        provider_registry,
        secret_store,
        system_agent_id,
        max_webhook_bytes: int,
        payload_store,
    ):
        self._uow_factory = uow_factory
        self._providers = provider_registry
        self._secrets = secret_store
        self._system_agent_id = system_agent_id
        self._max_webhook_bytes = max_webhook_bytes
        self._payload_store = payload_store

    async def intake(
        self, envelope: MonitorWebhookEnvelope
    ) -> MonitorWebhookReceipt:
        body = self._decode_body(envelope)
        body_hash = hashlib.sha256(body).hexdigest()
        if body_hash != envelope.raw_body_hash:
            raise validation_failed("Webhook 正文 Hash 不匹配")

        source_snapshot = await self._load_source(
            envelope.webhook_key_hash
        )
        webhook_ref = source_snapshot["webhook_secret_ref"]
        if not webhook_ref:
            raise AIOpsApplicationError(
                code="MONITOR_AUTH_FAILED",
                message="监控源未配置 Webhook 验签凭据",
                status_code=401,
            )
        try:
            secret = await self._secrets.resolve(webhook_ref)
        except AIOpsApplicationError as exc:
            raise dependency_unavailable("Webhook Secret 暂时不可用") from exc
        credentials = dict(secret.values)
        if "value" in credentials:
            credentials["webhook_secret"] = credentials.pop("value")
        adapter = self._providers.create(
            MonitorProviderContext(
                source_id=source_snapshot["source_id"],
                source_type=source_snapshot["source_type"],
                source_version=source_snapshot["source_version"],
                endpoint=source_snapshot["endpoint"],
                credentials=credentials,
                capabilities=source_snapshot["capabilities"],
            )
        )
        try:
            batch = await adapter.verify_and_parse_webhook(
                RawWebhookRequest(
                    headers={
                        key.lower(): value
                        for key, value in envelope.signature_headers.items()
                    },
                    body=body,
                    received_at=envelope.received_at,
                )
            )
        except MonitorAdapterError as exc:
            status = 401 if exc.code in {
                "MONITOR_AUTH_FAILED",
                "MONITOR_REPLAY_REJECTED",
            } else 422
            raise AIOpsApplicationError(
                code=exc.code,
                message=str(exc),
                status_code=status,
                retryable=exc.retryable,
            ) from exc

        delivery_key = self._delivery_key(
            source_id=source_snapshot["source_id"],
            body_hash=body_hash,
            provider_delivery_id=batch.provider_delivery_id,
        )
        stored_payload = await self._payload_store.store_verified(
            source_id=source_snapshot["source_id"],
            body=body,
            content_hash=body_hash,
        )
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            source = await uow.monitor_sources.get_scoped(
                monitor_source_id=source_snapshot["source_uuid"],
                domain_id=source_snapshot["domain_id"],
                lock=True,
            )
            if (
                source is None
                or source.status != "ACTIVE"
                or int(source.row_version)
                != source_snapshot["source_version"]
            ):
                raise validation_failed("监控源配置在验签期间发生变化")
            source_system = f"MONITOR:{source.monitor_source_id}"
            existing = await uow.inbox.get_by_message(
                source_system=source_system,
                message_key=delivery_key,
                lock=True,
            )
            if existing is not None:
                return await self._duplicate_receipt(uow, existing)
            inbox = InboxEntity(
                inbox_id=uuid7(),
                source_system=source_system,
                message_key=delivery_key,
                message_type="MONITOR_WEBHOOK.v1",
                payload_json={
                    "content_type": envelope.content_type,
                    "event_count": len(batch.events),
                    "normalizers": sorted(
                        {
                            item.normalizer_version
                            for item in batch.events
                        }
                    ),
                },
                payload_uri=stored_payload.uri,
                payload_hash=body_hash,
                status="PROCESSING",
                received_at=envelope.received_at,
                row_version=1,
            )
            try:
                async with uow.session.begin_nested():
                    await uow.inbox.add(inbox)
            except IntegrityError:
                existing = await uow.inbox.get_by_message(
                    source_system=source_system,
                    message_key=delivery_key,
                    lock=True,
                )
                if existing is None:
                    raise
                return await self._duplicate_receipt(uow, existing)

            event_ids = []
            alert_ids = []
            for normalized in batch.events:
                result = await self._process_event(
                    uow=uow,
                    source=source,
                    inbox=inbox,
                    event=normalized,
                    received_at=envelope.received_at,
                    trace_id=envelope.request_id,
                    now=now,
                )
                if result is not None:
                    event_id, alert_id = result
                    event_ids.append(event_id)
                    if alert_id is not None:
                        alert_ids.append(alert_id)
            if batch.events and not event_ids:
                inbox.status = "IGNORED"
                inbox.error_code = "MONITOR_TARGET_NOT_FOUND"
                inbox.error_message = "经验证事件未匹配 Active Target"
            else:
                inbox.status = "PROCESSED"
            inbox.processed_at = now
            await uow.commit()
            return MonitorWebhookReceipt(
                inbox_id=inbox.inbox_id,
                accepted=True,
                event_ids=tuple(event_ids),
                alert_ids=tuple(dict.fromkeys(alert_ids)),
            )

    @staticmethod
    async def _duplicate_receipt(uow, inbox) -> MonitorWebhookReceipt:
        event_ids = await uow.alerts.list_event_ids_by_inbox(
            inbox_id=inbox.inbox_id
        )
        return MonitorWebhookReceipt(
            inbox_id=inbox.inbox_id,
            accepted=inbox.status in {"PROCESSED", "IGNORED"},
            duplicate=True,
            event_ids=tuple(event_ids),
        )

    async def _load_source(self, webhook_key_hash: str) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            source = await uow.monitor_sources.get_by_webhook_hash(
                webhook_key_hash=webhook_key_hash, now=now
            )
            if source is None:
                raise AIOpsApplicationError(
                    code="MONITOR_ROUTE_NOT_FOUND",
                    message="Webhook 路由不存在",
                    status_code=404,
                )
            return {
                "source_uuid": source.monitor_source_id,
                "source_id": str(source.monitor_source_id),
                "source_type": source.source_type,
                "source_version": int(source.row_version),
                                "domain_id": int(source.domain_id),
                "endpoint": source.endpoint or "https://webhook.invalid",
                "webhook_secret_ref": source.webhook_secret_ref,
                "capabilities": dict(source.capabilities_json or {}),
            }

    def _decode_body(self, envelope: MonitorWebhookEnvelope) -> bytes:
        if envelope.raw_body_base64 is None:
            raise validation_failed("当前部署尚未启用 Webhook 对象存储")
        try:
            body = base64.b64decode(
                envelope.raw_body_base64, validate=True
            )
        except ValueError as exc:
            raise validation_failed("Webhook Base64 正文无效") from exc
        if not body or len(body) > self._max_webhook_bytes:
            raise validation_failed("Webhook 正文为空或超过大小限制")
        return body

    @staticmethod
    def _delivery_key(
        *,
        source_id: str,
        body_hash: str,
        provider_delivery_id: str | None,
    ) -> str:
        basis = provider_delivery_id or body_hash
        return hashlib.sha256(
            f"{source_id}|{basis}|{body_hash}".encode("utf-8")
        ).hexdigest()

    async def _process_event(
        self,
        *,
        uow,
        source,
        inbox,
        event: NormalizedMonitorEvent,
        received_at: datetime,
        trace_id: str,
        now: datetime,
    ):
        existing_event = await uow.alerts.get_event_by_source(
            monitor_source_id=source.monitor_source_id,
            source_event_key=event.source_event_key,
        )
        if existing_event is not None:
            return existing_event.event_id, existing_event.alert_id
        monitor = await uow.targets.get_monitor_by_external(
            monitor_source_id=source.monitor_source_id,
            external_target_key=event.external_target_key,
            lock=True,
        )
        if monitor is None:
            return None
        target = await uow.targets.get_scoped(
            target_id=monitor.target_id,
            domain_id=int(source.domain_id),
            lock=True,
        )
        if target is None or target.status != "ACTIVE":
            return None
        fingerprint = hashlib.sha256(
            (
                f"{source.monitor_source_id}|"
                f"{event.external_target_key}|"
                f"{event.fingerprint_basis}"
            ).encode("utf-8")
        ).hexdigest()
        entity = OpsEventEntity(
            event_id=uuid7(),
            target_id=target.target_id,
            monitor_source_id=source.monitor_source_id,
            source_inbox_id=inbox.inbox_id,
            source_event_key=event.source_event_key,
            event_type=event.event_type,
            severity=str(event.severity),
            event_status=str(event.event_status),
            occurred_at=event.occurred_at,
            received_at=received_at,
            fingerprint=fingerprint,
            payload_json={
                "summary": event.summary,
                "provider_attributes": event.provider_attributes,
            },
            payload_hash=sha256_json(event.model_dump(mode="json")),
            normalizer_version=event.normalizer_version,
            processing_status="RECEIVED",
            trace_id=trace_id,
        )
        try:
            async with uow.session.begin_nested():
                await uow.alerts.add_event(entity)
        except IntegrityError:
            existing_event = await uow.alerts.get_event_by_source(
                monitor_source_id=source.monitor_source_id,
                source_event_key=event.source_event_key,
            )
            if existing_event is None:
                raise
            return existing_event.event_id, existing_event.alert_id

        alert = await uow.alerts.get_active_alert(
            target_id=target.target_id,
            fingerprint=fingerprint,
            lock=True,
        )
        if event.event_status == "RESOLVED":
            if alert is not None and event.occurred_at >= alert.last_seen_at:
                alert.status = "RESOLVED"
                alert.resolved_at = event.occurred_at
                alert.last_seen_at = event.occurred_at
                alert.event_count = int(alert.event_count) + 1
        elif alert is None:
            candidate = OpsAlertEntity(
                alert_id=uuid7(),
                target_id=target.target_id,
                fingerprint=fingerprint,
                status="OPEN",
                severity=str(event.severity),
                summary=event.summary,
                correlation_json={
                    "monitor_source_id": str(source.monitor_source_id),
                    "external_target_fingerprint": hashlib.sha256(
                        event.external_target_key.encode()
                    ).hexdigest(),
                },
                first_seen_at=event.occurred_at,
                last_seen_at=event.occurred_at,
                event_count=1,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
            try:
                async with uow.session.begin_nested():
                    await uow.alerts.add_alert(candidate)
                alert = candidate
            except IntegrityError:
                alert = await uow.alerts.get_active_alert(
                    target_id=target.target_id,
                    fingerprint=fingerprint,
                    lock=True,
                )
                if alert is None:
                    raise
                self._merge_alert(alert, event)
        else:
            self._merge_alert(alert, event)

        entity.alert_id = alert.alert_id if alert is not None else None
        entity.processing_status = (
            "CORRELATED" if alert is not None else "IGNORED"
        )
        if (
            alert is not None
            and alert.status != "RESOLVED"
            and await self._auto_run_allowed(
                uow=uow,
                target=target,
                severity=alert.severity,
                fingerprint=alert.fingerprint,
                now=now,
            )
            and await uow.runs.get_active_by_alert(
                alert_id=alert.alert_id
            )
            is None
        ):
            await self._enqueue_auto_run(
                uow=uow,
                target=target,
                alert=alert,
                event_entity=entity,
                trace_id=trace_id,
                now=now,
            )
        return entity.event_id, entity.alert_id

    @staticmethod
    def _merge_alert(alert, event: NormalizedMonitorEvent) -> None:
        alert.last_seen_at = max(alert.last_seen_at, event.occurred_at)
        if _SEVERITY_RANK[str(event.severity)] >= _SEVERITY_RANK[
            alert.severity
        ]:
            alert.severity = str(event.severity)
            alert.summary = event.summary
        alert.event_count = int(alert.event_count) + 1

    async def _auto_run_allowed(
        self,
        *,
        uow,
        target,
        severity: str,
        fingerprint: str,
        now: datetime,
    ) -> bool:
        binding = await uow.targets.get_agent_binding(
            target_id=target.target_id,
            agent_id=self._system_agent_id,
            domain_id=int(target.domain_id),
        )
        if (
            binding is None
            or binding.status != "ACTIVE"
        ):
            return False
        minimum = "CRITICAL"
        cooldown_seconds = 900
        if binding.policy_id is not None:
            policy = await uow.policies.get_scoped(
                policy_id=binding.policy_id,
                domain_id=int(target.domain_id),
            )
            if policy is None or policy.status != "ACTIVE":
                return False
            minimum = str(
                policy.rules_json.get(
                    "auto_observe_min_severity", "CRITICAL"
                )
            )
            cooldown_seconds = int(
                policy.rules_json.get("alert_cooldown_seconds", 900)
            )
        if _SEVERITY_RANK[severity] < _SEVERITY_RANK.get(
            minimum, _SEVERITY_RANK["CRITICAL"]
        ):
            return False
        latest = await uow.runs.get_latest_by_alert_fingerprint(
            target_id=target.target_id,
            fingerprint=fingerprint,
        )
        return latest is None or (
            now - latest.created_at
        ).total_seconds() >= cooldown_seconds

    async def _enqueue_auto_run(
        self,
        *,
        uow,
        target,
        alert,
        event_entity,
        trace_id: str,
        now: datetime,
    ) -> None:
        idempotency_key = f"alert:{alert.alert_id}:observe-run"
        if (
            await uow.outbox.get_by_idempotency(
                idempotency_key=idempotency_key
            )
            is not None
        ):
            return
        payload = {
                        "domain_id": int(target.domain_id),
            "agent_id": str(self._system_agent_id),
            "target_id": str(target.target_id),
            "alert_id": str(alert.alert_id),
            "event_id": str(event_entity.event_id),
            "occurred_at": event_entity.occurred_at.isoformat(),
            "trace_id": trace_id,
        }
        await uow.outbox.add(
            OutboxEntity(
                outbox_id=uuid7(),
                aggregate_type="ALERT",
                aggregate_id=alert.alert_id,
                event_type="OPS_ALERT_AUTO_RUN_REQUESTED",
                idempotency_key=idempotency_key,
                payload_json=payload,
                payload_hash=hashlib.sha256(
                    canonical_bytes(payload)
                ).hexdigest(),
                status="PENDING",
                available_at=now,
                max_attempts=5,
                trace_id=trace_id,
                created_at=now,
                updated_at=now,
            )
        )
