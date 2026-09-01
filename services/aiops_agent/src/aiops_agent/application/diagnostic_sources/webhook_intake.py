"""Webhook 验签、去重、Target 映射与故障情境关联。"""

from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from typing import Any

from platform_core.contracts.aiops import (
    SignalEventEnvelope,
    SignalEventIntakeReceipt,
)
from platform_core.identity import uuid7
from sqlalchemy.exc import IntegrityError

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.errors import (
    AIOpsApplicationError,
    dependency_unavailable,
    validation_failed,
)
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialService
from aiops_agent.application.runtime.service import (
    canonical_bytes,
    sha256_json,
)
from aiops_agent.contracts.evidence import NormalizedSignalEvent
from aiops_agent.domain.evidence import correlate_signal_event
from aiops_agent.entities import (
    InboxEntity,
    OutboxEntity,
    SignalEventEntity,
    SituationEntity,
    SituationEventEntity,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_RECEIVE,
    DiagnosticSourceContext,
    SignalWebhookRequest,
)

_SEVERITY_RANK = {
    "INFO": 0,
    "WARNING": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}


class SignalEventIntakeService:
    def __init__(
        self,
        *,
        uow_factory,
        diagnostic_source_registry,
        secret_store,
        system_agent_id,
        max_webhook_bytes: int,
        payload_store,
    ):
        self._uow_factory = uow_factory
        self._diagnostic_sources = diagnostic_source_registry
        self._secrets = secret_store
        self._system_agent_id = system_agent_id
        self._max_webhook_bytes = max_webhook_bytes
        self._payload_store = payload_store

    async def intake(
        self, envelope: SignalEventEnvelope
    ) -> SignalEventIntakeReceipt:
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
                code="SOURCE_AUTH_FAILED",
                message="诊断源未配置 Webhook 验签凭据",
                status_code=401,
            )
        try:
            secret = await self._secrets.resolve(webhook_ref)
        except AIOpsApplicationError as exc:
            raise dependency_unavailable("Webhook Secret 暂时不可用") from exc
        credentials = dict(secret.values)
        if "value" in credentials:
            credentials["webhook_secret"] = credentials.pop("value")
        try:
            adapter = self._diagnostic_sources.create(
                DiagnosticSourceContext(
                    source_id=source_snapshot["source_id"],
                    source_type=source_snapshot["source_type"],
                    adapter_id=source_snapshot["adapter_id"],
                    adapter_version=source_snapshot["adapter_version"],
                    config_version=source_snapshot["config_version"],
                    endpoint=source_snapshot["endpoint"],
                    credentials=credentials,
                    declared_capabilities=source_snapshot[
                        "declared_capabilities"
                    ],
                    config=source_snapshot["config"],
                ),
                capability=CAPABILITY_EVENT_RECEIVE,
            )
            batch = await adapter.verify_and_normalize_webhook(
                SignalWebhookRequest(
                    headers={
                        key.lower(): value
                        for key, value in envelope.signature_headers.items()
                    },
                    body=body,
                    received_at=envelope.received_at,
                )
            )
        except DiagnosticSourceAdapterError as exc:
            status = 401 if exc.code in {
                "SOURCE_AUTH_FAILED",
                "SOURCE_REPLAY_REJECTED",
            } else 422
            raise AIOpsApplicationError(
                code=exc.code,
                message=str(exc),
                status_code=status,
                retryable=exc.retryable,
            ) from exc
        except LookupError as exc:
            raise validation_failed(str(exc)) from exc

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
            source = await uow.diagnostic_sources.get_scoped(
                diagnostic_source_id=source_snapshot["source_uuid"],
                domain_id=source_snapshot["domain_id"],
                lock=True,
            )
            if (
                source is None
                or source.status != "ENABLED"
                or int(source.row_version)
                != source_snapshot["config_version"]
            ):
                raise validation_failed("诊断源配置在验签期间发生变化")
            source_system = f"DIAGNOSTIC_SOURCE:{source.diagnostic_source_id}"
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
                message_type="SOURCE_WEBHOOK.v1",
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

            signal_event_ids = []
            situation_ids = []
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
                    signal_event_id, situation_id = result
                    signal_event_ids.append(signal_event_id)
                    if situation_id is not None:
                        situation_ids.append(situation_id)
            if batch.events and not signal_event_ids:
                inbox.status = "IGNORED"
                inbox.error_code = "SOURCE_TARGET_NOT_FOUND"
                inbox.error_message = "经验证事件未匹配 Active Target"
            else:
                inbox.status = "PROCESSED"
            inbox.processed_at = now
            await uow.commit()
            return SignalEventIntakeReceipt(
                inbox_id=inbox.inbox_id,
                accepted=True,
                signal_event_ids=tuple(signal_event_ids),
                situation_ids=tuple(dict.fromkeys(situation_ids)),
            )

    @staticmethod
    async def _duplicate_receipt(uow, inbox) -> SignalEventIntakeReceipt:
        signal_event_ids = await uow.situations.list_signal_event_ids_by_inbox(
            inbox_id=inbox.inbox_id
        )
        return SignalEventIntakeReceipt(
            inbox_id=inbox.inbox_id,
            accepted=inbox.status in {"PROCESSED", "IGNORED"},
            duplicate=True,
            signal_event_ids=tuple(signal_event_ids),
        )

    async def _load_source(self, webhook_key_hash: str) -> dict[str, Any]:
        async with self._uow_factory() as uow:
            now = await uow.runs.database_now()
            source = await uow.diagnostic_sources.get_by_webhook_hash(
                webhook_key_hash=webhook_key_hash, now=now
            )
            if source is None:
                raise AIOpsApplicationError(
                    code="SOURCE_ROUTE_NOT_FOUND",
                    message="Webhook 路由不存在",
                    status_code=404,
                )
            return {
                "source_uuid": source.diagnostic_source_id,
                "source_id": str(source.diagnostic_source_id),
                "source_type": source.source_type,
                "adapter_id": source.adapter_id,
                "adapter_version": source.adapter_version,
                "config_version": int(source.row_version),
                "domain_id": int(source.domain_id),
                "endpoint": source.endpoint,
                "webhook_secret_ref": (
                    AIOpsManagedCredentialService.reference(
                        domain_id=int(source.domain_id),
                        external_key=source.diagnostic_source_id,
                        credential_kind="source_webhook",
                        credential_id=source.webhook_credential_id,
                    )
                    if source.webhook_credential_id
                    else None
                ),
                "declared_capabilities": dict(
                    source.declared_capabilities_json or {}
                ),
                "config": dict(source.config_json or {}),
            }

    def _decode_body(self, envelope: SignalEventEnvelope) -> bytes:
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
        event: NormalizedSignalEvent,
        received_at: datetime,
        trace_id: str,
        now: datetime,
    ):
        existing_event = await uow.situations.get_event_by_source(
            diagnostic_source_id=source.diagnostic_source_id,
            source_event_key=event.source_event_key,
        )
        if existing_event is not None:
            return existing_event.signal_event_id, None
        source_binding = await uow.targets.get_source_binding_by_locator(
            diagnostic_source_id=source.diagnostic_source_id,
            source_locator_key=event.source_locator_key,
            lock=True,
        )
        if source_binding is None:
            return None
        target = await uow.targets.get_scoped(
            target_id=source_binding.target_id,
            domain_id=int(source.domain_id),
            lock=True,
        )
        if target is None or target.status != "ENABLED":
            return None
        source_incident_hash = hashlib.sha256(
            (
                f"{source.diagnostic_source_id}|"
                f"{event.source_locator_key}|"
                f"{event.fingerprint_basis}"
            ).encode("utf-8")
        ).hexdigest()
        correlation = correlate_signal_event(
            target_id=str(target.target_id),
            source_event_class=event.event_type,
            mapping_overrides=source_binding.mapping_overrides_json,
        )
        entity = SignalEventEntity(
            signal_event_id=uuid7(),
            domain_id=int(source.domain_id),
            target_id=target.target_id,
            diagnostic_source_id=source.diagnostic_source_id,
            source_binding_id=source_binding.target_source_binding_id,
            source_inbox_id=inbox.inbox_id,
            source_event_key=event.source_event_key,
            signal_kind=(
                "RECOVERY"
                if str(event.event_status) == "RESOLVED"
                else "FAULT"
            ),
            event_class=event.event_type,
            severity=str(event.severity),
            normalized_status=(
                "RESOLVED"
                if str(event.event_status) == "RESOLVED"
                else "OPEN"
            ),
            source_status=str(event.event_status),
            summary=event.summary,
            occurred_at=event.occurred_at,
            received_at=received_at,
            dedup_hash=source_incident_hash,
            payload_json={
                "summary": event.summary,
                "provider_attributes": event.provider_attributes,
            },
            payload_hash=sha256_json(event.model_dump(mode="json")),
            evidence_locator_json={
                "source_locator_key": event.source_locator_key,
                "canonical_event_class": (
                    correlation.canonical_event_class
                ),
            },
            normalizer_version=event.normalizer_version,
            processing_status="RECEIVED",
            trace_id=trace_id,
        )
        try:
            async with uow.session.begin_nested():
                await uow.situations.add_event(entity)
        except IntegrityError:
            existing_event = await uow.situations.get_event_by_source(
                diagnostic_source_id=source.diagnostic_source_id,
                source_event_key=event.source_event_key,
            )
            if existing_event is None:
                raise
            return existing_event.signal_event_id, None

        situation_created = False
        situation_resolved = False
        situation = await uow.situations.get_active_situation(
            target_id=target.target_id,
            correlation_hash=correlation.correlation_hash,
            lock=True,
        )
        if event.event_status == "RESOLVED":
            if situation is not None:
                situation.last_observed_at = max(
                    situation.last_observed_at, event.occurred_at
                )
                situation.event_count = int(situation.event_count) + 1
        elif situation is None:
            candidate = SituationEntity(
                situation_id=uuid7(),
                domain_id=int(source.domain_id),
                target_id=target.target_id,
                situation_type=correlation.canonical_event_class,
                title=event.summary,
                summary=event.summary,
                status="OPEN",
                severity=str(event.severity),
                correlation_key=correlation.correlation_key,
                correlation_hash=correlation.correlation_hash,
                correlation_version=correlation.correlation_version,
                correlation_json={
                    "canonical_event_class": (
                        correlation.canonical_event_class
                    ),
                    "diagnostic_source_ids": [
                        str(source.diagnostic_source_id)
                    ],
                },
                first_observed_at=event.occurred_at,
                last_observed_at=event.occurred_at,
                event_count=1,
                row_version=1,
                created_at=now,
                updated_at=now,
            )
            try:
                async with uow.session.begin_nested():
                    await uow.situations.add_situation(candidate)
                situation = candidate
                situation_created = True
            except IntegrityError:
                situation = await uow.situations.get_active_situation(
                    target_id=target.target_id,
                    correlation_hash=correlation.correlation_hash,
                    lock=True,
                )
                if situation is None:
                    raise
                self._merge_situation(situation, event)
        else:
            self._merge_situation(situation, event)

        if situation is not None:
            self._merge_correlation_metadata(
                situation=situation,
                diagnostic_source_id=str(source.diagnostic_source_id),
                canonical_event_class=correlation.canonical_event_class,
            )

        entity.processing_status = (
            "CORRELATED" if situation is not None else "IGNORED"
        )
        if situation is not None:
            await uow.situations.add_situation_event(
                SituationEventEntity(
                    situation_event_id=uuid7(),
                    situation_id=situation.situation_id,
                    signal_event_id=entity.signal_event_id,
                    relation_type=(
                        "RECOVERY"
                        if str(event.event_status) == "RESOLVED"
                        else "TRIGGER"
                    ),
                    correlation_method=correlation.method,
                    correlation_score=1,
                    correlation_detail_json={
                        **correlation.detail,
                        "correlation_version": (
                            correlation.correlation_version
                        ),
                    },
                    attached_at=now,
                    attached_by="system:signal-intake",
                )
            )
            if (
                event.event_status == "RESOLVED"
                and not await uow.situations.has_open_signal_state(
                    situation_id=situation.situation_id
                )
            ):
                situation_resolved = situation.status != "RESOLVED"
                situation.status = "RESOLVED"
                situation.resolved_at = event.occurred_at
        auto_agent = None
        if situation is not None and situation.status != "RESOLVED":
            auto_agent = await self._resolve_auto_agent(
                uow=uow,
                target=target,
                source_id=source.diagnostic_source_id,
                severity=situation.severity,
                fingerprint=situation.correlation_hash,
                now=now,
            )
        if (
            situation is not None
            and situation.status != "RESOLVED"
            and event.event_status != "RESOLVED"
            and auto_agent is not None
            and await uow.runs.get_active_by_situation(
                situation_id=situation.situation_id
            )
            is None
        ):
            await self._enqueue_auto_run(
                uow=uow,
                target=target,
                situation=situation,
                event_entity=entity,
                agent_id=auto_agent.agent_id,
                trace_id=trace_id,
                now=now,
            )
        notifier = getattr(uow, "platform_notifications", None)
        if notifier is not None and situation is not None:
            if situation_created:
                await notifier.emit_situation_event(
                    target=target,
                    situation=situation,
                    event_type="aiops.situation.detected",
                    stage="SITUATION_DETECTED",
                    summary=(
                        f"{target.display_name}：{situation.summary or situation.title}"
                    )[:1000],
                    trace_id=trace_id,
                )
            elif situation_resolved:
                await notifier.emit_situation_event(
                    target=target,
                    situation=situation,
                    event_type="aiops.situation.recovered",
                    stage="SITUATION_RECOVERED",
                    summary=f"{target.display_name} 的相关故障信号已全部恢复",
                    trace_id=trace_id,
                )
        return entity.signal_event_id, (
            situation.situation_id if situation is not None else None
        )

    @staticmethod
    def _merge_situation(situation, event: NormalizedSignalEvent) -> None:
        situation.last_observed_at = max(
            situation.last_observed_at, event.occurred_at
        )
        if _SEVERITY_RANK[str(event.severity)] >= _SEVERITY_RANK[
            situation.severity
        ]:
            situation.severity = str(event.severity)
            situation.title = event.summary
            situation.summary = event.summary
        situation.event_count = int(situation.event_count) + 1

    @staticmethod
    def _merge_correlation_metadata(
        *,
        situation,
        diagnostic_source_id: str,
        canonical_event_class: str,
    ) -> None:
        metadata = dict(situation.correlation_json or {})
        source_ids = {
            str(item)
            for item in metadata.get("diagnostic_source_ids", ())
        }
        source_ids.add(diagnostic_source_id)
        metadata.update(
            {
                "canonical_event_class": canonical_event_class,
                "diagnostic_source_ids": sorted(source_ids),
            }
        )
        situation.correlation_json = metadata

    async def _resolve_auto_agent(
        self,
        *,
        uow,
        target,
        source_id,
        severity: str,
        fingerprint: str,
        now: datetime,
    ):
        resolved = await uow.agents.resolve_auto_alert(
            domain_id=int(target.domain_id),
            source_id=source_id,
            target_id=target.target_id,
        )
        if resolved is None:
            return None
        binding, policy = resolved
        minimum = str(policy.rules_json.get("auto_observe_min_severity", "CRITICAL"))
        cooldown_seconds = int(policy.rules_json.get("alert_cooldown_seconds", 900))
        if _SEVERITY_RANK[severity] < _SEVERITY_RANK.get(
            minimum, _SEVERITY_RANK["CRITICAL"]
        ):
            return None
        latest = await uow.runs.get_latest_by_situation_correlation(
            target_id=target.target_id,
            fingerprint=fingerprint,
        )
        allowed = latest is None or (
            now - latest.created_at
        ).total_seconds() >= cooldown_seconds
        return binding if allowed else None

    async def _enqueue_auto_run(
        self,
        *,
        uow,
        target,
        situation,
        event_entity,
        agent_id,
        trace_id: str,
        now: datetime,
    ) -> None:
        idempotency_key = (
            f"situation:{situation.situation_id}:agent:{agent_id}:observe-run"
        )
        if (
            await uow.outbox.get_by_idempotency(
                idempotency_key=idempotency_key
            )
            is not None
        ):
            return
        payload = {
            "domain_id": int(target.domain_id),
            "agent_id": str(agent_id),
            "target_id": str(target.target_id),
            "situation_id": str(situation.situation_id),
            "signal_event_id": str(event_entity.signal_event_id),
            "occurred_at": event_entity.occurred_at.isoformat(),
            "trace_id": trace_id,
        }
        await uow.outbox.add(
            OutboxEntity(
                outbox_id=uuid7(),
                aggregate_type="SITUATION",
                aggregate_id=situation.situation_id,
                event_type="OPS_SITUATION_AUTO_RUN_REQUESTED",
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
