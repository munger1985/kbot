"""将共享 Outbox 事件幂等投影为 Inbox、待办和后台任务。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from loguru import logger

from main_api.entities import (
    BackgroundOperationEntity,
    NotificationInboxEntity,
    WorkItemEntity,
)
from platform_core.notifications import NotificationEnvelope, event_definition
from platform_core.notifications.aiops import presentation_severity, work_item_identity
from platform_core.notifications.source_order import output_event_version, source_version_decision
from platform_core.contracts.aiops.output_projection import OutputProjectionRequest
from platform_core.contracts.aiops.notification_projection import WorkflowProjectionRequest
from main_api.application.aiops_workflow_reconciliation import WorkflowSourceUnavailable
from main_api.application.aiops_output_reconciliation import projection_presentation


_TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}


class NotificationProjectionService:
    def __init__(self, *, external_config=None):
        self._external_config = external_config

    async def project(self, *, uow, outbox) -> None:
        if uow.notifications is None:
            raise RuntimeError("Notification Repository 未初始化")
        envelope = NotificationEnvelope.model_validate(outbox.payload_json)
        definition = event_definition(envelope.event_type)
        if outbox.producer_service != definition.producer_service:
            raise ValueError("NOTIFICATION_EVENT_PRODUCER_INVALID")
        if int(outbox.domain_id) != envelope.domain_id:
            raise ValueError("NOTIFICATION_EVENT_DOMAIN_MISMATCH")
        if (outbox.producer_service == "aiops" and definition.work_item_action
            and not definition.resolve_work_item):
            resource_type, resource_id = work_item_identity(resource_type=envelope.resource_type,
                resource_id=envelope.resource_id, facts=envelope.safe_data)
            if await uow.notifications.resource_work_item_closed(domain_id=envelope.domain_id,
                resource_type=resource_type, resource_id=resource_id, action_type=definition.work_item_action):
                return

        operation, deliver = await self._project_operation(
            repository=uow.notifications,
            outbox=outbox,
            envelope=envelope,
            definition=definition,
        )
        if not deliver:
            return
        recipients = set(envelope.recipient_actor_ids)
        if envelope.event_type == "aiops.proposal.review_required":
            recipients = await uow.access.notification_approvers(
                domain_id=envelope.domain_id, exclude_user_id=envelope.initiator_actor_id)
            if not recipients:
                # 进入隔离队列并暴露稳定错误码，不能向申请人伪装审批已送达。
                raise ValueError("NOTIFICATION_APPROVER_UNAVAILABLE")
        if operation is not None and definition.operation_status in _TERMINAL:
            recipients.update(await uow.notifications.watcher_actor_ids(
                operation_id=operation.operation_id,
                domain_id=envelope.domain_id,
            ))
        title = definition.title
        if self._external_config is not None and envelope.event_type.startswith("aiops."):
            from main_api.application.aiops_notification_routes import aiops_notification_route
            from platform_core.notifications.delivery import DeliveryScope, EXTERNAL_EVENTS
            from platform_core.notifications.external_projection import enqueue_external_event
            if envelope.event_type in EXTERNAL_EVENTS:
                route = aiops_notification_route(envelope, outbox.payload_json, domain_id=envelope.domain_id)
                if route:
                    await enqueue_external_event(config=self._external_config, repo=uow.external_delivery,
                        scope=DeliveryScope(domain_id=str(envelope.domain_id)), source_event_id=outbox.outbox_id,
                        event_type=envelope.event_type, route=route, legitimate_actors=recipients,
                        occurred_at=envelope.occurred_at, severity=envelope.safe_data.get("severity"))
        if envelope.resource_name:
            title = f"{title}：{envelope.resource_name}"[:300]

        if definition.resolve_work_item and envelope.event_type.startswith("aiops."):
            # 结果通知的收件人可能为空或不同，仍需关闭全部审批人的同一资源待办。
            await self._project_work_item(repository=uow.notifications, outbox=outbox,
                envelope=envelope, definition=definition,
                actor_id=envelope.initiator_actor_id or "system:notification", title=title)

        for actor_id in sorted(recipients):
            if definition.notify and await uow.notifications.preference_enabled(
                domain_id=envelope.domain_id,
                actor_id=actor_id,
                event_type=envelope.event_type,
            ):
                if not await uow.notifications.inbox_exists(
                    outbox_id=outbox.outbox_id, actor_id=actor_id,
                ):
                    await uow.notifications.add_inbox(NotificationInboxEntity(
                        outbox_id=outbox.outbox_id,
                        domain_id=envelope.domain_id,
                        recipient_actor_id=actor_id,
                        event_type=envelope.event_type,
                        category=definition.category,
                        severity=presentation_severity(envelope.safe_data, definition.severity)
                            if outbox.producer_service == "aiops" and envelope.event_type.startswith("aiops.situation.")
                            else definition.severity,
                        title=title,
                        summary=envelope.summary,
                        resource_type=envelope.resource_type,
                        resource_id=envelope.resource_id,
                        operation_id=(operation.operation_id if operation else None),
                        expires_at=envelope.occurred_at + timedelta(
                            days=definition.retention_days
                        ),
                    ))
            await self._project_work_item(
                repository=uow.notifications,
                outbox=outbox,
                envelope=envelope,
                definition=definition,
                actor_id=actor_id,
                title=title,
            )

    @staticmethod
    async def _project_operation(
        *, repository, outbox, envelope, definition,
    ) -> tuple[BackgroundOperationEntity | None, bool]:
        if definition.operation_status is None or envelope.operation_id is None:
            return None, True
        source_version = output_event_version(producer=outbox.producer_service,
            event_type=envelope.event_type, params=envelope.safe_data, data=envelope.safe_data)
        if source_version is not None:
            locator = OutputProjectionRequest(job_id=envelope.safe_data["job_id"],
                job_type=envelope.safe_data["job_type"], actor_id=envelope.initiator_actor_id,
                domain_id=str(envelope.domain_id))
            if str(locator.job_id) != envelope.operation_id:
                raise ValueError("NOTIFICATION_OUTPUT_JOB_MISMATCH")
        operation = await repository.operation_for_update(
            producer=outbox.producer_service,
            source_operation_id=envelope.operation_id,
        )
        if operation is None:
            operation = BackgroundOperationEntity(
                domain_id=envelope.domain_id,
                producer_service=outbox.producer_service,
                source_operation_id=envelope.operation_id,
                initiator_actor_id=envelope.initiator_actor_id,
                resource_type=envelope.resource_type,
                resource_id=envelope.resource_id,
                resource_name=envelope.resource_name,
                status=definition.operation_status,
                progress_current=envelope.safe_data.get("progress_current"),
                progress_total=envelope.safe_data.get("progress_total"),
                error_code=envelope.safe_data.get("error_code"),
                summary=envelope.summary,
                last_outbox_id=outbox.outbox_id,
                last_occurred_at=envelope.occurred_at,
                source_version=source_version,
            )
            await repository.add_operation(operation)
            return operation, True
        if operation.domain_id != envelope.domain_id:
            raise ValueError("NOTIFICATION_OPERATION_DOMAIN_MISMATCH")
        if source_version is not None:
            if operation.initiator_actor_id != envelope.initiator_actor_id:
                raise ValueError("NOTIFICATION_OUTPUT_ACTOR_MISMATCH")
            previous = NotificationEnvelope.model_validate(await repository.operation_event_payload(
                outbox_id=operation.last_outbox_id))
            if (previous.domain_id != envelope.domain_id
                or previous.operation_id != envelope.operation_id
                or previous.initiator_actor_id != envelope.initiator_actor_id
                or previous.safe_data.get("job_type") != envelope.safe_data.get("job_type")):
                raise ValueError("NOTIFICATION_OUTPUT_SCOPE_MISMATCH")
            decision = source_version_decision(incoming=source_version,
                current=operation.source_version, incoming_status=definition.operation_status,
                current_status=operation.status)
            if decision != "ADVANCE":
                return operation, decision == "DELIVER"
        elif operation.source_version is not None:
            raise ValueError("NOTIFICATION_OUTPUT_SOURCE_VERSION_REQUIRED")
        elif envelope.occurred_at < operation.last_occurred_at:
            return operation, True
        operation.status = definition.operation_status
        operation.resource_name = envelope.resource_name
        operation.progress_current = envelope.safe_data.get("progress_current")
        operation.progress_total = envelope.safe_data.get("progress_total")
        operation.error_code = envelope.safe_data.get("error_code")
        operation.summary = envelope.summary
        operation.last_outbox_id = outbox.outbox_id
        operation.last_occurred_at = envelope.occurred_at
        operation.source_version = source_version
        operation.row_version = int(operation.row_version) + 1
        return operation, True

    @staticmethod
    async def _project_work_item(
        *, repository, outbox, envelope, definition,
        actor_id: str, title: str,
    ) -> None:
        action_type = definition.work_item_action
        if action_type is None:
            return
        resource_type, resource_id = work_item_identity(resource_type=envelope.resource_type,
            resource_id=envelope.resource_id, facts=envelope.safe_data)
        if definition.resolve_work_item and envelope.event_type.startswith("aiops."):
            await repository.resolve_resource_work_items(domain_id=envelope.domain_id,
                resource_type=resource_type, resource_id=resource_id, action_type=action_type,
                outbox_id=outbox.outbox_id, occurred_at=envelope.occurred_at,
                status=definition.work_item_resolution)
        item = await repository.work_item_for_update(
            domain_id=envelope.domain_id,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action_type=action_type,
        )
        if definition.resolve_work_item:
            if item is None:
                item = WorkItemEntity(
                    domain_id=envelope.domain_id,
                    actor_id=actor_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action_type=action_type,
                    title=title,
                    summary=envelope.summary,
                    status=definition.work_item_resolution,
                    opened_outbox_id=outbox.outbox_id,
                    resolved_outbox_id=outbox.outbox_id,
                    last_occurred_at=envelope.occurred_at,
                )
                await repository.add_work_item(item)
                return
            if (
                item.status == "OPEN"
                and envelope.occurred_at >= item.last_occurred_at
            ):
                item.status = definition.work_item_resolution
                item.resolved_outbox_id = outbox.outbox_id
                item.last_occurred_at = envelope.occurred_at
                item.row_version = int(item.row_version) + 1
            return
        if envelope.event_type.startswith("aiops.") and await repository.resource_work_item_closed(
            domain_id=envelope.domain_id, resource_type=resource_type, resource_id=resource_id,
            action_type=action_type):
            return
        if item is None:
            await repository.add_work_item(WorkItemEntity(
                domain_id=envelope.domain_id,
                actor_id=actor_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action_type=action_type,
                title=title,
                summary=envelope.summary,
                opened_outbox_id=outbox.outbox_id,
                last_occurred_at=envelope.occurred_at,
            ))
        elif envelope.occurred_at >= item.last_occurred_at:
            item.status = "OPEN"
            item.resolved_outbox_id = None
            item.title = title
            item.summary = envelope.summary
            item.opened_outbox_id = outbox.outbox_id
            item.last_occurred_at = envelope.occurred_at
            item.row_version = int(item.row_version) + 1


class NotificationDispatcher:
    """短租约领取；每条事件使用独立 UoW 投影与隔离。"""

    def __init__(
        self, *, uow_factory, batch_size: int = 50,
        lease_seconds: int = 60, max_attempts: int = 5,
        external_config=None,
    ) -> None:
        self._uow_factory = uow_factory
        self._batch_size = batch_size
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._projection = NotificationProjectionService(external_config=external_config)

    async def dispatch_once(self) -> int:
        async with self._uow_factory() as uow:
            if uow.notifications is None:
                raise RuntimeError("Notification Repository 未初始化")
            await uow.notifications.recover_expired(
                now=datetime.now(timezone.utc)
            )
            claimed = await uow.notifications.claim_batch(
                limit=self._batch_size,
                lease_seconds=self._lease_seconds,
            )
            await uow.commit()
        for outbox_id, lease_token in claimed:
            try:
                async with self._uow_factory() as uow:
                    if uow.notifications is None:
                        raise RuntimeError("Notification Repository 未初始化")
                    row = await uow.notifications.claimed(
                        outbox_id=outbox_id, lease_token=lease_token,
                    )
                    if row is None:
                        continue
                    await self._projection.project(uow=uow, outbox=row)
                    await uow.notifications.mark_published(row)
                    await uow.commit()
            except Exception as exc:
                error_code = ("NOTIFICATION_APPROVER_UNAVAILABLE"
                    if str(exc) == "NOTIFICATION_APPROVER_UNAVAILABLE" else type(exc).__name__.upper()[:128])
                async with self._uow_factory() as uow:
                    if uow.notifications is None:
                        raise RuntimeError("Notification Repository 未初始化")
                    status = await uow.notifications.release_failed(
                        outbox_id=outbox_id,
                        lease_token=lease_token,
                        error_code=error_code,
                        max_attempts=self._max_attempts,
                    )
                    await uow.commit()
                logger.warning(
                    "通知事件投影失败 | outbox_id={} | status={} | error_code={}",
                    outbox_id, status, error_code,
                )
        return len(claimed)


async def reconcile_output_operations(*, uow_factory, resolver, limit: int = 50) -> int:
    """在读事务外调用权威服务，每条快照只允许 CAS 写回观察到的版本。"""
    async with uow_factory() as uow:
        rows = await uow.notifications.output_operation_candidates(limit=limit)
    applied = 0
    for row in rows:
        try:
            envelope = NotificationEnvelope.model_validate(row["envelope"])
            if (envelope.domain_id != row["domain_id"]
                or envelope.operation_id != row["source_operation_id"]
                or envelope.initiator_actor_id != row["initiator_actor_id"]
                or str(envelope.safe_data.get("job_id")) != row["source_operation_id"]):
                raise ValueError("通知任务定位与来源不一致")
            snapshot = await resolver.resolve(OutputProjectionRequest(
                job_id=row["source_operation_id"], job_type=envelope.safe_data["job_type"],
                actor_id=row["initiator_actor_id"], domain_id=str(int(row["domain_id"]))))
            async with uow_factory() as uow:
                # 平台后台任务的 RUNNING 包含排队；业务回执仍保留精确 QUEUED。
                changed = await uow.notifications.reconcile_output_operation(
                    operation_id=row["operation_id"], domain_id=int(row["domain_id"]),
                    source_operation_id=row["source_operation_id"],
                    expected_row_version=int(row["row_version"]),
                    source_version=snapshot.source_version,
                    source_updated_at=snapshot.source_updated_at,
                    summary=projection_presentation(snapshot)["summary"],
                    error_code=snapshot.error_code,
                    status="RUNNING" if snapshot.status == "QUEUED" else snapshot.status)
                await uow.commit()
                applied += int(changed)
        except Exception as exc:
            logger.warning("AIOps 通知任务对账失败 | operation_id={} | error={}",
                row["operation_id"], type(exc).__name__)
            async with uow_factory() as uow:
                await uow.notifications.defer_output_reconciliation(
                    operation_id=row["operation_id"], domain_id=int(row["domain_id"]),
                    expected_row_version=int(row["row_version"]))
                await uow.commit()
    return applied


async def reconcile_workflow_items(*, uow_factory, resolver, limit: int = 50) -> int:
    """外部读取不持有数据库锁；成功和失败均按观察版本轮转候选。"""
    async with uow_factory() as uow:
        rows = await uow.notifications.workflow_candidates(limit=limit)
        operations = await uow.notifications.workflow_operation_candidates(limit=limit)
    applied = 0
    for row in rows:
        status = None
        try:
            snapshot = await resolver.resolve(WorkflowProjectionRequest(
                resource_type=row["resource_type"], resource_id=row["resource_id"],
                domain_id=str(row["domain_id"])))
            status = snapshot.work_item_status
        except WorkflowSourceUnavailable:
            status = "CANCELLED"
            logger.info("通知待办来源已不可用，关闭失效投影 | work_item_id={}", row["work_item_id"])
        except Exception as exc:
            logger.warning("AIOps 待办对账失败 | work_item_id={} | error={}",
                row["work_item_id"], type(exc).__name__)
        async with uow_factory() as uow:
            changed = await uow.notifications.reconcile_workflow_item(row=row, status=status)
            await uow.commit()
            applied += int(changed and status is not None)
    for row in operations:
        status = None
        source_unavailable = False
        try:
            snapshot = await resolver.resolve(WorkflowProjectionRequest(resource_type="aiops_run",
                resource_id=row["resource_id"], domain_id=str(row["domain_id"])))
            status = {"CREATED": "RUNNING", "QUEUED": "RUNNING", "RUNNING": "RUNNING",
                "WAITING_INPUT": "WAITING_USER", "WAITING_APPROVAL": "WAITING_USER",
                "COMPLETED": "SUCCEEDED", "PARTIAL": "PARTIAL", "FAILED": "FAILED",
                "CANCELLED": "CANCELLED", "EXPIRED": "EXPIRED"}[snapshot.source_status]
        except WorkflowSourceUnavailable:
            status, source_unavailable = "EXPIRED", True
            logger.info("运行来源已不可用，任务投影已失效；不推断原执行结果 | operation_id={}", row["operation_id"])
        except Exception as exc:
            logger.warning("AIOps 运行投影对账失败 | operation_id={} | error={}",
                row["operation_id"], type(exc).__name__)
        async with uow_factory() as uow:
            await uow.notifications.reconcile_workflow_operation(row=row, status=status,
                **({"source_unavailable": True} if source_unavailable else {}))
            await uow.commit()
    return applied
