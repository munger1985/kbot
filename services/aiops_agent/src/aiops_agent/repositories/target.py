"""Target、Agent Binding 与策略聚合的 Repository。"""

from collections.abc import Callable, Collection
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, and_, case, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.application.errors import StateConflictError
from aiops_agent.entities import (
    ActivitySampleEntity,
    AIOpsAgentVersionTargetEntity,
    ApprovalTokenEntity,
    ChangeProposalEntity,
    DiagnosticSourceEntity,
    EvidenceRequestEntity,
    ExecutionEntity,
    HitlEntity,
    ImageEvidenceProcessingEntity,
    NotificationSubscriptionEntity,
    OperationsKnowledgeSourceEntity,
    OpsAnswerBlockEntity,
    OpsAnswerCitationEntity,
    OpsArtifactEntity,
    OpsConversationEntity,
    OpsConversationMessageEntity,
    OpsConversationTurnEntity,
    OpsInvestigationRevisionEntity,
    OpsPlaybookInvocationEntity,
    OpsRunEntity,
    OpsRunEventEntity,
    OpsTaskEntity,
    OpsToolInvocationEntity,
    OpsTurnEventEntity,
    OpsTurnEvidenceEntity,
    OpsTurnInputItemEntity,
    OpsTurnRunEntity,
    PolicyEntity,
    RecoveryDrillEntity,
    RecoveryProfileEntity,
    ReportEntity,
    ReportSourceEntity,
    SignalEventEntity,
    SituationEntity,
    SituationEventEntity,
    TargetBindingEntity,
    TargetEntity,
    TargetFactEntity,
    TargetSourceBindingEntity,
    WorkItemActivityEntity,
    WorkItemEntity,
    WorkItemLinkEntity,
    WorkItemOccurrenceEntity,
    WorkloadMetricEntity,
    WorkloadSnapshotEntity,
    WorkloadStatementEntity,
)
from aiops_agent.domain.operations import TERMINAL_RUN_STATUSES
from aiops_agent.repositories._base import AIOpsRepository


class TargetRepository(AIOpsRepository):
    _WORKLOAD_RUNTIME_FIELDS = frozenset(
        {
            "workload_next_run_at",
            "workload_consecutive_failures",
            "workload_last_collected_at",
            "workload_last_error_code",
            "activity_next_sample_at",
            "activity_sampler_status",
            "activity_sampler_disabled_reason",
            "activity_consecutive_failures",
            "activity_daily_bytes",
            "activity_daily_bucket",
            "activity_last_sampled_at",
        }
    )
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ):
        super().__init__(session, assert_active)

    async def add_target(self, entity: TargetEntity) -> TargetEntity:
        return await self._add(entity)

    async def has_active_runs(self, *, target_id: UUID) -> bool:
        """判断 Target 是否仍有不可安全中断的运行。"""
        self._check_active()
        terminal = tuple(status.value for status in TERMINAL_RUN_STATUSES)
        statement = select(func.count()).select_from(OpsRunEntity).where(
            OpsRunEntity.target_id == target_id,
            OpsRunEntity.status.not_in(terminal),
        )
        return bool((await self._session.execute(statement)).scalar_one())

    async def delete_target_with_history(self, entity: TargetEntity) -> None:
        """在当前事务内删除 Target 专属配置和历史，保留共享配置。"""
        self._check_active()

        target_id = entity.target_id
        run_ids = select(OpsRunEntity.ops_run_id).where(
            OpsRunEntity.target_id == target_id
        )
        artifact_ids = select(OpsArtifactEntity.artifact_id).where(
            OpsArtifactEntity.ops_run_id.in_(run_ids)
        )
        task_ids = select(OpsTaskEntity.ops_task_id).where(
            OpsTaskEntity.ops_run_id.in_(run_ids)
        )
        proposal_ids = select(ChangeProposalEntity.proposal_id).where(
            or_(
                ChangeProposalEntity.target_id == target_id,
                ChangeProposalEntity.ops_run_id.in_(run_ids),
            )
        )
        hitl_ids = select(HitlEntity.hitl_id).where(
            HitlEntity.ops_run_id.in_(run_ids)
        )
        report_ids = select(ReportEntity.report_id).where(
            or_(
                ReportEntity.target_id == target_id,
                ReportEntity.ops_run_id.in_(run_ids),
            )
        )
        conversation_ids = select(OpsConversationEntity.conversation_id).where(
            OpsConversationEntity.target_id == target_id
        )
        turn_ids = select(OpsConversationTurnEntity.turn_id).where(
            OpsConversationTurnEntity.conversation_id.in_(conversation_ids)
        )
        revision_ids = select(OpsInvestigationRevisionEntity.revision_id).where(
            OpsInvestigationRevisionEntity.turn_id.in_(turn_ids)
        )
        playbook_ids = select(
            OpsPlaybookInvocationEntity.playbook_invocation_id
        ).where(OpsPlaybookInvocationEntity.turn_id.in_(turn_ids))
        tool_ids = select(OpsToolInvocationEntity.tool_invocation_id).where(
            OpsToolInvocationEntity.turn_id.in_(turn_ids)
        )
        evidence_ids = select(OpsTurnEvidenceEntity.turn_evidence_id).where(
            OpsTurnEvidenceEntity.turn_id.in_(turn_ids)
        )
        answer_block_ids = select(OpsAnswerBlockEntity.answer_block_id).where(
            OpsAnswerBlockEntity.turn_id.in_(turn_ids)
        )
        request_ids = select(EvidenceRequestEntity.request_id).where(
            EvidenceRequestEntity.turn_id.in_(turn_ids)
        )
        work_item_ids = select(WorkItemEntity.work_item_id).where(
            WorkItemEntity.target_id == target_id
        )
        source_binding_ids = select(
            TargetSourceBindingEntity.target_source_binding_id
        ).where(TargetSourceBindingEntity.target_id == target_id)
        signal_ids = select(SignalEventEntity.signal_event_id).where(
            or_(
                SignalEventEntity.target_id == target_id,
                SignalEventEntity.source_binding_id.in_(source_binding_ids),
            )
        )
        situation_ids = select(SituationEntity.situation_id).where(
            SituationEntity.target_id == target_id
        )
        recovery_profile_ids = select(
            RecoveryProfileEntity.recovery_profile_id
        ).where(RecoveryProfileEntity.target_id == target_id)

        # 先解除其他 Target 可能持有的可空历史引用，避免误删共享父对象。
        await self._execute_update(
            OpsConversationTurnEntity,
            OpsConversationTurnEntity.resolved_target_id == target_id,
            {"resolved_target_id": None},
        )
        await self._execute_update(
            OpsConversationEntity,
            OpsConversationEntity.source_situation_id.in_(situation_ids),
            {"source_situation_id": None},
        )
        await self._execute_update(
            OpsConversationEntity,
            OpsConversationEntity.source_run_id.in_(run_ids),
            {"source_run_id": None},
        )
        await self._execute_update(
            OpsConversationEntity,
            OpsConversationEntity.source_report_id.in_(report_ids),
            {"source_report_id": None},
        )
        await self._execute_update(
            OpsRunEntity,
            OpsRunEntity.trigger_signal_event_id.in_(signal_ids),
            {"trigger_signal_event_id": None},
        )
        await self._execute_update(
            OpsRunEntity,
            OpsRunEntity.situation_id.in_(situation_ids),
            {"situation_id": None},
        )
        await self._execute_update(
            RecoveryDrillEntity,
            RecoveryDrillEntity.source_ops_run_id.in_(run_ids),
            {"source_ops_run_id": None},
        )

        # 清除 Run/Artifact、报告和自引用形成的环，再按子到父顺序删除。
        await self._execute_update(
            OpsRunEntity,
            OpsRunEntity.ops_run_id.in_(run_ids),
            {
                "final_artifact_id": None,
                "source_proposal_id": None,
                "source_result_artifact_id": None,
            },
        )
        await self._execute_update(
            OpsTaskEntity,
            OpsTaskEntity.ops_run_id.in_(run_ids),
            {"parent_task_id": None, "output_artifact_id": None},
        )
        await self._execute_update(
            ExecutionEntity,
            or_(
                ExecutionEntity.target_id == target_id,
                ExecutionEntity.ops_run_id.in_(run_ids),
            ),
            {"result_artifact_id": None, "rollback_of_execution_id": None},
        )
        await self._execute_update(
            ReportEntity,
            ReportEntity.report_id.in_(report_ids),
            {"content_artifact_id": None, "supersedes_report_id": None},
        )
        await self._execute_update(
            OpsPlaybookInvocationEntity,
            OpsPlaybookInvocationEntity.playbook_invocation_id.in_(playbook_ids),
            {"parent_invocation_id": None},
        )
        await self._execute_update(
            OpsAnswerBlockEntity,
            OpsAnswerBlockEntity.answer_block_id.in_(answer_block_ids),
            {"supersedes_id": None},
        )
        await self._execute_update(
            EvidenceRequestEntity,
            EvidenceRequestEntity.request_id.in_(request_ids),
            {"parent_request_id": None},
        )

        await self._execute_delete(
            ImageEvidenceProcessingEntity,
            ImageEvidenceProcessingEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsAnswerCitationEntity,
            or_(
                OpsAnswerCitationEntity.answer_block_id.in_(answer_block_ids),
                OpsAnswerCitationEntity.turn_evidence_id.in_(evidence_ids),
            ),
        )
        await self._execute_delete(
            OpsTurnEventEntity,
            OpsTurnEventEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsTurnEvidenceEntity,
            OpsTurnEvidenceEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsToolInvocationEntity,
            OpsToolInvocationEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsPlaybookInvocationEntity,
            OpsPlaybookInvocationEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            ImageEvidenceProcessingEntity,
            ImageEvidenceProcessingEntity.evidence_request_id.in_(request_ids),
        )
        await self._execute_delete(
            EvidenceRequestEntity,
            EvidenceRequestEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsAnswerBlockEntity,
            OpsAnswerBlockEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsTurnInputItemEntity,
            OpsTurnInputItemEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsConversationMessageEntity,
            OpsConversationMessageEntity.turn_id.in_(turn_ids),
        )

        await self._execute_delete(
            OperationsKnowledgeSourceEntity,
            or_(
                OperationsKnowledgeSourceEntity.source_report_id.in_(report_ids),
                OperationsKnowledgeSourceEntity.source_run_id.in_(run_ids),
                OperationsKnowledgeSourceEntity.source_artifact_id.in_(artifact_ids),
            ),
        )
        await self._execute_delete(
            ReportSourceEntity,
            or_(
                ReportSourceEntity.report_id.in_(report_ids),
                ReportSourceEntity.ops_run_id.in_(run_ids),
            ),
        )
        await self._execute_delete(
            WorkItemOccurrenceEntity,
            or_(
                WorkItemOccurrenceEntity.work_item_id.in_(work_item_ids),
                WorkItemOccurrenceEntity.ops_run_id.in_(run_ids),
                WorkItemOccurrenceEntity.situation_id.in_(situation_ids),
            ),
        )
        await self._execute_delete(
            ExecutionEntity,
            or_(
                ExecutionEntity.target_id == target_id,
                ExecutionEntity.ops_run_id.in_(run_ids),
            ),
        )
        await self._execute_delete(
            ApprovalTokenEntity,
            or_(
                ApprovalTokenEntity.proposal_id.in_(proposal_ids),
                ApprovalTokenEntity.hitl_id.in_(hitl_ids),
            ),
        )
        await self._execute_delete(
            HitlEntity,
            HitlEntity.ops_run_id.in_(run_ids),
        )
        await self._execute_delete(
            ChangeProposalEntity,
            ChangeProposalEntity.proposal_id.in_(proposal_ids),
        )
        await self._execute_delete(
            ReportEntity,
            ReportEntity.report_id.in_(report_ids),
        )
        await self._execute_delete(
            OpsInvestigationRevisionEntity,
            OpsInvestigationRevisionEntity.revision_id.in_(revision_ids),
        )
        await self._execute_delete(
            OpsTurnRunEntity,
            OpsTurnRunEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsConversationTurnEntity,
            OpsConversationTurnEntity.turn_id.in_(turn_ids),
        )
        await self._execute_delete(
            OpsConversationEntity,
            OpsConversationEntity.conversation_id.in_(conversation_ids),
        )
        await self._execute_delete(
            OpsRunEventEntity,
            OpsRunEventEntity.ops_run_id.in_(run_ids),
        )
        await self._execute_delete(
            OpsArtifactEntity,
            OpsArtifactEntity.artifact_id.in_(artifact_ids),
        )
        await self._execute_delete(
            OpsTaskEntity,
            OpsTaskEntity.ops_task_id.in_(task_ids),
        )
        await self._execute_delete(
            RecoveryDrillEntity,
            RecoveryDrillEntity.target_id == target_id,
        )
        await self._execute_delete(
            OpsRunEntity,
            OpsRunEntity.ops_run_id.in_(run_ids),
        )

        await self._execute_delete(
            SituationEventEntity,
            or_(
                SituationEventEntity.situation_id.in_(situation_ids),
                SituationEventEntity.signal_event_id.in_(signal_ids),
            ),
        )
        await self._execute_delete(
            SituationEntity,
            SituationEntity.situation_id.in_(situation_ids),
        )
        await self._execute_delete(
            SignalEventEntity,
            SignalEventEntity.signal_event_id.in_(signal_ids),
        )

        for model in (
            WorkItemActivityEntity,
            WorkItemLinkEntity,
        ):
            await self._execute_delete(model, model.work_item_id.in_(work_item_ids))
        await self._execute_delete(
            WorkItemEntity,
            WorkItemEntity.work_item_id.in_(work_item_ids),
        )
        await self._execute_delete(
            RecoveryDrillEntity,
            RecoveryDrillEntity.recovery_profile_id.in_(recovery_profile_ids),
        )
        await self._execute_delete(
            RecoveryProfileEntity,
            RecoveryProfileEntity.recovery_profile_id.in_(recovery_profile_ids),
        )
        for model in (
            WorkloadStatementEntity,
            WorkloadMetricEntity,
        ):
            await self._execute_delete(model, model.target_id == target_id)
        await self._execute_delete(
            WorkloadSnapshotEntity,
            WorkloadSnapshotEntity.target_id == target_id,
        )
        await self._execute_delete(
            ActivitySampleEntity,
            ActivitySampleEntity.target_id == target_id,
        )

        for model in (
            NotificationSubscriptionEntity,
            AIOpsAgentVersionTargetEntity,
            TargetFactEntity,
            TargetBindingEntity,
            TargetSourceBindingEntity,
        ):
            await self._execute_delete(model, model.target_id == target_id)
        await self._execute_delete(
            TargetEntity,
            TargetEntity.target_id == target_id,
        )
        await self._session.flush()

    async def _execute_delete(self, model, predicate) -> None:
        await self._session.execute(
            delete(model).where(predicate).execution_options(
                synchronize_session=False
            )
        )

    async def _execute_update(self, model, predicate, values: dict) -> None:
        await self._session.execute(
            update(model)
            .where(predicate)
            .values(**values)
            .execution_options(synchronize_session=False)
        )

    async def add_binding(
        self, entity: TargetBindingEntity
    ) -> TargetBindingEntity:
        return await self._add(entity)

    async def add_source_binding(
        self, entity: TargetSourceBindingEntity
    ) -> TargetSourceBindingEntity:
        return await self._add(entity)

    async def add_target_fact(
        self, entity: TargetFactEntity
    ) -> TargetFactEntity:
        return await self._add(entity)

    async def target_ids_shared_by_sources(
        self,
        *,
        domain_id: int,
        source_ids: Collection[UUID],
    ) -> list[UUID]:
        """返回所有指定监控源共同映射的授权 Target，不以运行健康拦截对话。"""
        normalized = tuple(dict.fromkeys(source_ids))
        if not normalized:
            return []
        rows = await self._session.execute(
            select(TargetSourceBindingEntity.target_id)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetEntity.domain_id == domain_id,
                TargetSourceBindingEntity.status == "ACTIVE",
                TargetSourceBindingEntity.diagnostic_source_id.in_(normalized),
            )
            .group_by(TargetSourceBindingEntity.target_id)
            .having(
                func.count(
                    func.distinct(
                        TargetSourceBindingEntity.diagnostic_source_id
                    )
                )
                == len(normalized)
            )
            .order_by(TargetSourceBindingEntity.target_id)
        )
        return list(rows.scalars())

    async def get_scoped(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetEntity | None:
        self._check_active()
        statement: Select = select(TargetEntity).where(
            TargetEntity.target_id == target_id,
            TargetEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None = None,
    ) -> list[TargetEntity]:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(TargetEntity.status.in_(statuses))
        statement = statement.order_by(
            TargetEntity.display_name,
            TargetEntity.target_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_monitoring_observation_candidates(
        self,
        *,
        limit: int,
    ) -> list[tuple[UUID, int]]:
        """列出已启用且具备指标监控映射的 Target 标识。"""

        self._check_active()
        statement = (
            select(TargetEntity.target_id, TargetEntity.domain_id)
            .join(
                TargetSourceBindingEntity,
                TargetSourceBindingEntity.target_id == TargetEntity.target_id,
            )
            .join(
                DiagnosticSourceEntity,
                DiagnosticSourceEntity.diagnostic_source_id
                == TargetSourceBindingEntity.diagnostic_source_id,
            )
            .where(
                TargetEntity.status == "ENABLED",
                TargetSourceBindingEntity.status == "ACTIVE",
                DiagnosticSourceEntity.status == "ENABLED",
                DiagnosticSourceEntity.source_type.in_(
                    {"PROMETHEUS", "ZABBIX"}
                ),
            )
            .distinct()
            .order_by(TargetEntity.target_id)
            .limit(limit)
        )
        rows = (await self._session.execute(statement)).all()
        return [(row.target_id, int(row.domain_id)) for row in rows]

    async def page_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_created_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[TargetEntity]:
        self._check_active()
        statement = select(TargetEntity).where(
            TargetEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(TargetEntity.status.in_(statuses))
        if before_created_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    TargetEntity.created_at < before_created_at,
                    and_(
                        TargetEntity.created_at == before_created_at,
                        TargetEntity.target_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            TargetEntity.created_at.desc(),
            TargetEntity.target_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def claim_due_connectivity(
        self, *, due_before: datetime, pending_before: datetime
    ) -> TargetEntity | None:
        """锁定一个到期 Target，供多副本 Scheduler 安全发起检查。"""
        claimed_id = await self._claim_oracle_uuid(
            plsql="""
                DECLARE
                    CURSOR c_claim IS
                        SELECT TARGET_ID
                        FROM KBOT_OPS_TARGET
                        WHERE READONLY_CONNECTION_ENABLED = 1
                          AND (
                              (
                                  LAST_CONNECTIVITY_CHECK_AT IS NULL
                                  AND (
                                      CONNECTIVITY_CHECK_REQUESTED_AT IS NULL
                                      OR CONNECTIVITY_CHECK_REQUESTED_AT
                                          <= :pending_before
                                  )
                              )
                              OR (
                                  LAST_CONNECTIVITY_CHECK_AT <= :due_before
                                  AND CONNECTIVITY_STATUS <> 'CHECKING'
                              )
                              OR (
                                  CONNECTIVITY_STATUS = 'CHECKING'
                                  AND CONNECTIVITY_CHECK_REQUESTED_AT
                                      <= :pending_before
                              )
                          )
                        ORDER BY LAST_CONNECTIVITY_CHECK_AT NULLS FIRST,
                                 TARGET_ID
                        FOR UPDATE OF TARGET_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={
                "due_before": due_before,
                "pending_before": pending_before,
            },
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(TargetEntity).where(TargetEntity.target_id == claimed_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(f"领取后的 Target 不存在：{claimed_id}")
        return entity

    async def claim_due_workload(self, *, now: datetime) -> TargetEntity | None:
        """以Oracle服务端游标领取一个到期工作负载Target。"""
        return await self._claim_due_collection(
            now=now,
            due_column="WORKLOAD_NEXT_RUN_AT",
            activity=False,
        )

    async def claim_due_activity(self, *, now: datetime) -> TargetEntity | None:
        """以Oracle服务端游标领取一个到期活动采样Target。"""
        return await self._claim_due_collection(
            now=now,
            due_column="ACTIVITY_NEXT_SAMPLE_AT",
            activity=True,
        )

    async def _claim_due_collection(
        self, *, now: datetime, due_column: str, activity: bool
    ) -> TargetEntity | None:
        activity_clause = (
            "AND ACTIVITY_SAMPLER_STATUS IN ('READY', 'DEGRADED')"
            if activity
            else ""
        )
        claimed_id = await self._claim_oracle_uuid(
            plsql=f"""
                DECLARE
                    CURSOR c_claim IS
                        SELECT TARGET_ID
                        FROM KBOT_OPS_TARGET
                        WHERE DB_TYPE IN ('MYSQL', 'POSTGRESQL')
                          AND STATUS = 'ENABLED'
                          AND READONLY_CONNECTION_ENABLED = 1
                          AND CONNECTIVITY_STATUS IN ('CONNECTED', 'DEGRADED')
                          {activity_clause}
                          AND {due_column} IS NOT NULL
                          AND {due_column} <= :now
                        ORDER BY {due_column}, TARGET_ID
                        FOR UPDATE OF TARGET_ID SKIP LOCKED;
                BEGIN
                    :claimed_id := NULL;
                    OPEN c_claim;
                    FETCH c_claim INTO :claimed_id;
                    CLOSE c_claim;
                END;
            """,
            parameters={"now": now},
        )
        if claimed_id is None:
            return None
        entity = (
            await self._session.execute(
                select(TargetEntity).where(TargetEntity.target_id == claimed_id)
            )
        ).scalar_one_or_none()
        if entity is None:
            raise StateConflictError(f"领取后的 Target 不存在：{claimed_id}")
        await self._session.flush()
        return entity

    async def update_workload_runtime_state(
        self, *, target_id: UUID, values: dict[str, object]
    ) -> None:
        """更新采集运行态，不改变用于凭据Grant围栏的Target row_version。"""
        self._check_active()
        invalid = set(values) - self._WORKLOAD_RUNTIME_FIELDS
        if invalid:
            raise ValueError(
                f"不允许更新的工作负载采集运行态字段：{', '.join(sorted(invalid))}"
            )
        if not values:
            return
        await self._session.execute(
            update(TargetEntity)
            .where(TargetEntity.target_id == target_id)
            .values(**values)
            .execution_options(synchronize_session=False)
        )

    async def update_target(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def get_agent_binding(
        self,
        *,
        target_id: UUID,
        agent_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.target_id == target_id,
                TargetBindingEntity.agent_id == agent_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_binding_scoped(
        self,
        *,
        binding_id: UUID,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.binding_id == binding_id,
                TargetBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_agent_bindings(
        self,
        *,
        target_id: UUID,
        domain_id: int,
    ) -> list[TargetBindingEntity]:
        self._check_active()
        statement = (
            select(TargetBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetBindingEntity.target_id,
            )
            .where(
                TargetBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
            .order_by(
                TargetBindingEntity.created_at,
                TargetBindingEntity.binding_id,
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def update_binding(
        self,
        *,
        binding_id: UUID,
        target_id: UUID,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetBindingEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetBindingEntity)
            .where(
                TargetBindingEntity.binding_id == binding_id,
                TargetBindingEntity.target_id == target_id,
                TargetBindingEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


    async def list_target_facts(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        active_only: bool = True,
    ) -> list[TargetFactEntity]:
        self._check_active()
        statement = (
            select(TargetFactEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetFactEntity.target_id,
            )
            .where(
                TargetFactEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if active_only:
            statement = statement.where(TargetFactEntity.status == "ACTIVE")
        statement = statement.order_by(
            TargetFactEntity.fact_type,
            TargetFactEntity.fact_key,
            TargetFactEntity.target_fact_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_target_fact_scoped(
        self,
        *,
        fact_id: UUID,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetFactEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetFactEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetFactEntity.target_id,
            )
            .where(
                TargetFactEntity.target_fact_id == fact_id,
                TargetFactEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_active_target_fact(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        fact_type: str,
        fact_key: str,
        lock: bool = False,
    ) -> TargetFactEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetFactEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetFactEntity.target_id,
            )
            .where(
                TargetFactEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
                TargetFactEntity.fact_type == fact_type,
                TargetFactEntity.fact_key == fact_key,
                TargetFactEntity.status == "ACTIVE",
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_source_bindings(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        active_only: bool = True,
    ) -> list[TargetSourceBindingEntity]:
        self._check_active()
        statement = (
            select(TargetSourceBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetSourceBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if active_only:
            statement = statement.where(TargetSourceBindingEntity.status == "ACTIVE")
        statement = statement.order_by(
            case(
                (TargetSourceBindingEntity.role == "PRIMARY", 0),
                else_=1,
            ),
            TargetSourceBindingEntity.priority,
            TargetSourceBindingEntity.target_source_binding_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_source_bindings_by_source(
        self,
        *,
        diagnostic_source_id: UUID,
        domain_id: int,
    ) -> list[TargetSourceBindingEntity]:
        """按监控数据源列出 Domain 内全部 Target 映射。"""
        self._check_active()
        statement = (
            select(TargetSourceBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetSourceBindingEntity.diagnostic_source_id
                == diagnostic_source_id,
                TargetEntity.domain_id == domain_id,
            )
            .order_by(
                TargetSourceBindingEntity.created_at.desc(),
                TargetSourceBindingEntity.target_source_binding_id.desc(),
            )
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_source_binding_scoped(
        self,
        *,
        target_source_binding_id: UUID,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetSourceBindingEntity | None:
        self._check_active()
        statement: Select = (
            select(TargetSourceBindingEntity)
            .join(
                TargetEntity,
                TargetEntity.target_id == TargetSourceBindingEntity.target_id,
            )
            .where(
                TargetSourceBindingEntity.target_source_binding_id == target_source_binding_id,
                TargetSourceBindingEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
            )
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def delete_source_binding_with_history(
        self,
        entity: TargetSourceBindingEntity,
    ) -> None:
        """删除监控映射，并保留已接收信号的历史事实。"""
        self._check_active()
        await self._execute_update(
            SignalEventEntity,
            SignalEventEntity.source_binding_id
            == entity.target_source_binding_id,
            {"source_binding_id": None},
        )
        await self._session.delete(entity)
        await self._session.flush()

    async def get_source_binding_by_locator(
        self,
        *,
        diagnostic_source_id: UUID,
        source_locator_key: str,
        lock: bool = False,
    ) -> TargetSourceBindingEntity | None:
        """同一 Source 下只允许精确外部目标映射。"""
        self._check_active()
        statement: Select = select(TargetSourceBindingEntity).where(
            TargetSourceBindingEntity.diagnostic_source_id == diagnostic_source_id,
            TargetSourceBindingEntity.source_locator_key == source_locator_key,
            TargetSourceBindingEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_source_binding_by_locator_any_status(
        self,
        *,
        diagnostic_source_id: UUID,
        source_locator_key: str,
        lock: bool = False,
    ) -> TargetSourceBindingEntity | None:
        """映射自然键在停用后也不能被另一个 Target 复用。"""
        self._check_active()
        statement: Select = select(TargetSourceBindingEntity).where(
            TargetSourceBindingEntity.diagnostic_source_id
            == diagnostic_source_id,
            TargetSourceBindingEntity.source_locator_key
            == source_locator_key,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def update_monitor(
        self,
        *,
        target_source_binding_id: UUID,
        target_id: UUID,
        expected_version: int,
        values: dict,
    ) -> bool:
        self._check_active()
        update_values = dict(values)
        update_values.update(
            {
                "row_version": TargetSourceBindingEntity.row_version + 1,
                "updated_at": datetime.now(UTC),
            }
        )
        statement = (
            update(TargetSourceBindingEntity)
            .where(
                TargetSourceBindingEntity.target_source_binding_id == target_source_binding_id,
                TargetSourceBindingEntity.target_id == target_id,
                TargetSourceBindingEntity.row_version == expected_version,
            )
            .values(**update_values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def reduce_source_binding_health(
        self,
        *,
        target_source_binding_id: UUID,
        expected_config_version: int,
        expected_health_version: int,
        health_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetSourceBindingEntity)
            .where(
                TargetSourceBindingEntity.target_source_binding_id
                == target_source_binding_id,
                TargetSourceBindingEntity.row_version
                == expected_config_version,
                TargetSourceBindingEntity.health_version
                == expected_health_version,
            )
            .values(
                health_status=health_status,
                last_health_check_at=checked_at,
                last_error_code=last_error_code,
                health_version=TargetSourceBindingEntity.health_version + 1,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_state(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.domain_id == domain_id,
                TargetEntity.row_version == expected_version,
                TargetEntity.status.in_(allowed_statuses),
            )
            .values(
                status=new_status,
                updated_by=updated_by,
                row_version=TargetEntity.row_version + 1,
                updated_at=datetime.now(UTC),
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_observed_status(
        self,
        *,
        target_id: UUID,
        observed_status: str,
        checked_at: datetime,
        last_error_code: str | None,
    ) -> bool:
        self._check_active()
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                or_(
                    TargetEntity.last_observed_at.is_(None),
                    TargetEntity.last_observed_at <= checked_at,
                ),
            )
            .values(
                observed_status=observed_status,
                last_observed_at=checked_at,
                last_error_code=last_error_code,
            )
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1

    async def update_connectivity(
        self,
        *,
        target_id: UUID,
        connectivity_check_request_id: UUID,
        expected_config_version: int,
        expected_connectivity_version: int,
        connectivity_status: str,
        checked_at: datetime,
        last_error_code: str | None,
        oracle_observation: dict[str, object] | None = None,
        database_version: str | None = None,
        capability_observation: dict[str, object] | None = None,
    ) -> bool:
        """仅在配置和检查版本未变化时归并数据库连通性。"""
        self._check_active()
        values: dict[str, object] = {
            "connectivity_status": connectivity_status,
            "last_connectivity_check_at": checked_at,
            "last_connectivity_success_at": (
                checked_at
                if connectivity_status in {"CONNECTED", "DEGRADED"}
                else TargetEntity.last_connectivity_success_at
            ),
            "last_error_code": last_error_code,
            "connectivity_version": TargetEntity.connectivity_version + 1,
        }
        if oracle_observation is not None:
            values.update(oracle_observation)
        if database_version is not None:
            values["version_code"] = database_version
        if capability_observation is not None:
            values["capabilities_json"] = capability_observation
        statement = (
            update(TargetEntity)
            .where(
                TargetEntity.target_id == target_id,
                TargetEntity.connectivity_check_request_id
                == connectivity_check_request_id,
                TargetEntity.row_version == expected_config_version,
                TargetEntity.connectivity_version
                == expected_connectivity_version,
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1


class PolicyRepository(AIOpsRepository):
    async def add(self, entity: PolicyEntity) -> PolicyEntity:
        return await self._add(entity)

    async def get_active(
        self,
        *,
        domain_id: int,
        policy_key: str,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.domain_id == domain_id,
            PolicyEntity.policy_key == policy_key,
            PolicyEntity.status == "ACTIVE",
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_scoped(
        self,
        *,
        policy_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> PolicyEntity | None:
        self._check_active()
        statement: Select = select(PolicyEntity).where(
            PolicyEntity.policy_id == policy_id,
            PolicyEntity.domain_id == domain_id,
        )
        if lock:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def lock_versions(
        self,
        *,
        domain_id: int,
        policy_key: str,
    ) -> list[PolicyEntity]:
        self._check_active()
        statement = (
            select(PolicyEntity)
            .where(
                PolicyEntity.domain_id == domain_id,
                PolicyEntity.policy_key == policy_key,
            )
            .order_by(PolicyEntity.version_no)
            .with_for_update()
        )
        return list((await self._session.execute(statement)).scalars())

    async def page_scoped(
        self,
        *,
        domain_id: int,
        statuses: Collection[str] | None,
        before_updated_at: datetime | None,
        before_id: UUID | None,
        limit: int,
    ) -> list[PolicyEntity]:
        self._check_active()
        statement = select(PolicyEntity).where(
            PolicyEntity.domain_id == domain_id,
        )
        if statuses:
            statement = statement.where(PolicyEntity.status.in_(statuses))
        if before_updated_at is not None and before_id is not None:
            statement = statement.where(
                or_(
                    PolicyEntity.updated_at < before_updated_at,
                    and_(
                        PolicyEntity.updated_at == before_updated_at,
                        PolicyEntity.policy_id < before_id,
                    ),
                )
            )
        statement = statement.order_by(
            PolicyEntity.updated_at.desc(),
            PolicyEntity.policy_id.desc(),
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def transition_status(
        self,
        *,
        policy_id: UUID,
        expected_version: int,
        allowed_statuses: Collection[str],
        new_status: str,
        updated_by: str,
        effective_at: datetime | None = None,
        retired_at: datetime | None = None,
    ) -> bool:
        self._check_active()
        values = {
            "status": new_status,
            "updated_by": updated_by,
            "row_version": PolicyEntity.row_version + 1,
            "updated_at": datetime.now(UTC),
        }
        if effective_at is not None:
            values["effective_at"] = effective_at
        if retired_at is not None:
            values["retired_at"] = retired_at
        statement = (
            update(PolicyEntity)
            .where(
                PolicyEntity.policy_id == policy_id,
                PolicyEntity.row_version == expected_version,
                PolicyEntity.status.in_(allowed_statuses),
            )
            .values(**values)
            .execution_options(synchronize_session=False)
        )
        result = await self._session.execute(statement)
        return result.rowcount == 1
