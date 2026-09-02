"""AIOps Application 层依赖的持久化 Port。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from aiops_agent.entities import (
        ChangeProposalEntity,
        InboxEntity,
        InspectionPlanEntity,
        DiagnosticSourceEntity,
        SituationEntity,
        OpsArtifactEntity,
        SignalEventEntity,
        OpsRunEntity,
        OpsRunEventEntity,
        OpsTaskEntity,
        OutboxEntity,
        PolicyEntity,
        TargetEntity,
    )


class TargetRepositoryPort(Protocol):
    async def add_target(self, entity: TargetEntity) -> TargetEntity: ...

    async def get_scoped(
        self,
        *,
        target_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> TargetEntity | None: ...


class DiagnosticSourceRepositoryPort(Protocol):
    async def add(
        self, entity: DiagnosticSourceEntity
    ) -> DiagnosticSourceEntity: ...

    async def get_scoped(
        self,
        *,
        diagnostic_source_id: UUID,
        domain_id: int,
        lock: bool = False,
    ) -> DiagnosticSourceEntity | None: ...

    async def delete_source(self, entity: DiagnosticSourceEntity) -> None: ...


class PolicyRepositoryPort(Protocol):
    async def add(self, entity: PolicyEntity) -> PolicyEntity: ...

    async def get_active(
        self,
        *,
        domain_id: int,
        policy_key: str,
        lock: bool = False,
    ) -> PolicyEntity | None: ...


class SituationRepositoryPort(Protocol):
    async def add_event(self, entity: SignalEventEntity) -> SignalEventEntity: ...

    async def add_situation(self, entity: SituationEntity) -> SituationEntity: ...


class OpsRunRepositoryPort(Protocol):
    async def add_run(self, entity: OpsRunEntity) -> OpsRunEntity: ...

    async def add_task(self, entity: OpsTaskEntity) -> OpsTaskEntity: ...

    async def add_artifact(
        self, entity: OpsArtifactEntity
    ) -> OpsArtifactEntity: ...

    async def claim_task(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> OpsTaskEntity | None: ...

    async def append_event(
        self,
        *,
        ops_run_id: UUID,
        event_type: str,
        visibility: str,
        payload_json: dict,
        ops_task_id: UUID | None = None,
        event_key: str | None = None,
    ) -> OpsRunEventEntity: ...


class ChangeRepositoryPort(Protocol):
    async def add_proposal(
        self, entity: ChangeProposalEntity
    ) -> ChangeProposalEntity: ...


class InspectionRepositoryPort(Protocol):
    async def add_plan(
        self, entity: InspectionPlanEntity
    ) -> InspectionPlanEntity: ...

    async def claim_due_plan(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> InspectionPlanEntity | None: ...


class InboxRepositoryPort(Protocol):
    async def add(self, entity: InboxEntity) -> InboxEntity: ...

    async def get_by_message(
        self,
        *,
        source_system: str,
        message_key: str,
        lock: bool = False,
    ) -> InboxEntity | None: ...


class OutboxRepositoryPort(Protocol):
    async def add(self, entity: OutboxEntity) -> OutboxEntity: ...

    async def get_by_idempotency(
        self, *, idempotency_key: str
    ) -> OutboxEntity | None: ...

    async def get_latest_by_idempotency_prefix(
        self,
        *,
        aggregate_type: str,
        aggregate_id: UUID,
        idempotency_prefix: str,
        event_type: str,
    ) -> OutboxEntity | None: ...

    async def claim(
        self,
        *,
        now: datetime,
        lease_owner: str,
        lease_token: UUID,
        lease_until: datetime,
    ) -> OutboxEntity | None: ...


class AIOpsUnitOfWorkPort(Protocol):
    targets: TargetRepositoryPort
    diagnostic_sources: DiagnosticSourceRepositoryPort
    policies: PolicyRepositoryPort
    situations: SituationRepositoryPort
    runs: OpsRunRepositoryPort
    changes: ChangeRepositoryPort
    inspections: InspectionRepositoryPort
    inbox: InboxRepositoryPort
    outbox: OutboxRepositoryPort

    async def __aenter__(self) -> "AIOpsUnitOfWorkPort": ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...
