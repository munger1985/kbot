"""Data Query 内部运行服务。"""

from collections.abc import Callable
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from data_query.application.runs import DataQueryRunError, create_data_query_run
from data_query.application.value_domains import (
    ValueDomainObserver,
    planning_dimension_payload,
)
from data_query.contracts import (
    CreateDataQueryRun,
    DataQueryResultView,
    DataQueryRunReceipt,
    DataQueryRunView,
    DataQueryPlanningContext,
    PlanningSemanticModel,
    SemanticModelDefinition,
)
from data_query.entities import DataQueryEventEntity
from data_query.persistence import DataQueryUnitOfWork


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class DataQueryRuntimeService:
    def __init__(
        self, *, uow_factory: Callable[[], DataQueryUnitOfWork],
        query_guardrail: dict[str, int],
        value_domain_observer: ValueDomainObserver | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._query_guardrail = dict(query_guardrail)
        self._value_domain_observer = value_domain_observer

    async def create_run(
        self, *, domain_id: int, actor_id: str, trace_id: str,
        actor_roles: tuple[str, ...],
        command: CreateDataQueryRun,
    ) -> DataQueryRunReceipt:
        return await create_data_query_run(
            uow_factory=self._uow_factory,domain_id=domain_id,
            actor_id=actor_id, actor_roles=actor_roles,
            trace_id=trace_id, command=command,
            query_guardrail=self._query_guardrail,
        )

    async def get_planning_context(
        self, *, domain_id: int, actor_id: str, actor_roles: tuple[str, ...],
        consumer_app_id: str, agent_id: UUID, agent_version_id: UUID,
    ) -> DataQueryPlanningContext:
        """仅返回逻辑名称和逻辑值域，绝不将物理对象、列或策略细节交给 Planner LLM。"""
        async with self._uow_factory() as uow:
            assert uow.agent_bindings and uow.semantic_models and uow.semantic_model_versions
            assert uow.platform_access is not None
            assert uow.data_sources is not None
            resolved_domain_id = await self._resolve_agent_domain(
                uow.platform_access,
                domain_id=domain_id,
                consumer_app_id=consumer_app_id,
                agent_id=agent_id,
                agent_version_id=agent_version_id,
            )
            bindings = await uow.agent_bindings.list_active_for_agent(
                domain_id=domain_id, consumer_app_id=consumer_app_id,
                agent_id=agent_id, agent_version_id=agent_version_id,
            )
            pending: list[tuple[object, SemanticModelDefinition, int, str, object]] = []
            for binding in bindings:
                model = await uow.semantic_models.get_by_id(semantic_model_id=binding.semantic_model_id)
                if model is None:
                    continue
                if model.domain_id != resolved_domain_id:
                    continue
                active = await uow.semantic_model_versions.get_active(semantic_model_id=model.semantic_model_id)
                if active is None:
                    continue
                source = await uow.data_sources.get_by_id(data_source_id=active.data_source_id)
                if source is None:
                    continue
                definition = SemanticModelDefinition.model_validate(active.definition_json)
                pending.append((model, definition, active.version_no, source.source_type, active.data_source_id))
            await uow.commit()
        models: list[PlanningSemanticModel] = []
        for model, definition, version_no, source_type, data_source_id in pending:
            datasets = {item.name: item for item in definition.datasets}
            observed = await self._planning_value_members(
                definition=definition,
                datasets=datasets,
                source_type=source_type,
                data_source_id=data_source_id,
            )
            models.append(PlanningSemanticModel(
                semantic_model_id=model.semantic_model_id, semantic_model_version=version_no,
                display_name=model.display_name,
                datasets=tuple({"name": item.name, "display_name": item.display_name} for item in definition.datasets),
                dimensions=tuple(
                    planning_dimension_payload(
                        item,
                        value_members=observed.get(item.name),
                    )
                    for item in definition.dimensions
                ),
                measures=tuple({"name": item.name, "dataset": item.dataset, "aggregation": item.aggregation, "value_type": item.value_type} for item in definition.measures),
                max_rows=self._query_guardrail["max_rows"],
            ))
        return DataQueryPlanningContext(
            agent_id=agent_id, consumer_app_id=consumer_app_id,
            agent_version_id=agent_version_id, models=tuple(models),
        )

    async def _planning_value_members(
        self, *, definition: SemanticModelDefinition, datasets: dict,
        source_type: str, data_source_id,
    ) -> dict[str, tuple]:
        observer = self._value_domain_observer
        if observer is None:
            return {
                item.name: item.value_members for item in definition.dimensions
            }
        async def resolve(dimension):
            dataset = datasets.get(dimension.dataset)
            if dataset is None:
                return dimension.name, dimension.value_members
            members = await observer.resolve_planning_members(
                dimension=dimension,
                dataset=dataset,
                source_type=source_type,
                data_source_id=data_source_id,
            )
            return dimension.name, members
        resolved = await asyncio.gather(
            *(resolve(item) for item in definition.dimensions)
        )
        return {name: members for name, members in resolved}

    @staticmethod
    async def _resolve_agent_domain(
        platform_access,
        *,
        domain_id: int, consumer_app_id: str,
        agent_id: UUID, agent_version_id: UUID,
    ) -> int:
        agent_domain_id = await platform_access.agent_domain_id(
            domain_id=domain_id, consumer_app_id=consumer_app_id,
            agent_id=agent_id, agent_version_id=agent_version_id,
        )
        if agent_domain_id is None:
            raise DataQueryRunError("AGENT_DOMAIN_NOT_CONFIGURED")
        return agent_domain_id

    async def get_run(
        self, *, data_query_run_id: UUID, domain_id: int, actor_id: str,
    ) -> DataQueryRunView:
        async with self._uow_factory() as uow:
            assert uow.runs and uow.results and uow.platform_access is not None
            run = await uow.runs.get_by_id(data_query_run_id=data_query_run_id)
            if run is None or run.domain_id != domain_id or run.actor_id != actor_id:
                raise DataQueryRunError("RUN_NOT_FOUND")
            await self._resolve_agent_domain(
                uow.platform_access,
                domain_id=domain_id,
                consumer_app_id=run.consumer_app_id,
                agent_id=run.agent_id,
                agent_version_id=run.agent_version_id,
            )
            result = await uow.results.get_available_by_run_id(
                data_query_run_id=data_query_run_id, now=datetime.now(UTC),
            )
            await uow.commit()
            return DataQueryRunView(
                data_query_run_id=run.data_query_run_id, status=run.status,
                error_code=run.error_code, result_available=result is not None,
            )

    async def get_result(
        self, *, data_query_run_id: UUID, domain_id: int, actor_id: str,
    ) -> DataQueryResultView:
        async with self._uow_factory() as uow:
            assert uow.runs and uow.results and uow.platform_access is not None
            run = await uow.runs.get_by_id(data_query_run_id=data_query_run_id)
            result = await uow.results.get_available_by_run_id(
                data_query_run_id=data_query_run_id, now=datetime.now(UTC),
            )
            if run is None or result is None or run.domain_id != domain_id or run.actor_id != actor_id:
                raise DataQueryRunError("RESULT_NOT_FOUND")
            await self._resolve_agent_domain(
                uow.platform_access,
                domain_id=domain_id,
                consumer_app_id=run.consumer_app_id,
                agent_id=run.agent_id,
                agent_version_id=run.agent_version_id,
            )
            await uow.commit()
            return DataQueryResultView(
                data_query_run_id=run.data_query_run_id,
                columns=tuple(result.columns_json), preview_rows=tuple(result.preview_rows_json),
                row_count=int(result.row_count), observed_row_count=int(result.observed_row_count),
                truncated=result.truncated,
                provenance={
                    "data_source_id": str(
                        run.semantic_model_snapshot_json["data_source_id"]
                    ),
                    "semantic_model_id": str(
                        run.semantic_model_snapshot_json["model_id"]
                    ),
                    "semantic_model_version": str(
                        run.semantic_model_snapshot_json["version"]
                    ),
                    "query_plan_hash": _hash(run.plan_snapshot_json),
                },
            )

    async def cancel_run(
        self, *, data_query_run_id: UUID, domain_id: int, actor_id: str,
    ) -> DataQueryRunView:
        async with self._uow_factory() as uow:
            assert uow.runs and uow.events and uow.platform_access is not None
            run = await uow.runs.get_by_id(data_query_run_id=data_query_run_id, lock=True)
            if run is None or run.domain_id != domain_id or run.actor_id != actor_id:
                raise DataQueryRunError("RUN_NOT_FOUND")
            await self._resolve_agent_domain(
                uow.platform_access,
                domain_id=domain_id,
                consumer_app_id=run.consumer_app_id,
                agent_id=run.agent_id,
                agent_version_id=run.agent_version_id,
            )
            if run.status in {"COMPLETED", "COMPLETED_EMPTY", "REJECTED", "FAILED", "TIMED_OUT", "CANCELLED"}:
                await uow.commit()
                return DataQueryRunView(data_query_run_id=run.data_query_run_id, status=run.status, error_code=run.error_code, result_available=run.status.startswith("COMPLETED"))
            run.cancel_requested_at = datetime.now(UTC)
            run.status = "CANCEL_PENDING" if run.status == "EXECUTING" else "CANCELLED"
            sequence = await uow.events.next_sequence_no(data_query_run_id=run.data_query_run_id)
            await uow.events.append(DataQueryEventEntity(
                domain_id=run.domain_id,
                data_query_run_id=run.data_query_run_id, sequence_no=sequence,
                event_type="RUN_CANCELLED" if run.status == "CANCELLED" else "cancel.requested",
                event_key=(
                    "data.query.cancelled"
                    if run.status == "CANCELLED"
                    else "data.query.cancel.requested"
                ),
                visibility="PUBLIC", payload_json={"status": run.status},
            ))
            await uow.commit()
            return DataQueryRunView(data_query_run_id=run.data_query_run_id, status=run.status, error_code=None, result_available=False)
