"""运维知识资产、版本与发布审核 Repository。"""

from collections.abc import Callable, Sequence
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    OperationsKnowledgeAssetEntity,
    OperationsKnowledgeAssistantBaseEntity,
    OperationsKnowledgeAssistantBindingEntity,
    OperationsKnowledgeBaseEntity,
    OperationsKnowledgeIndexEntity,
    OperationsKnowledgeReviewEntity,
    OperationsKnowledgeScopeEntity,
    OperationsKnowledgeSourceEntity,
    OperationsKnowledgeTagEntity,
    OperationsKnowledgeVersionEntity,
    ReportEntity,
)
from aiops_agent.repositories._base import AIOpsRepository


class OperationsKnowledgeRepository(AIOpsRepository):
    def __init__(
        self,
        session: AsyncSession,
        assert_active: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(session, assert_active)

    async def add_knowledge_base(self, entity: OperationsKnowledgeBaseEntity) -> None:
        await self._add(entity)

    async def get_assistant_binding(
        self, *, domain_id: int, for_update: bool = False,
    ) -> OperationsKnowledgeAssistantBindingEntity | None:
        self._check_active()
        statement = select(OperationsKnowledgeAssistantBindingEntity).where(
            OperationsKnowledgeAssistantBindingEntity.domain_id == domain_id
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add_assistant_binding(
        self, entity: OperationsKnowledgeAssistantBindingEntity,
    ) -> None:
        await self._add(entity)

    async def assistant_knowledge_base_ids(self, *, domain_id: int) -> list[UUID]:
        self._check_active()
        rows = await self._session.scalars(
            select(OperationsKnowledgeAssistantBaseEntity.knowledge_base_id)
            .where(OperationsKnowledgeAssistantBaseEntity.domain_id == domain_id)
            .order_by(OperationsKnowledgeAssistantBaseEntity.knowledge_base_id)
        )
        return list(rows)

    async def replace_assistant_knowledge_bases(
        self, *, domain_id: int, knowledge_base_ids: tuple[UUID, ...],
    ) -> None:
        self._check_active()
        await self._session.execute(
            delete(OperationsKnowledgeAssistantBaseEntity).where(
                OperationsKnowledgeAssistantBaseEntity.domain_id == domain_id
            )
        )
        self._session.add_all(
            OperationsKnowledgeAssistantBaseEntity(
                domain_id=domain_id,
                knowledge_base_id=knowledge_base_id,
            )
            for knowledge_base_id in knowledge_base_ids
        )
        await self._session.flush()

    async def get_knowledge_base(
        self, *, domain_id: int, knowledge_base_id: UUID, for_update: bool = False,
    ) -> OperationsKnowledgeBaseEntity | None:
        self._check_active()
        statement = select(OperationsKnowledgeBaseEntity).where(
            OperationsKnowledgeBaseEntity.domain_id == domain_id,
            OperationsKnowledgeBaseEntity.knowledge_base_id == knowledge_base_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def list_knowledge_bases(
        self, *, domain_id: int, include_archived: bool = False,
    ) -> list[OperationsKnowledgeBaseEntity]:
        self._check_active()
        statement = select(OperationsKnowledgeBaseEntity).where(
            OperationsKnowledgeBaseEntity.domain_id == domain_id
        )
        if not include_archived:
            statement = statement.where(
                OperationsKnowledgeBaseEntity.status != "ARCHIVED"
            )
        statement = statement.order_by(
            OperationsKnowledgeBaseEntity.display_name,
            OperationsKnowledgeBaseEntity.knowledge_base_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def find_knowledge_base_name(
        self, *, domain_id: int, normalized_name: str,
        exclude_knowledge_base_id: UUID | None = None,
    ) -> OperationsKnowledgeBaseEntity | None:
        self._check_active()
        rows = await self.list_knowledge_bases(
            domain_id=domain_id, include_archived=True,
        )
        return next((
            row for row in rows
            if _normalize_name(row.display_name) == normalized_name
            and row.knowledge_base_id != exclude_knowledge_base_id
        ), None)

    async def knowledge_base_usage(
        self, *, knowledge_base_id: UUID,
    ) -> dict[str, int]:
        self._check_active()
        from aiops_agent.entities import (
            AIOpsAgentEntity,
            AIOpsAgentVersionKnowledgeBaseEntity,
        )

        assets = await self._session.scalar(
            select(func.count()).select_from(OperationsKnowledgeAssetEntity).where(
                OperationsKnowledgeAssetEntity.knowledge_base_id == knowledge_base_id
            )
        )
        agents = await self._session.scalar(
            select(func.count())
            .select_from(AIOpsAgentVersionKnowledgeBaseEntity)
            .join(
                AIOpsAgentEntity,
                AIOpsAgentEntity.current_version_id
                == AIOpsAgentVersionKnowledgeBaseEntity.agent_version_id,
            )
            .where(
                AIOpsAgentVersionKnowledgeBaseEntity.knowledge_base_id
                == knowledge_base_id
            )
        )
        assistants = await self._session.scalar(
            select(func.count()).select_from(
                OperationsKnowledgeAssistantBaseEntity
            ).where(
                OperationsKnowledgeAssistantBaseEntity.knowledge_base_id
                == knowledge_base_id
            )
        )
        return {
            "assets": int(assets or 0),
            "agent_versions": int(agents or 0),
            "knowledge_assistants": int(assistants or 0),
        }

    async def delete_knowledge_base(
        self, entity: OperationsKnowledgeBaseEntity,
    ) -> None:
        self._check_active()
        await self._session.delete(entity)
        await self._session.flush()

    async def add_asset_version(
        self,
        *,
        asset: OperationsKnowledgeAssetEntity,
        version: OperationsKnowledgeVersionEntity,
        scopes: Sequence[OperationsKnowledgeScopeEntity],
        sources: Sequence[OperationsKnowledgeSourceEntity],
        indexes: Sequence[OperationsKnowledgeIndexEntity],
    ) -> None:
        self._check_active()
        self._session.add(asset)
        await self._session.flush()
        self._session.add(version)
        await self._session.flush()
        self._session.add_all([*scopes, *sources, *indexes])
        await self._session.flush()

    async def add_version(
        self,
        *,
        version: OperationsKnowledgeVersionEntity,
        scopes: Sequence[OperationsKnowledgeScopeEntity],
        sources: Sequence[OperationsKnowledgeSourceEntity],
        indexes: Sequence[OperationsKnowledgeIndexEntity],
    ) -> None:
        self._check_active()
        self._session.add(version)
        await self._session.flush()
        self._session.add_all([*scopes, *sources, *indexes])
        await self._session.flush()

    async def list_tags(
        self, *, domain_id: int, status: str | None = None,
    ) -> list[OperationsKnowledgeTagEntity]:
        self._check_active()
        statement = select(OperationsKnowledgeTagEntity).where(
            OperationsKnowledgeTagEntity.domain_id == domain_id
        )
        if status is not None:
            statement = statement.where(OperationsKnowledgeTagEntity.status == status)
        statement = statement.order_by(
            OperationsKnowledgeTagEntity.category,
            OperationsKnowledgeTagEntity.sort_order,
            OperationsKnowledgeTagEntity.display_name,
            OperationsKnowledgeTagEntity.tag_id,
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_tag(
        self, *, domain_id: int, tag_id: UUID, for_update: bool = False,
    ) -> OperationsKnowledgeTagEntity | None:
        self._check_active()
        statement = select(OperationsKnowledgeTagEntity).where(
            OperationsKnowledgeTagEntity.domain_id == domain_id,
            OperationsKnowledgeTagEntity.tag_id == tag_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def find_tag_name(
        self, *, domain_id: int, category: str, normalized_name: str,
        exclude_tag_id: UUID | None = None,
    ) -> OperationsKnowledgeTagEntity | None:
        self._check_active()
        statement = select(OperationsKnowledgeTagEntity).where(
            OperationsKnowledgeTagEntity.domain_id == domain_id,
            OperationsKnowledgeTagEntity.category == category,
            OperationsKnowledgeTagEntity.normalized_name == normalized_name,
        )
        if exclude_tag_id is not None:
            statement = statement.where(OperationsKnowledgeTagEntity.tag_id != exclude_tag_id)
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def add_tag(self, tag: OperationsKnowledgeTagEntity) -> None:
        await self._add(tag)

    async def replace_tag_scopes(
        self, *, version: OperationsKnowledgeVersionEntity,
        tags: Sequence[OperationsKnowledgeTagEntity],
    ) -> None:
        """用审批结果整体替换手册标签，保留其他业务范围。"""
        self._check_active()
        await self._session.execute(delete(OperationsKnowledgeScopeEntity).where(
            OperationsKnowledgeScopeEntity.asset_version_id == version.asset_version_id,
            OperationsKnowledgeScopeEntity.tag_id.is_not(None),
        ))
        self._session.add_all([
            OperationsKnowledgeScopeEntity(
                asset_version_id=version.asset_version_id,
                tag_id=tag.tag_id,
                scope_kind=tag.category,
                scope_value=tag.display_name,
                normalized_value=str(tag.tag_id).upper(),
                source_kind="APPROVED_TAG",
                source_locator_json={"tag_id": str(tag.tag_id)},
            )
            for tag in tags
        ])
        await self._session.flush()

    async def find_by_source_hash(
        self, *, domain_id: int, asset_kind: str, source_hash: str,
        knowledge_base_id: UUID | None = None,
    ) -> tuple[OperationsKnowledgeAssetEntity, OperationsKnowledgeVersionEntity] | None:
        self._check_active()
        statement = (
            select(OperationsKnowledgeAssetEntity, OperationsKnowledgeVersionEntity)
            .join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_id
                == OperationsKnowledgeAssetEntity.asset_id,
            )
            .where(
                OperationsKnowledgeAssetEntity.domain_id == domain_id,
                OperationsKnowledgeAssetEntity.asset_kind == asset_kind,
                OperationsKnowledgeVersionEntity.source_hash == source_hash,
            )
            .order_by(OperationsKnowledgeVersionEntity.version_no.desc())
        )
        if knowledge_base_id is not None:
            statement = statement.where(
                OperationsKnowledgeAssetEntity.knowledge_base_id
                == knowledge_base_id
            )
        return (await self._session.execute(statement)).first()

    async def get_asset(
        self, *, domain_id: int, asset_id: UUID, for_update: bool = False
    ) -> OperationsKnowledgeAssetEntity | None:
        self._check_active()
        statement = select(OperationsKnowledgeAssetEntity).where(
            OperationsKnowledgeAssetEntity.domain_id == domain_id,
            OperationsKnowledgeAssetEntity.asset_id == asset_id,
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def get_version(
        self, *, domain_id: int, asset_version_id: UUID, for_update: bool = False
    ) -> OperationsKnowledgeVersionEntity | None:
        self._check_active()
        statement = (
            select(OperationsKnowledgeVersionEntity)
            .join(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeAssetEntity.asset_id
                == OperationsKnowledgeVersionEntity.asset_id,
            )
            .where(
                OperationsKnowledgeAssetEntity.domain_id == domain_id,
                OperationsKnowledgeVersionEntity.asset_version_id
                == asset_version_id,
            )
        )
        if for_update:
            statement = statement.with_for_update()
        return (await self._session.execute(statement)).scalar_one_or_none()

    async def next_version_no(self, *, asset_id: UUID) -> int:
        self._check_active()
        value = await self._session.scalar(
            select(func.max(OperationsKnowledgeVersionEntity.version_no)).where(
                OperationsKnowledgeVersionEntity.asset_id == asset_id
            )
        )
        return int(value or 0) + 1

    async def mark_cases_for_report_review(
        self, *, report_id: UUID, reviewer_id: str
    ) -> int:
        """来源报告被更正后，让关联案例立即退出新的候选集合。"""
        self._check_active()
        rows = list((await self._session.execute(
            select(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeVersionEntity,
            )
            .join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_id
                == OperationsKnowledgeAssetEntity.asset_id,
            )
            .join(
                OperationsKnowledgeSourceEntity,
                OperationsKnowledgeSourceEntity.asset_version_id
                == OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .where(
                OperationsKnowledgeSourceEntity.source_report_id == report_id,
                OperationsKnowledgeAssetEntity.asset_kind == "DIAGNOSIS_CASE",
                OperationsKnowledgeVersionEntity.status == "PUBLISHED",
            )
            .with_for_update()
        )).all())
        affected: set[UUID] = set()
        for asset, version in rows:
            if version.asset_version_id in affected:
                continue
            affected.add(version.asset_version_id)
            version.status = "REVIEW_REQUIRED"
            asset.status = "REVIEW_REQUIRED"
            if asset.current_version_id == version.asset_version_id:
                asset.current_version_id = None
            asset.updated_by = reviewer_id
            self._session.add(OperationsKnowledgeReviewEntity(
                asset_version_id=version.asset_version_id,
                decision="SOURCE_REPORT_CORRECTED",
                reviewer_id=reviewer_id,
                comment_text="来源正式报告已发布更正版，案例需重新审核",
                before_status="PUBLISHED",
                after_status="REVIEW_REQUIRED",
            ))
        await self._session.flush()
        return len(affected)

    async def list_assets(
        self,
        *,
        domain_id: int,
        asset_kind: str | None,
        status: str | None,
        knowledge_base_id: UUID | None = None,
        limit: int | None,
    ) -> list[OperationsKnowledgeAssetEntity]:
        self._check_active()
        statement = select(OperationsKnowledgeAssetEntity).where(
            OperationsKnowledgeAssetEntity.domain_id == domain_id
        )
        if asset_kind:
            statement = statement.where(
                OperationsKnowledgeAssetEntity.asset_kind == asset_kind
            )
        if status:
            statement = statement.where(OperationsKnowledgeAssetEntity.status == status)
        if knowledge_base_id is not None:
            statement = statement.where(
                OperationsKnowledgeAssetEntity.knowledge_base_id == knowledge_base_id
            )
        statement = statement.order_by(
            OperationsKnowledgeAssetEntity.updated_at.desc(),
            OperationsKnowledgeAssetEntity.asset_id,
        ).limit(limit)
        return list((await self._session.execute(statement)).scalars())

    async def list_versions(
        self, *, domain_id: int, asset_id: UUID
    ) -> list[OperationsKnowledgeVersionEntity]:
        self._check_active()
        statement = (
            select(OperationsKnowledgeVersionEntity)
            .join(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeAssetEntity.asset_id
                == OperationsKnowledgeVersionEntity.asset_id,
            )
            .where(
                OperationsKnowledgeAssetEntity.domain_id == domain_id,
                OperationsKnowledgeVersionEntity.asset_id == asset_id,
            )
            .order_by(OperationsKnowledgeVersionEntity.version_no.desc())
        )
        return list((await self._session.execute(statement)).scalars())

    async def get_version_relations(
        self, *, asset_version_id: UUID
    ) -> dict[str, list]:
        self._check_active()
        scopes = list((await self._session.execute(
            select(OperationsKnowledgeScopeEntity).where(
                OperationsKnowledgeScopeEntity.asset_version_id == asset_version_id
            ).order_by(
                OperationsKnowledgeScopeEntity.scope_kind,
                OperationsKnowledgeScopeEntity.normalized_value,
            )
        )).scalars())
        sources = list((await self._session.execute(
            select(OperationsKnowledgeSourceEntity).where(
                OperationsKnowledgeSourceEntity.asset_version_id == asset_version_id
            ).order_by(OperationsKnowledgeSourceEntity.created_at)
        )).scalars())
        indexes = list((await self._session.execute(
            select(OperationsKnowledgeIndexEntity).where(
                OperationsKnowledgeIndexEntity.asset_version_id == asset_version_id
            ).order_by(OperationsKnowledgeIndexEntity.created_at)
        )).scalars())
        reviews = list((await self._session.execute(
            select(OperationsKnowledgeReviewEntity).where(
                OperationsKnowledgeReviewEntity.asset_version_id == asset_version_id
            ).order_by(OperationsKnowledgeReviewEntity.created_at)
        )).scalars())
        return {"scopes": scopes, "sources": sources, "indexes": indexes, "reviews": reviews}

    async def add_review(self, review: OperationsKnowledgeReviewEntity) -> None:
        await self._add(review)

    async def list_reviews(
        self, *, domain_id: int, limit: int
    ) -> list[OperationsKnowledgeReviewEntity]:
        self._check_active()
        statement = (
            select(OperationsKnowledgeReviewEntity)
            .join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_version_id
                == OperationsKnowledgeReviewEntity.asset_version_id,
            )
            .join(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeAssetEntity.asset_id
                == OperationsKnowledgeVersionEntity.asset_id,
            )
            .where(OperationsKnowledgeAssetEntity.domain_id == domain_id)
            .order_by(OperationsKnowledgeReviewEntity.created_at.desc())
            .limit(limit)
        )
        return list((await self._session.execute(statement)).scalars())

    async def list_published_candidates(
        self,
        *,
        domain_id: int,
        target_id: UUID | None,
        source_kinds: Sequence[str],
        knowledge_base_ids: Sequence[UUID] = (),
        max_security_level: int,
        scope_filters: dict[str, Sequence[str]],
        limit: int,
    ) -> list[tuple[OperationsKnowledgeAssetEntity, OperationsKnowledgeVersionEntity, OperationsKnowledgeIndexEntity]]:
        """先按发布状态和确定性范围筛选，再交给 KC 检索正文。"""
        self._check_active()
        statement = (
            select(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeIndexEntity,
            )
            .join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_version_id
                == OperationsKnowledgeAssetEntity.current_version_id,
            )
            .join(
                OperationsKnowledgeIndexEntity,
                OperationsKnowledgeIndexEntity.asset_version_id
                == OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .where(
                OperationsKnowledgeAssetEntity.domain_id == domain_id,
                OperationsKnowledgeAssetEntity.status == "PUBLISHED",
                OperationsKnowledgeVersionEntity.status == "PUBLISHED",
                OperationsKnowledgeIndexEntity.index_status == "READY",
                OperationsKnowledgeAssetEntity.security_level <= max_security_level,
                ~OperationsKnowledgeVersionEntity.asset_version_id.in_(
                    select(OperationsKnowledgeSourceEntity.asset_version_id)
                    .join(
                        ReportEntity,
                        ReportEntity.report_id
                        == OperationsKnowledgeSourceEntity.source_report_id,
                    )
                    .where(ReportEntity.is_current == 0)
                ),
            )
        )
        if source_kinds:
            statement = statement.where(
                OperationsKnowledgeAssetEntity.asset_kind.in_(tuple(source_kinds))
            )
        if knowledge_base_ids:
            statement = statement.where(
                (
                    OperationsKnowledgeAssetEntity.asset_kind == "DIAGNOSIS_CASE"
                )
                | OperationsKnowledgeAssetEntity.knowledge_base_id.in_(
                    tuple(knowledge_base_ids)
                )
            )
        statement = statement.order_by(
            OperationsKnowledgeAssetEntity.asset_id,
            OperationsKnowledgeVersionEntity.asset_version_id,
        )
        candidate_rows = list((await self._session.execute(statement)).all())
        if not scope_filters:
            return candidate_rows[:limit] if limit is not None else candidate_rows
        result = []
        for row in candidate_rows:
            version = row[1]
            scopes = list((await self._session.execute(
                select(
                    OperationsKnowledgeScopeEntity.scope_kind,
                    OperationsKnowledgeScopeEntity.normalized_value,
                    OperationsKnowledgeScopeEntity.tag_id,
                ).where(
                    OperationsKnowledgeScopeEntity.asset_version_id
                    == version.asset_version_id
                )
            )).all())
            by_kind: dict[str, set[str]] = {}
            tag_ids: set[str] = set()
            for kind, value, tag_id in scopes:
                by_kind.setdefault(str(kind), set()).add(str(value))
                if tag_id is not None:
                    tag_ids.add(str(tag_id).upper())
            applicable = True
            target_scopes = by_kind.get("TARGET_ID", set())
            if target_scopes and (
                target_id is None or str(target_id).upper() not in target_scopes
            ):
                applicable = False
            for kind, requested in scope_filters.items():
                if kind == "TAG_ID":
                    wanted_ids = {str(value).strip().upper() for value in requested if str(value).strip()}
                    if wanted_ids and not wanted_ids.issubset(tag_ids):
                        applicable = False
                        break
                    continue
                known = by_kind.get(kind, set())
                wanted = {str(value).strip().upper() for value in requested if str(value).strip()}
                if known and wanted and known.isdisjoint(wanted):
                    applicable = False
                    break
            if applicable:
                result.append(row)
            if limit is not None and len(result) >= limit:
                break
        return result

    async def overview(self, *, domain_id: int) -> dict[str, int]:
        self._check_active()
        rows = (await self._session.execute(
            select(
                OperationsKnowledgeAssetEntity.asset_kind,
                OperationsKnowledgeAssetEntity.status,
                func.count(),
            )
            .where(OperationsKnowledgeAssetEntity.domain_id == domain_id)
            .group_by(
                OperationsKnowledgeAssetEntity.asset_kind,
                OperationsKnowledgeAssetEntity.status,
            )
        )).all()
        summary = {
            "published_manuals": 0,
            "published_cases": 0,
            "review_required": 0,
            "failed": 0,
            "index_abnormal": 0,
        }
        for kind, status, count in rows:
            if kind == "MANUAL" and status == "PUBLISHED":
                summary["published_manuals"] += int(count)
            if kind == "DIAGNOSIS_CASE" and status == "PUBLISHED":
                summary["published_cases"] += int(count)
            if status in {"DRAFT", "REVIEW_REQUIRED"}:
                summary["review_required"] += int(count)
            if status == "FAILED":
                summary["failed"] += int(count)
        abnormal = await self._session.scalar(
            select(func.count()).select_from(OperationsKnowledgeIndexEntity).join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_version_id
                == OperationsKnowledgeIndexEntity.asset_version_id,
            ).join(
                OperationsKnowledgeAssetEntity,
                OperationsKnowledgeAssetEntity.asset_id
                == OperationsKnowledgeVersionEntity.asset_id,
            ).where(
                OperationsKnowledgeAssetEntity.domain_id == domain_id,
                OperationsKnowledgeIndexEntity.index_status.not_in(("READY", "PROCESSING")),
            )
        )
        summary["index_abnormal"] = int(abnormal or 0)
        return summary

    async def next_processing_version(
        self,
    ) -> tuple[int, UUID] | None:
        """为对账循环选择一个处理中版本；外部调用不在本事务内执行。"""
        self._check_active()
        statement = (
            select(
                OperationsKnowledgeAssetEntity.domain_id,
                OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .join(
                OperationsKnowledgeVersionEntity,
                OperationsKnowledgeVersionEntity.asset_id
                == OperationsKnowledgeAssetEntity.asset_id,
            )
            .join(
                OperationsKnowledgeIndexEntity,
                OperationsKnowledgeIndexEntity.asset_version_id
                == OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .where(
                (
                    (OperationsKnowledgeVersionEntity.status == "TAGGING")
                    & (OperationsKnowledgeIndexEntity.index_status == "PENDING_APPROVAL")
                )
                | (
                    (OperationsKnowledgeVersionEntity.status == "PROCESSING")
                    & (OperationsKnowledgeIndexEntity.index_status == "PROCESSING")
                ),
            )
            .order_by(
                OperationsKnowledgeIndexEntity.updated_at,
                OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .limit(1)
        )
        row = (await self._session.execute(statement)).first()
        return (int(row[0]), row[1]) if row else None


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().casefold().split())
