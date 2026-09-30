"""运维知识资产、版本与发布审核 Repository。"""

from collections.abc import Callable, Sequence
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from aiops_agent.entities import (
    OperationsKnowledgeAssetEntity,
    OperationsKnowledgeIndexEntity,
    OperationsKnowledgeReviewEntity,
    OperationsKnowledgeScopeEntity,
    OperationsKnowledgeSourceEntity,
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
        self._session.add(version)
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
        self._session.add_all([*scopes, *sources, *indexes])
        await self._session.flush()

    async def find_by_source_hash(
        self, *, domain_id: int, asset_kind: str, source_hash: str
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
        limit: int,
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
        candidate_rows = list((await self._session.execute(
            statement.order_by(
                OperationsKnowledgeAssetEntity.updated_at.desc(),
                OperationsKnowledgeAssetEntity.asset_id,
            ).limit(limit * 4)
        )).all())
        if not scope_filters:
            return candidate_rows[:limit]
        result = []
        for row in candidate_rows:
            version = row[1]
            scopes = list((await self._session.execute(
                select(
                    OperationsKnowledgeScopeEntity.scope_kind,
                    OperationsKnowledgeScopeEntity.normalized_value,
                ).where(
                    OperationsKnowledgeScopeEntity.asset_version_id
                    == version.asset_version_id
                )
            )).all())
            by_kind: dict[str, set[str]] = {}
            for kind, value in scopes:
                by_kind.setdefault(str(kind), set()).add(str(value))
            applicable = True
            target_scopes = by_kind.get("TARGET_ID", set())
            if target_scopes and (
                target_id is None or str(target_id).upper() not in target_scopes
            ):
                applicable = False
            for kind, requested in scope_filters.items():
                known = by_kind.get(kind, set())
                wanted = {str(value).strip().upper() for value in requested if str(value).strip()}
                if known and wanted and known.isdisjoint(wanted):
                    applicable = False
                    break
            if applicable:
                result.append(row)
            if len(result) >= limit:
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
                OperationsKnowledgeVersionEntity.status == "PROCESSING",
                OperationsKnowledgeIndexEntity.index_status == "PROCESSING",
            )
            .order_by(
                OperationsKnowledgeIndexEntity.updated_at,
                OperationsKnowledgeVersionEntity.asset_version_id,
            )
            .limit(1)
        )
        row = (await self._session.execute(statement)).first()
        return (int(row[0]), row[1]) if row else None
