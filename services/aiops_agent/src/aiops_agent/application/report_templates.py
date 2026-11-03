"""巡检内容模板与会话报告展示模板服务。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import (
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.application.inspection_template_lifecycle import InspectionTemplateService
from aiops_agent.application.reporting import (
    ReportTemplate,
    resolve_system_template,
    template_summary,
    validate_template_definition,
)
from aiops_agent.entities import (
    SessionReportTemplateEntity,
    SessionReportTemplateVersionEntity,
)
from platform_core.identity import uuid7


_FORBIDDEN_KEYS = {"sql", "query", "command", "tool", "script", "statement"}
_FORBIDDEN_TEXT = re.compile(
    r"(?:<\s*script\b|javascript\s*:|\b(?:select|insert|update|delete|merge|alter|drop|truncate|create|grant|revoke)\b[\s\S]{0,80}\b(?:from|into|set|table|on|to)\b)",
    re.IGNORECASE,
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_safe_definition(value: Any, path: str = "definition") -> None:
    if isinstance(value, dict):
        if path == "definition" and not value:
            raise validation_failed("模板定义不能为空")
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise validation_failed(f"模板禁止字段 {path}.{key}")
            _validate_safe_definition(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_safe_definition(child, f"{path}[{index}]")
    elif isinstance(value, str) and (
        len(value) > 8000 or _FORBIDDEN_TEXT.search(value)
    ):
        raise validation_failed(f"模板禁止超长文本、脚本或 SQL：{path}")


class SessionReportTemplateService:
    """管理 Session 正式报告的章节和展示顺序。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    @staticmethod
    def _normalize_definition(
        *, display_name: str, definition: dict[str, Any]
    ) -> tuple[dict[str, Any], ReportTemplate]:
        _validate_safe_definition(definition)
        sections = [
            dict(item)
            for item in definition.get("sections") or ()
            if isinstance(item, dict) and item.get("kind")
        ]
        kinds = [str(item["kind"]) for item in sections]
        if len(kinds) != len(set(kinds)):
            raise validation_failed("会话报告模板章节不能重复")
        if "EXECUTIVE_SUMMARY" not in kinds:
            sections.insert(0, {"kind": "EXECUTIVE_SUMMARY"})
        if "EVIDENCE_BOUNDARY" not in kinds:
            sections.append({"kind": "EVIDENCE_BOUNDARY"})
        if sections[0]["kind"] != "EXECUTIVE_SUMMARY":
            raise validation_failed("执行摘要必须是会话报告模板的第一章节")
        if sections[-1]["kind"] != "EVIDENCE_BOUNDARY":
            raise validation_failed("证据边界必须是会话报告模板的最后章节")
        normalized = {
            **definition,
            "schema_version": "SESSION_REPORT_TEMPLATE.v1",
            "display_name": display_name,
            "applicable_source_kinds": ["CHAT"],
            "allowed_period_kinds": ["AD_HOC"],
            "sections": sections,
        }
        return normalized, validate_template_definition(normalized)

    async def list(self, *, domain_id: int):
        standard = resolve_system_template("system:diagnosis.standard")
        async with self._uow_factory() as uow:
            rows = await uow.inspections.list_session_report_templates(
                domain_id=domain_id
            )
            template_refs = [
                "system:diagnosis.standard",
                *(f"domain:{row.template_id}" for row in rows),
            ]
            usage_counts = await uow.inspections.count_reports_by_template_refs(
                domain_id=domain_id,
                template_refs=template_refs,
            )
            custom_default_exists = any(
                row.status == "ACTIVE" and int(row.is_default) == 1
                for row in rows
            )
            result = []
            if standard is not None:
                result.append({
                    **template_summary(standard),
                    "status": "ACTIVE",
                    "is_default": not custom_default_exists,
                    "usage_count": usage_counts.get(standard.template_ref, 0),
                    "updated_at": None,
                })
            for row in rows:
                version = await uow.inspections.get_session_report_template_version(
                    template_version_id=row.current_version_id
                )
                if version is None:
                    raise state_conflict("会话报告模板当前版本不存在")
                definition = dict(version.definition_json)
                definition.setdefault("display_name", row.display_name)
                template = validate_template_definition(definition)
                result.append({
                    **self._view(row),
                    **template_summary(ReportTemplate(
                        template_ref=f"domain:{row.template_id}",
                        version=str(version.version_no),
                        display_name=row.display_name,
                        applicable_source_kinds=("CHAT",),
                        allowed_period_kinds=("AD_HOC",),
                        sections=template.sections,
                        definition=definition,
                    )),
                    "template_version_id": str(version.template_version_id),
                    "version_no": int(version.version_no),
                    "content_hash": version.content_hash,
                    "usage_count": usage_counts.get(
                        f"domain:{row.template_id}", 0
                    ),
                })
            return result

    async def get(self, *, domain_id: int, template_id: UUID):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id
            )
            if row is None:
                raise resource_not_found("SessionReportTemplate")
            version = await uow.inspections.get_session_report_template_version(
                template_version_id=row.current_version_id
            )
            if version is None:
                raise state_conflict("会话报告模板当前版本不存在")
            return {
                **self._view(row),
                "template_ref": f"domain:{row.template_id}",
                "template_version_id": str(version.template_version_id),
                "definition": dict(version.definition_json),
                "content_hash": version.content_hash,
                "version_no": int(version.version_no),
            }

    async def list_versions(self, *, domain_id: int, template_id: UUID):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id
            )
            if row is None:
                raise resource_not_found("SessionReportTemplate")
            versions = (
                await uow.inspections.list_session_report_template_versions(
                    template_id=template_id
                )
            )
            return [
                {
                    "template_version_id": str(version.template_version_id),
                    "version_no": int(version.version_no),
                    "content_hash": version.content_hash,
                    "definition": dict(version.definition_json),
                    "created_by": str(version.created_by),
                    "created_at": version.created_at.isoformat(),
                    "current": (
                        version.template_version_id == row.current_version_id
                    ),
                }
                for version in versions
            ]

    async def create(
        self, *, domain_id: int, actor_id: str,
        display_name: str, definition: dict[str, Any],
    ):
        display_name = display_name.strip()
        if not display_name:
            raise validation_failed("会话报告模板名称不能为空")
        definition, _ = self._normalize_definition(
            display_name=display_name, definition=definition
        )
        template_id, version_id = uuid7(), uuid7()
        async with self._uow_factory() as uow:
            row = SessionReportTemplateEntity(
                template_id=template_id, domain_id=domain_id,
                display_name=display_name, status="ACTIVE",
                is_default=0,
                current_version_id=version_id,
                created_by=actor_id, updated_by=actor_id,
            )
            version = SessionReportTemplateVersionEntity(
                template_version_id=version_id, domain_id=domain_id,
                template_id=template_id, version_no=1,
                definition_json=definition, content_hash=_hash(definition),
                created_by=actor_id,
            )
            await uow.inspections.add_session_report_template(row)
            await uow.inspections.add_session_report_template_version(version)
            await uow.commit()
            return {
                **self._view(row), "definition": definition,
                "template_ref": f"domain:{row.template_id}",
                "template_version_id": str(version.template_version_id),
                "content_hash": version.content_hash, "version_no": 1,
            }

    async def create_version(
        self, *, domain_id: int, actor_id: str, template_id: UUID,
        expected_row_version: int, definition: dict[str, Any],
    ):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id, lock=True
            )
            if row is None:
                raise resource_not_found("SessionReportTemplate")
            if int(row.row_version) != expected_row_version:
                raise state_conflict("会话报告模板版本已变化")
            definition, _ = self._normalize_definition(
                display_name=row.display_name, definition=definition
            )
            version = SessionReportTemplateVersionEntity(
                template_version_id=uuid7(), domain_id=domain_id,
                template_id=template_id,
                version_no=(await uow.inspections.next_session_report_template_version(template_id=template_id)),
                definition_json=definition, content_hash=_hash(definition),
                created_by=actor_id,
            )
            await uow.inspections.add_session_report_template_version(version)
            row.current_version_id = version.template_version_id
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            await uow.commit()
            return {
                **self._view(row), "definition": definition,
                "template_ref": f"domain:{row.template_id}",
                "template_version_id": str(version.template_version_id),
                "content_hash": version.content_hash,
                "version_no": int(version.version_no),
            }

    async def update(
        self,
        *,
        domain_id: int,
        actor_id: str,
        template_id: UUID,
        expected_row_version: int,
        status: str,
    ):
        if status not in {"ACTIVE", "INACTIVE"}:
            raise validation_failed("会话报告模板状态无效")
        async with self._uow_factory() as uow:
            if not await uow.inspections.lock_domain(domain_id=domain_id):
                raise resource_not_found("Domain")
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id, lock=True
            )
            if row is None:
                raise resource_not_found("SessionReportTemplate")
            if int(row.row_version) != expected_row_version:
                raise state_conflict("会话报告模板版本已变化")
            row.status = status
            if status == "INACTIVE":
                row.is_default = 0
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            await uow.commit()
            return self._view(row)

    async def set_default(
        self, *, domain_id: int, actor_id: str, template_ref: str
    ):
        async with self._uow_factory() as uow:
            if not await uow.inspections.lock_domain(domain_id=domain_id):
                raise resource_not_found("Domain")
            if template_ref == "system:diagnosis.standard":
                await uow.inspections.clear_default_session_report_template(
                    domain_id=domain_id
                )
                await uow.commit()
                return {"template_ref": template_ref, "is_default": True}
            if not template_ref.startswith("domain:"):
                raise validation_failed("会话报告模板引用无效")
            try:
                template_id = UUID(template_ref.removeprefix("domain:"))
            except ValueError as exc:
                raise validation_failed("会话报告模板引用无效") from exc
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id, lock=True
            )
            if row is None:
                raise resource_not_found("SessionReportTemplate")
            if row.status != "ACTIVE":
                raise state_conflict("停用的会话报告模板不能设为默认")
            await uow.inspections.clear_default_session_report_template(
                domain_id=domain_id,
                keep_template_id=template_id,
            )
            row.is_default = 1
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            await uow.commit()
            return {**self._view(row), "template_ref": template_ref}

    async def resolve(
        self,
        *,
        domain_id: int,
        template_ref: str,
        template_version_id: UUID | None = None,
    ) -> ReportTemplate:
        system = resolve_system_template(template_ref)
        if system is not None:
            if template_version_id is not None:
                raise validation_failed("系统报告模板不接受版本标识")
            if "CHAT" not in system.applicable_source_kinds:
                raise validation_failed("会话报告模板引用无效")
            return system
        if not template_ref.startswith("domain:"):
            raise validation_failed("会话报告模板引用无效")
        try:
            template_id = UUID(template_ref.removeprefix("domain:"))
        except ValueError as exc:
            raise validation_failed("会话报告模板引用无效") from exc
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_session_report_template(
                domain_id=domain_id, template_id=template_id
            )
            if row is None or row.status != "ACTIVE":
                raise resource_not_found("SessionReportTemplate")
            version = await uow.inspections.get_session_report_template_version(
                template_version_id=template_version_id or row.current_version_id
            )
            if (
                version is None
                or version.template_id != row.template_id
                or int(version.domain_id) != domain_id
            ):
                raise state_conflict("会话报告模板版本不存在或不属于该模板")
            definition = dict(version.definition_json)
            definition.setdefault("display_name", row.display_name)
            template = validate_template_definition(definition)
            return ReportTemplate(
                template_ref=template_ref, version=str(version.version_no),
                display_name=row.display_name,
                applicable_source_kinds=("CHAT",),
                allowed_period_kinds=("AD_HOC",),
                sections=template.sections, definition=definition,
            )

    @staticmethod
    def _view(row):
        return {
            "template_id": str(row.template_id),
            "domain_id": int(row.domain_id),
            "display_name": row.display_name,
            "status": row.status,
            "is_default": int(getattr(row, "is_default", 0)) == 1,
            "current_version_id": str(row.current_version_id),
            "row_version": int(row.row_version),
            "created_at": (
                row.created_at.isoformat()
                if getattr(row, "created_at", None) else None
            ),
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }
