"""巡检报告展示模板；定义不允许携带 SQL 或脚本。"""

import hashlib
import json
import re
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.entities import InspectionReportTemplateEntity, InspectionReportTemplateVersionEntity
from platform_core.identity import uuid7


_FORBIDDEN_KEYS = {"sql", "query", "command", "tool", "script", "statement"}
_FORBIDDEN_TEXT = re.compile(r"(?:<\s*script\b|javascript\s*:|\b(?:select|insert|update|delete|merge|alter|drop|truncate|create|grant|revoke)\b[\s\S]{0,80}\b(?:from|into|set|table|on|to)\b)", re.IGNORECASE)


def _validate(value: Any, path: str = "definition") -> None:
    if isinstance(value, dict):
        if path == "definition" and not value: raise validation_failed("报告模板展示定义不能为空")
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS: raise validation_failed(f"报告模板禁止字段 {path}.{key}")
            _validate(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value): _validate(child, f"{path}[{index}]")
    elif isinstance(value, str) and (len(value) > 8000 or _FORBIDDEN_TEXT.search(value)):
        raise validation_failed(f"报告模板禁止超长文本、脚本或 SQL：{path}")


class InspectionReportTemplateService:
    def __init__(self, *, uow_factory): self._uow_factory = uow_factory

    async def list(self, *, domain_id: int):
        async with self._uow_factory() as uow:
            rows = await uow.inspections.list_report_templates(domain_id=domain_id)
            result = []
            for row in rows:
                version = await uow.inspections.get_report_template_version(template_version_id=row.current_version_id)
                if version is None: raise state_conflict("巡检报告模板当前版本不存在")
                result.append({**self._view(row), "version_no": int(version.version_no), "content_hash": version.content_hash})
            return result

    async def get(self, *, domain_id: int, template_id: UUID):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_report_template(domain_id=domain_id, template_id=template_id)
            if row is None: raise resource_not_found("InspectionReportTemplate")
            version = await uow.inspections.get_report_template_version(template_version_id=row.current_version_id)
            if version is None: raise state_conflict("巡检报告模板当前版本不存在")
            return {**self._view(row), "definition": dict(version.definition_json), "content_hash": version.content_hash, "version_no": int(version.version_no)}

    async def create(self, *, domain_id: int, actor_id: str, display_name: str, definition: dict[str, Any]):
        _validate(definition); display_name = display_name.strip()
        if not display_name: raise validation_failed("巡检报告模板名称不能为空")
        async with self._uow_factory() as uow:
            template_id, version_id = uuid7(), uuid7(); digest = self._hash(definition)
            row = InspectionReportTemplateEntity(template_id=template_id, domain_id=domain_id, display_name=display_name, status="ACTIVE", current_version_id=version_id, created_by=actor_id, updated_by=actor_id)
            version = InspectionReportTemplateVersionEntity(template_version_id=version_id, domain_id=domain_id, template_id=template_id, version_no=1, definition_json=definition, content_hash=digest, created_by=actor_id)
            await uow.inspections.add_report_template(row); await uow.inspections.add_report_template_version(version); await uow.commit()
            return {**self._view(row), "definition": definition, "content_hash": digest, "version_no": 1}

    async def create_version(self, *, domain_id: int, actor_id: str, template_id: UUID, expected_row_version: int, definition: dict[str, Any]):
        _validate(definition)
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_report_template(domain_id=domain_id, template_id=template_id, lock=True)
            if row is None: raise resource_not_found("InspectionReportTemplate")
            if int(row.row_version) != expected_row_version: raise state_conflict("巡检报告模板版本已变化")
            version_id, number, digest = uuid7(), await uow.inspections.next_report_template_version(template_id=template_id), self._hash(definition)
            await uow.inspections.add_report_template_version(InspectionReportTemplateVersionEntity(template_version_id=version_id, domain_id=domain_id, template_id=template_id, version_no=number, definition_json=definition, content_hash=digest, created_by=actor_id))
            row.current_version_id, row.updated_by = version_id, actor_id
            await uow.commit(); return {**self._view(row), "definition": definition, "content_hash": digest, "version_no": number}

    @staticmethod
    def _hash(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    @staticmethod
    def _view(row): return {"template_id": str(row.template_id), "domain_id": str(row.domain_id), "display_name": row.display_name, "status": row.status, "current_version_id": str(row.current_version_id), "row_version": int(row.row_version), "updated_at": row.updated_at.isoformat() if row.updated_at else None}
