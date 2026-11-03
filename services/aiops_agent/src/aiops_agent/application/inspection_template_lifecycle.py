"""巡检模板草稿、发布、默认选择与版本隔离。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.application.inspections.check_catalog import (
    compile_selected_check_steps, default_selected_check_ids, load_check_catalog, normalize_selected_check_ids,
)
from aiops_agent.application.inspections.daily_sections import normalize_daily_layout
from aiops_agent.entities import InspectionTemplateEntity, InspectionTemplateVersionEntity
from platform_core.identity import uuid7


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def metadata(row):
    return {"source_kind": "CUSTOM", "template_kind": "DAILY", "system_key": None,
            "description": "", "is_default": False, "draft": None, **(row.management_json or {})}


class InspectionTemplateService:
    """根记录保存工作草稿，发布历史与已绑定计划保持不可变。"""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    @staticmethod
    def _definition(*, display_name, selected_check_ids, schedule_type="DAILY", daily_report_layout=None):
        try:
            selected = normalize_selected_check_ids(selected_check_ids)
            return {
                "schema_version": "INSPECTION_TEMPLATE.v1", "display_name": display_name.strip(),
                "catalog_hash": load_check_catalog().catalog_hash,
                "selected_check_ids": list(selected),
                "evidence_steps": list(compile_selected_check_steps(selected, schedule_type=schedule_type)),
                **({"daily_report_layout": normalize_daily_layout(daily_report_layout)} if schedule_type != "MONTHLY" else {}),
            }
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc

    @staticmethod
    def _summary(row, version):
        meta = metadata(row)
        definition = dict(version.definition_json) if version else {}
        draft = meta["draft"]
        return {
            "inspection_template_id": str(row.inspection_template_id),
            "inspection_template_version_id": str(row.current_version_id) if row.current_version_id else None,
            "display_name": row.display_name, "status": row.status,
            "source_kind": meta["source_kind"], "template_kind": meta["template_kind"],
            "system_key": meta["system_key"], "description": meta["description"],
            "is_default": meta["is_default"], "has_draft": draft is not None,
            "has_unpublished_changes": bool(draft and content_hash(draft) != (version.content_hash if version else None)),
            "version_no": int(version.version_no) if version else None,
            "selected_check_ids": list(definition.get("selected_check_ids") or ()),
            "daily_report_layout": definition.get("daily_report_layout"),
            "content_hash": version.content_hash if version else None,
            "row_version": int(row.row_version), "updated_at": row.updated_at.isoformat(),
        }

    async def _version(self, uow, row):
        if row.current_version_id is None:
            return None
        version = await uow.inspections.get_inspection_template_version(inspection_template_version_id=row.current_version_id)
        if version is None or version.inspection_template_id != row.inspection_template_id:
            raise state_conflict("巡检模板当前版本不存在")
        return version

    async def _locked(self, uow, domain_id, template_id, expected_row_version, *, custom=False):
        row = await uow.inspections.get_inspection_template(domain_id=domain_id, inspection_template_id=template_id, lock=True)
        if row is None:
            raise resource_not_found("InspectionTemplate")
        if int(row.row_version) != expected_row_version:
            raise state_conflict("模板已被修改，请刷新后重试")
        if custom and metadata(row)["source_kind"] == "BUILTIN":
            raise state_conflict("系统内置模板只读，请基于此创建自定义模板")
        return row

    async def list(self, *, domain_id, actor_id, **entity_scope):
        async with self._uow_factory() as uow:
            await uow.inspections.lock_inspection_template_defaults(domain_id=domain_id)
            rows = await uow.inspections.list_inspection_templates(domain_id=domain_id)
            added = {}
            for name, kind, key in (("日常巡检", "DAILY", "inspection.daily"), ("月度巡检", "MONTHLY", "inspection.monthly")):
                if any(metadata(row)["system_key"] == key for row in rows):
                    continue
                definition = self._definition(display_name=name, selected_check_ids=default_selected_check_ids(kind), schedule_type=kind)
                version_id = uuid7()
                row = InspectionTemplateEntity(
                    **entity_scope, domain_id=domain_id, inspection_template_id=uuid7(), display_name=name,
                    status="ACTIVE", current_version_id=version_id, created_by=actor_id, updated_by=actor_id,
                    management_json={"source_kind": "BUILTIN", "template_kind": kind, "system_key": key,
                                     "is_default": not any(metadata(item)["template_kind"] == kind and metadata(item)["is_default"] for item in rows)},
                )
                version = InspectionTemplateVersionEntity(
                    **entity_scope, domain_id=domain_id, inspection_template_id=row.inspection_template_id,
                    inspection_template_version_id=version_id, version_no=1, definition_json=definition,
                    content_hash=content_hash(definition), created_by=actor_id,
                )
                await uow.inspections.add_inspection_template(row)
                await uow.inspections.add_inspection_template_version(version)
                rows.append(row)
                added[version_id] = version
            counts = await uow.inspections.count_plans_by_inspection_templates(domain_id=domain_id, template_ids=[row.inspection_template_id for row in rows])
            result = []
            for row in rows:
                version = added.get(row.current_version_id) or await self._version(uow, row)
                result.append({**self._summary(row, version), "usage_count": counts.get(row.inspection_template_id, 0)})
            if added:
                await uow.commit()
            return result

    async def get(self, *, domain_id, template_id):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_inspection_template(domain_id=domain_id, inspection_template_id=template_id)
            if row is None:
                raise resource_not_found("InspectionTemplate")
            version = await self._version(uow, row)
            return {**self._summary(row, version), "definition": dict(version.definition_json) if version else None,
                    "draft": metadata(row)["draft"]}

    async def create(self, *, domain_id, actor_id, display_name, selected_check_ids, daily_report_layout=None, **entity_scope):
        if not display_name.strip():
            raise validation_failed("巡检模板名称不能为空")
        definition = self._definition(display_name=display_name, selected_check_ids=selected_check_ids, daily_report_layout=daily_report_layout)
        async with self._uow_factory() as uow:
            row = InspectionTemplateEntity(
                **entity_scope, domain_id=domain_id, inspection_template_id=uuid7(), display_name=display_name.strip(),
                status="ACTIVE", current_version_id=None, created_by=actor_id, updated_by=actor_id,
                management_json={"source_kind": "CUSTOM", "template_kind": "DAILY", "draft": definition},
            )
            await uow.inspections.add_inspection_template(row)
            await uow.commit()
            return {**self._summary(row, None), "draft": definition, "definition": None}

    async def save_draft(self, *, domain_id, actor_id, template_id, expected_row_version, display_name, selected_check_ids, daily_report_layout=None):
        if not display_name.strip():
            raise validation_failed("巡检模板名称不能为空")
        async with self._uow_factory() as uow:
            row = await self._locked(uow, domain_id, template_id, expected_row_version, custom=True)
            if metadata(row)["template_kind"] != "DAILY":
                raise validation_failed("当前编辑器只管理日报模板")
            definition = self._definition(display_name=display_name, selected_check_ids=selected_check_ids, daily_report_layout=daily_report_layout)
            row.management_json = {**metadata(row), "draft": definition}
            if row.current_version_id is None:
                row.display_name = display_name.strip()
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            version = await self._version(uow, row)
            await uow.commit()
            return {**self._summary(row, version), "draft": definition, "definition": dict(version.definition_json) if version else None}

    async def publish(self, *, domain_id, actor_id, template_id, expected_row_version, **entity_scope):
        async with self._uow_factory() as uow:
            row = await self._locked(uow, domain_id, template_id, expected_row_version, custom=True)
            meta = metadata(row)
            draft = meta["draft"]
            if not draft:
                raise state_conflict("没有可发布的草稿")
            definition = self._definition(display_name=draft["display_name"], selected_check_ids=draft["selected_check_ids"], daily_report_layout=draft.get("daily_report_layout"))
            digest = content_hash(definition)
            versions = await uow.inspections.list_inspection_template_versions(inspection_template_id=template_id)
            if any(version.content_hash == digest for version in versions):
                raise state_conflict("内容与已发布版本相同，无需重复发布")
            version = InspectionTemplateVersionEntity(
                **entity_scope, domain_id=domain_id, inspection_template_id=template_id,
                inspection_template_version_id=uuid7(), version_no=max((int(item.version_no) for item in versions), default=0) + 1,
                definition_json=definition, content_hash=digest, created_by=actor_id,
            )
            await uow.inspections.add_inspection_template_version(version)
            row.current_version_id = version.inspection_template_version_id
            row.display_name = definition["display_name"]
            row.management_json = {**meta, "draft": None}
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            await uow.commit()
            return self._summary(row, version)

    async def discard_draft(self, *, domain_id, actor_id, template_id, expected_row_version):
        async with self._uow_factory() as uow:
            row = await self._locked(uow, domain_id, template_id, expected_row_version, custom=True)
            if row.current_version_id is None:
                counts = await uow.inspections.count_plans_by_inspection_templates(domain_id=domain_id, template_ids=[template_id])
                versions = await uow.inspections.list_inspection_template_versions(inspection_template_id=template_id)
                if counts.get(template_id, 0) or versions:
                    raise state_conflict("模板仍有引用或发布历史，不能删除")
                await uow.inspections.delete_inspection_template(row)
            else:
                row.management_json = {**metadata(row), "draft": None}
                row.updated_by = actor_id
                row.updated_at = datetime.now(UTC)
            await uow.commit()
            return {"discarded": True}

    async def set_default(self, *, domain_id, actor_id, template_id, expected_row_version):
        async with self._uow_factory() as uow:
            await uow.inspections.lock_inspection_template_defaults(domain_id=domain_id)
            row = await self._locked(uow, domain_id, template_id, expected_row_version)
            if row.status != "ACTIVE" or not row.current_version_id:
                raise state_conflict("只能将已发布且启用的模板设为新计划默认")
            kind = metadata(row)["template_kind"]
            for item in await uow.inspections.list_inspection_templates(domain_id=domain_id, lock=True):
                meta = metadata(item)
                if meta["template_kind"] == kind and (meta["is_default"] or item.inspection_template_id == template_id):
                    item.management_json = {**meta, "is_default": item.inspection_template_id == template_id}
                    item.updated_by = actor_id
                    item.updated_at = datetime.now(UTC)
            version = await self._version(uow, row)
            await uow.commit()
            return self._summary(row, version)

    async def update(self, *, domain_id, actor_id, template_id, expected_row_version, status):
        if status not in {"ACTIVE", "INACTIVE"}:
            raise validation_failed("巡检模板状态无效")
        async with self._uow_factory() as uow:
            await uow.inspections.lock_inspection_template_defaults(domain_id=domain_id)
            row = await self._locked(uow, domain_id, template_id, expected_row_version, custom=True)
            if status == "INACTIVE" and metadata(row)["is_default"]:
                raise state_conflict("请先指定替代的新计划默认模板，再停用此模板")
            if not row.current_version_id:
                raise state_conflict("未发布模板请继续编辑或放弃草稿")
            row.status = status
            row.updated_by = actor_id
            row.updated_at = datetime.now(UTC)
            version = await self._version(uow, row)
            await uow.commit()
            return self._summary(row, version)

    async def list_versions(self, *, domain_id, template_id):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_inspection_template(domain_id=domain_id, inspection_template_id=template_id)
            if row is None:
                raise resource_not_found("InspectionTemplate")
            versions = await uow.inspections.list_inspection_template_versions(inspection_template_id=template_id)
            return [{"template_version_id": str(item.inspection_template_version_id), "version_no": int(item.version_no),
                     "definition": dict(item.definition_json), "content_hash": item.content_hash,
                     "created_at": item.created_at.isoformat(), "created_by": str(item.created_by),
                     "current": item.inspection_template_version_id == row.current_version_id} for item in versions]

    async def list_plans(self, *, domain_id, template_id):
        async with self._uow_factory() as uow:
            row = await uow.inspections.get_inspection_template(domain_id=domain_id, inspection_template_id=template_id)
            if row is None:
                raise resource_not_found("InspectionTemplate")
            plans = await uow.inspections.list_plans_by_inspection_template(domain_id=domain_id, template_id=template_id)
            versions = await uow.inspections.list_inspection_template_versions(inspection_template_id=template_id)
            numbers = {item.inspection_template_version_id: int(item.version_no) for item in versions}
            return [{"inspection_plan_id": str(plan.inspection_plan_id), "display_name": plan.display_name,
                     "status": plan.status, "template_version_no": numbers.get(plan.inspection_template_version_id),
                     "is_latest": plan.inspection_template_version_id == row.current_version_id} for plan in plans]
