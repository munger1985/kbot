"""确定性 Skill 选择、能力围栏和通用 Task 编译。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from aiops_agent.orchestration.blueprints import TaskSpec
from platform_core.contracts.aiops.conversation import DbaIntent
from platform_core.contracts.aiops.skills import (
    DbaCapabilitySnapshot,
    DbaIntentPlan,
    DbaSkillManifest,
    DbaSkillPlan,
    SkillPlanItem,
)

from .registry import DbaSkillRegistry


class SkillUnavailableError(ValueError):
    code = "AIOPS_SKILL_UNAVAILABLE"


class CapabilityUnavailableError(ValueError):
    code = "AIOPS_CAPABILITY_UNAVAILABLE"

    def __init__(self, skill_id: str, missing: tuple[str, ...]):
        super().__init__(
            f"Skill {skill_id} 缺少能力：{', '.join(missing)}"
        )
        self.skill_id = skill_id
        self.missing = missing


@dataclass(frozen=True, slots=True)
class CompiledSkillPlan:
    tasks: tuple[TaskSpec, ...]
    invocation_task_keys: tuple[str, ...]
    monitoring_task_keys: tuple[str, ...]


class DbaSkillPlanner:
    """只从冻结目录选择满足当前能力快照的最小 Skill 集合。"""

    def __init__(self, registry: DbaSkillRegistry) -> None:
        self._registry = registry

    def plan(
        self,
        *,
        intent: DbaIntentPlan,
        capabilities: DbaCapabilitySnapshot,
        suggested_skill_ids: Iterable[str] | None = None,
    ) -> DbaSkillPlan:
        overview = (
            intent.subject == "DATABASE_OVERVIEW"
            and intent.primary_intent in {DbaIntent.OBSERVE, DbaIntent.INSPECT}
        )
        candidate_ids = sorted(
            {
                manifest.skill_id
                for manifest in self._registry.manifests()
                if (
                    self._matches_overview(manifest, capabilities)
                    if overview
                    else self._matches_intent(manifest, intent, capabilities)
                )
            }
        )
        candidates = tuple(
            self._registry.latest(skill_id) for skill_id in candidate_ids
        )
        by_id = {manifest.skill_id: manifest for manifest in candidates}
        if suggested_skill_ids is None:
            available = [
                manifest
                for manifest in candidates
                if not self._missing_capabilities(manifest, capabilities)
            ]
            ordered = tuple(
                sorted(
                    available,
                    key=lambda item: (
                        item.limits.cost_units,
                        item.skill_id,
                        item.version,
                    ),
                )
            )
            # 数据库概览需要组合多个互补事实；单一专业问题仍坚持最小 Skill。
            selected = ordered if overview else ordered[:1]
            if not selected and candidates:
                unavailable = min(
                    candidates,
                    key=lambda item: (item.limits.cost_units, item.skill_id),
                )
                raise CapabilityUnavailableError(
                    unavailable.skill_id,
                    self._missing_capabilities(unavailable, capabilities),
                )
        else:
            requested = tuple(dict.fromkeys(suggested_skill_ids))
            unknown = tuple(value for value in requested if value not in by_id)
            if unknown:
                raise SkillUnavailableError(
                    f"计划包含目录外或不适用 Skill：{', '.join(unknown)}"
                )
            selected = tuple(by_id[value] for value in requested)
        if not selected:
            raise SkillUnavailableError(
                f"当前目录没有支持 {intent.primary_intent}/"
                f"{intent.primary_domain} 的可用 Skill"
            )

        items: list[SkillPlanItem] = []
        for ordinal, manifest in enumerate(selected, start=1):
            missing = self._missing_capabilities(manifest, capabilities)
            if missing:
                raise CapabilityUnavailableError(manifest.skill_id, missing)
            input_payload = dict(manifest.defaults)
            if intent.subject is not None:
                input_payload["subject"] = intent.subject
            if intent.time_window is not None:
                input_payload["time_window"] = intent.time_window.model_dump(
                    mode="json", exclude_none=True
                )
            if intent.requested_limit is not None:
                input_payload["limit"] = intent.requested_limit
            if intent.requested_order:
                input_payload["order"] = list(intent.requested_order)
            items.append(
                SkillPlanItem(
                    ordinal=ordinal,
                    skill_id=manifest.skill_id,
                    skill_version=manifest.version,
                    manifest_hash=self._registry.manifest_hash(
                        manifest.skill_id, manifest.version
                    ),
                    reason=(
                        "数据库综合概览所需的互补事实，且当前能力满足 Manifest"
                        if overview
                        else (
                            f"匹配 {intent.primary_intent}/"
                            f"{intent.primary_domain}，且当前能力满足 Manifest"
                        )
                    ),
                    evidence_question=(
                        f"{manifest.skill_id} 能为当前问题提供哪些可验证事实？"
                    ),
                    measurement_semantics=manifest.measurement_semantics,
                    input=input_payload,
                )
            )
        return DbaSkillPlan(
            catalog_hash=self._registry.catalog_hash,
            items=tuple(items),
        )

    @staticmethod
    def _matches_intent(
        manifest: DbaSkillManifest,
        intent: DbaIntentPlan,
        capabilities: DbaCapabilitySnapshot,
    ) -> bool:
        return (
            capabilities.database_type in manifest.database_types
            and DbaSkillPlanner._supports_database_version(
                manifest, capabilities.database_version
            )
            and intent.primary_intent in manifest.supported_intents
            and intent.primary_domain in manifest.domains
            and (
                not manifest.subjects
                or (
                    intent.subject is not None
                    and intent.subject in manifest.subjects
                )
            )
        )

    @staticmethod
    def _matches_overview(
        manifest: DbaSkillManifest,
        capabilities: DbaCapabilitySnapshot,
    ) -> bool:
        """为综合概览选择同库版本下所有可观测 Skill。"""
        return (
            capabilities.database_type in manifest.database_types
            and DbaSkillPlanner._supports_database_version(
                manifest, capabilities.database_version
            )
            and DbaIntent.OBSERVE in manifest.supported_intents
        )

    @staticmethod
    def _supports_database_version(
        manifest: DbaSkillManifest,
        database_version: str | None,
    ) -> bool:
        if database_version is None:
            return (
                manifest.version_range.minimum is None
                and manifest.version_range.maximum is None
            )
        match = re.search(r"\d+", database_version)
        if match is None:
            return False
        major = int(match.group(0))
        minimum = manifest.version_range.minimum
        maximum = manifest.version_range.maximum
        return (
            (minimum is None or major >= int(minimum))
            and (maximum is None or major <= int(maximum))
        )

    @staticmethod
    def _missing_capabilities(
        manifest: DbaSkillManifest,
        snapshot: DbaCapabilitySnapshot,
    ) -> tuple[str, ...]:
        available_sources = snapshot.available_source_capabilities
        target_capabilities = set(snapshot.target_capabilities)
        privileges = set(snapshot.privileges)
        entitlements = set(snapshot.entitlements)
        missing: list[str] = []
        if not snapshot.target_enabled:
            missing.append("TARGET_ENABLED")
        if manifest.required_target_capabilities and not snapshot.target_reachable:
            missing.append("TARGET_REACHABLE")
        missing.extend(
            f"SOURCE:{value}"
            for value in manifest.required_source_capabilities
            if value not in available_sources
        )
        missing.extend(
            f"TARGET:{value}"
            for value in manifest.required_target_capabilities
            if value not in target_capabilities
        )
        # 空清单表示权限尚未探测，而不是明确缺失。Tool执行器仍会把真实的
        # ORA-00942等权限错误转换成Evidence Gap；显式清单则用于提前拦截。
        if privileges:
            missing.extend(
                f"PRIVILEGE:{value}"
                for value in manifest.required_privileges
                if value not in privileges
            )
        missing.extend(
            f"ENTITLEMENT:{value}"
            for value in manifest.required_entitlements
            if value not in entitlements
        )
        return tuple(missing)


class SkillPlanCompiler:
    """将验证后的 Skill Plan 编译为通用运行内核 Task DAG。"""

    def __init__(self, registry: DbaSkillRegistry) -> None:
        self._registry = registry

    def compile(
        self,
        plan: DbaSkillPlan,
        *,
        monitoring_binding_ids: tuple[str, ...] = (),
    ) -> CompiledSkillPlan:
        if plan.catalog_hash != self._registry.catalog_hash:
            raise SkillUnavailableError("Skill Plan 的目录 Hash 已失效")
        task_keys: dict[int, str] = {}
        tasks: list[TaskSpec] = []
        for item in plan.items:
            manifest = self._registry.resolve(
                item.skill_id, item.skill_version
            )
            if item.manifest_hash != self._registry.manifest_hash(
                item.skill_id, item.skill_version
            ):
                raise SkillUnavailableError(
                    f"Skill Manifest Hash 已失效：{item.skill_id}"
                )
            task_key = f"skill:{item.ordinal}:{item.skill_id}"
            task_keys[item.ordinal] = task_key
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="SKILL_INVOKE",
                    handler_id="dba.skill.invoke",
                    handler_version="1",
                    input_schema_version=manifest.input_schema,
                    output_schema_version="DBA_SKILL_RESULT.v1",
                    depends_on=tuple(task_keys[value] for value in item.depends_on),
                    timeout_seconds=manifest.limits.timeout_seconds,
                    max_attempts=manifest.limits.max_attempts,
                    priority=50 + item.ordinal,
                )
            )
        invocation_keys = tuple(task.task_key for task in tasks)
        monitoring_keys: list[str] = []
        for binding_id in dict.fromkeys(monitoring_binding_ids):
            task_key = f"observe:{binding_id}"
            monitoring_keys.append(task_key)
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="SKILL_INVOKE",
                    handler_id="monitor.observe",
                    handler_version="1",
                    input_schema_version="MONITOR_OBSERVE_INPUT.v1",
                    output_schema_version="OBSERVATION_SET.v1",
                    timeout_seconds=120,
                    max_attempts=3,
                    priority=45,
                )
            )
        evidence_keys = (*invocation_keys, *monitoring_keys)
        tasks.append(
            TaskSpec(
                task_key="evidence:assess",
                task_type="EVIDENCE_ASSESS",
                handler_id="dba.evidence.assess",
                handler_version="1",
                input_schema_version="DBA_EVIDENCE_ASSESS_INPUT.v1",
                output_schema_version="DBA_SUFFICIENCY.v1",
                depends_on=evidence_keys,
                input_artifact_keys=evidence_keys,
                timeout_seconds=30,
                max_attempts=2,
                priority=90,
            )
        )
        tasks.append(
            TaskSpec(
                task_key="answer:compose",
                task_type="ANSWER",
                handler_id="dba.answer.compose",
                handler_version="1",
                input_schema_version="DBA_ANSWER_INPUT.v1",
                output_schema_version="AIOPS_TURN_RESULT.v1",
                depends_on=("evidence:assess",),
                input_artifact_keys=("evidence:assess",),
                timeout_seconds=120,
                max_attempts=2,
                priority=100,
            )
        )
        return CompiledSkillPlan(
            tasks=tuple(tasks),
            invocation_task_keys=invocation_keys,
            monitoring_task_keys=tuple(monitoring_keys),
        )
