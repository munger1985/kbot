"""把 Skill Plan 编译为可供 DB Executor 审计重放的冻结快照。"""

from __future__ import annotations

import re

from aiops_agent.diagnostics import DiagnosticRegistry
from platform_core.contracts.aiops.skills import (
    DbaCapabilitySnapshot,
    DbaSkillPlan,
)

from .planner import CompiledSkillPlan
from .registry import DbaSkillRegistry, SkillCatalogError, canonical_hash


class SkillExecutionSnapshotBuilder:
    def __init__(
        self,
        *,
        skill_registry: DbaSkillRegistry,
        diagnostic_registry: DiagnosticRegistry,
    ) -> None:
        self._skills = skill_registry
        self._tools = diagnostic_registry

    def validate_catalog(self) -> None:
        """离线验证所有 Skill 对 Tool 版本和要求的声明完整性。"""
        for manifest in self._skills.manifests():
            version = manifest.version_range.minimum or "0"
            for database_type in manifest.database_types:
                for step in manifest.tool_dag:
                    try:
                        resolved = self._tools.resolve(
                            tool_id=step.tool_id,
                            tool_version=step.tool_version,
                            db_type=str(database_type),
                            db_version=version,
                            capabilities=set(
                                manifest.required_target_capabilities
                            ),
                            entitlements=set(
                                manifest.required_entitlements
                            ),
                        )
                    except (LookupError, ValueError) as exc:
                        raise SkillCatalogError(
                            f"Skill {manifest.skill_id} 的 Tool 无法精确解析："
                            f"{step.tool_id}@{step.tool_version}"
                        ) from exc
                    definition = resolved.definition
                    self._validate_manifest_requirements(
                        manifest=manifest,
                        tool_id=definition.tool_id,
                        required_capabilities=set(
                            definition.required_capabilities
                        ),
                        required_entitlements=set(
                            definition.required_entitlements
                        ),
                        required_privileges=set(
                            definition.required_privileges
                        ),
                    )

    def discover_tools(
        self, capabilities: DbaCapabilitySnapshot
    ) -> tuple[dict, ...]:
        """直接从原子Tool目录发现能力，Playbook不参与准入。"""
        version = capabilities.database_version or ""
        match = re.search(r"\d+", version)
        if match is None:
            return ()
        major = int(match.group(0))
        configured_privileges = set(capabilities.privileges)
        discovered = []
        for item in self._tools.tools:
            definition = item.definition
            if definition.db_type != str(capabilities.database_type):
                continue
            if not (
                definition.supported_version_min
                <= major
                < definition.supported_version_max_exclusive
            ):
                continue
            if not set(definition.required_capabilities) <= set(
                capabilities.target_capabilities
            ):
                continue
            if not set(definition.required_entitlements) <= set(
                capabilities.entitlements
            ):
                continue
            if configured_privileges and not set(
                definition.required_privileges
            ) <= configured_privileges:
                continue
            discovered.append(
                {
                    "tool_id": definition.tool_id,
                    "version": definition.version,
                    "tool_class": "ORACLE_SQL",
                    "description": f"受控只读数据库观测：{definition.tool_id}",
                    "input": {
                        parameter.name: {
                            "type": parameter.type,
                            "required": parameter.required,
                            "default": parameter.default,
                        }
                        for parameter in definition.parameters
                    },
                }
            )
        return tuple(
            sorted(
                discovered,
                key=lambda value: (value["tool_id"], value["version"]),
            )
        )

    def build(
        self,
        *,
        plan: DbaSkillPlan,
        compiled: CompiledSkillPlan,
        capabilities: DbaCapabilitySnapshot,
        database_execution: dict,
        dynamic_queries: tuple[dict, ...] = (),
        direct_actions: tuple[object, ...] = (),
    ) -> dict:
        capability_payload = capabilities.model_dump(mode="json")
        invocations = {}
        for item, task_key in zip(
            plan.items, compiled.invocation_task_keys, strict=True
        ):
            manifest = self._skills.resolve(
                item.skill_id, item.skill_version
            )
            tools = []
            selected_steps = tuple(
                step
                for step in manifest.tool_dag
                if item.selected_tool_id is None
                or step.tool_id == item.selected_tool_id
            )
            if item.selected_tool_id is not None and len(selected_steps) != 1:
                raise SkillCatalogError(
                    f"Playbook {manifest.skill_id} 无法唯一解析所选Tool："
                    f"{item.selected_tool_id}"
                )
            for step in selected_steps:
                resolved = self._tools.resolve(
                    tool_id=step.tool_id,
                    tool_version=step.tool_version,
                    db_type=str(capabilities.database_type),
                    db_version=capabilities.database_version or "",
                    capabilities=set(capabilities.target_capabilities),
                    entitlements=set(capabilities.entitlements),
                )
                definition = resolved.definition
                self._validate_manifest_requirements(
                    manifest=manifest,
                    tool_id=definition.tool_id,
                    required_capabilities=set(
                        definition.required_capabilities
                    ),
                    required_entitlements=set(
                        definition.required_entitlements
                    ),
                    required_privileges=set(
                        definition.required_privileges
                    ),
                )
                parameters = dict(step.input)
                declared = {value.name for value in definition.parameters}
                parameters.update(
                    {
                        key: value
                        for key, value in item.input.items()
                        if key in declared
                    }
                )
                parameters = self._tools.validate_parameters(
                    resolved, parameters
                )
                tools.append(
                    {
                        "step_id": step.step_id,
                        "depends_on": (
                            []
                            if item.selected_tool_id is not None
                            else list(step.depends_on)
                        ),
                        "tool_id": definition.tool_id,
                        "tool_version": definition.version,
                        "variant": definition.variant,
                        "template_sha256": definition.template_sha256,
                        "manual_sql": resolved.sql,
                        "required_privileges": list(
                            definition.required_privileges
                        ),
                        "supported_version_min": (
                            definition.supported_version_min
                        ),
                        "supported_version_max_exclusive": (
                            definition.supported_version_max_exclusive
                        ),
                        "parameters": parameters,
                        "output_columns": [
                            column.model_dump(mode="json")
                            for column in definition.output_columns
                        ],
                        "limits": {
                            "statement_timeout_seconds": (
                                definition.timeout_seconds
                            ),
                            "max_result_rows": definition.max_rows,
                            "max_result_bytes": definition.max_bytes,
                            "max_columns": 128,
                            "max_cell_chars": 32768,
                        },
                    }
                )
            invocations[task_key] = {
                "skill_id": item.skill_id,
                "skill_version": item.skill_version,
                "manifest_hash": item.manifest_hash,
                "measurement_semantics": item.measurement_semantics,
                "presentation_kind": manifest.presentation_kind,
                "output_schema": manifest.output_schema,
                "action_id": item.action_id,
                "selected_tool_id": item.selected_tool_id,
                "tools": tools,
            }
        if len(dynamic_queries) != len(compiled.dynamic_task_keys):
            raise SkillCatalogError("动态查询快照与编译任务数量不一致")
        dynamic_invocations = {
            task_key: dict(query)
            for task_key, query in zip(
                compiled.dynamic_task_keys,
                dynamic_queries,
                strict=True,
            )
        }
        direct_invocations = {}
        for action, task_key in zip(
            direct_actions, compiled.diagnostic_task_keys, strict=True
        ):
            resolved = self._tools.resolve(
                tool_id=action.tool_id,
                tool_version="1.0.0",
                db_type=str(capabilities.database_type),
                db_version=capabilities.database_version or "",
                capabilities=set(capabilities.target_capabilities),
                entitlements=set(capabilities.entitlements),
            )
            definition = resolved.definition
            direct_invocations[task_key] = {
                "action_id": action.action_id,
                "question": action.question,
                "measurement_semantics": action.measurement_semantics,
                "presentation_kind": "TABLE",
                "catalog_hash": self._tools.catalog_hash,
                "tool": {
                    "step_id": action.action_id,
                    "depends_on": [],
                    "tool_id": definition.tool_id,
                    "tool_version": definition.version,
                    "variant": definition.variant,
                    "template_sha256": definition.template_sha256,
                    "manual_sql": resolved.sql,
                    "required_privileges": list(
                        definition.required_privileges
                    ),
                    "supported_version_min": (
                        definition.supported_version_min
                    ),
                    "supported_version_max_exclusive": (
                        definition.supported_version_max_exclusive
                    ),
                    "parameters": self._tools.validate_parameters(
                        resolved, dict(action.input)
                    ),
                    "output_columns": [
                        column.model_dump(mode="json")
                        for column in definition.output_columns
                    ],
                    "limits": {
                        "statement_timeout_seconds": (
                            definition.timeout_seconds
                        ),
                        "max_result_rows": definition.max_rows,
                        "max_result_bytes": definition.max_bytes,
                        "max_columns": 128,
                        "max_cell_chars": 32768,
                    },
                },
            }
        return {
            "schema_version": "DBA_SKILL_EXECUTION_SNAPSHOT.v1",
            "catalog_hash": plan.catalog_hash,
            "diagnostic_catalog_hash": self._tools.catalog_hash,
            "capability_snapshot": capability_payload,
            "capability_snapshot_hash": canonical_hash(capability_payload),
            "database": dict(database_execution),
            "invocations": invocations,
            "dynamic_invocations": dynamic_invocations,
            "direct_invocations": direct_invocations,
        }

    @staticmethod
    def _validate_manifest_requirements(
        *,
        manifest,
        tool_id: str,
        required_capabilities: set[str],
        required_entitlements: set[str],
        required_privileges: set[str],
    ) -> None:
        missing = (
            required_capabilities
            - set(manifest.required_target_capabilities)
        ) | (
            required_entitlements
            - set(manifest.required_entitlements)
        ) | (
            required_privileges
            - set(manifest.required_privileges)
        )
        if missing:
            raise SkillCatalogError(
                f"Skill {manifest.skill_id} 未声明 Tool {tool_id} 的要求："
                f"{', '.join(sorted(missing))}"
            )
