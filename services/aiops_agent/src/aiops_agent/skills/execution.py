"""把 Skill Plan 编译为可供 DB Executor 审计重放的冻结快照。"""

from __future__ import annotations

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

    def build(
        self,
        *,
        plan: DbaSkillPlan,
        compiled: CompiledSkillPlan,
        capabilities: DbaCapabilitySnapshot,
        database_execution: dict,
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
            for step in manifest.tool_dag:
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
                        "depends_on": list(step.depends_on),
                        "tool_id": definition.tool_id,
                        "tool_version": definition.version,
                        "variant": definition.variant,
                        "template_sha256": definition.template_sha256,
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
                "tools": tools,
            }
        return {
            "schema_version": "DBA_SKILL_EXECUTION_SNAPSHOT.v1",
            "catalog_hash": plan.catalog_hash,
            "diagnostic_catalog_hash": self._tools.catalog_hash,
            "capability_snapshot": capability_payload,
            "capability_snapshot_hash": canonical_hash(capability_payload),
            "database": dict(database_execution),
            "invocations": invocations,
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
