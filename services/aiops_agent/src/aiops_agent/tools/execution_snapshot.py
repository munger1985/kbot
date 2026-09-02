"""把调查 Tool 与 Playbook 计划编译为可审计重放的冻结快照。"""

from __future__ import annotations

import re

from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.playbooks import (
    PlaybookCatalogError,
    PlaybookRegistry,
    canonical_hash,
)
from platform_core.contracts.aiops.playbooks import (
    DbaCapabilitySnapshot,
    DbaPlaybookPlan,
)

from .compiler import CompiledInvestigationPlan


class ToolExecutionSnapshotBuilder:
    def __init__(
        self,
        *,
        playbook_registry: PlaybookRegistry,
        diagnostic_registry: DiagnosticRegistry,
    ) -> None:
        self._playbooks = playbook_registry
        self._tools = diagnostic_registry

    def validate_catalog(self) -> None:
        """离线验证所有 Playbook 对 Tool 版本和要求的声明完整性。"""
        for manifest in self._playbooks.manifests():
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
                        raise PlaybookCatalogError(
                            f"Playbook {manifest.playbook_id} 的 Tool 无法精确解析："
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
        normalized_privileges = {
            str(value).strip().upper() for value in configured_privileges
        }
        dictionary_access = (
            str(capabilities.database_type) == "ORACLE"
            and "SELECT ANY DICTIONARY" in normalized_privileges
        )
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
            if configured_privileges and not dictionary_access and not set(
                definition.required_privileges
            ) <= configured_privileges:
                continue
            output_names = tuple(
                column.name for column in definition.output_columns
            )
            discovered.append(
                {
                    "tool_id": definition.tool_id,
                    "version": definition.version,
                    "tool_class": "ORACLE_SQL",
                    "description": (
                        f"受控只读数据库观测：{definition.tool_id}；"
                        f"返回字段：{', '.join(output_names)}"
                    ),
                    "input": {
                        parameter.name: parameter.model_dump(
                            mode="json",
                            exclude={"name"},
                            exclude_none=True,
                        )
                        for parameter in definition.parameters
                    },
                    "returns": list(output_names),
                }
            )
        return tuple(
            sorted(
                discovered,
                key=lambda value: (value["tool_id"], value["version"]),
            )
        )

    def validate_direct_actions(
        self,
        *,
        actions: tuple[object, ...],
        capabilities: DbaCapabilitySnapshot,
    ) -> dict[str, dict[str, object]]:
        """在编译前校验模型选择的固定目录 Tool 参数并补齐默认值。"""
        normalized = {}
        for action in actions:
            resolved = self._tools.resolve(
                tool_id=action.tool_id,
                tool_version="1.0.0",
                db_type=str(capabilities.database_type),
                db_version=capabilities.database_version or "",
                capabilities=set(capabilities.target_capabilities),
                entitlements=set(capabilities.entitlements),
            )
            try:
                parameters = self._tools.validate_parameters(
                    resolved, dict(action.input)
                )
            except ValueError as exc:
                raise ValueError(
                    f"工具 {action.tool_id} 输入无效：{exc}"
                ) from exc
            normalized[action.action_id] = parameters
        return normalized

    def build(
        self,
        *,
        plan: DbaPlaybookPlan,
        compiled: CompiledInvestigationPlan,
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
            manifest = self._playbooks.resolve(
                item.playbook_id, item.playbook_version
            )
            tools = []
            selected_steps = tuple(
                step
                for step in manifest.tool_dag
                if item.selected_tool_id is None
                or step.tool_id == item.selected_tool_id
            )
            if item.selected_tool_id is not None and len(selected_steps) != 1:
                raise PlaybookCatalogError(
                    f"Playbook {manifest.playbook_id} 无法唯一解析所选Tool："
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
                "playbook_id": item.playbook_id,
                "playbook_version": item.playbook_version,
                "manifest_hash": item.manifest_hash,
                "measurement_semantics": item.measurement_semantics,
                "presentation_kind": manifest.presentation_kind,
                "output_schema": manifest.output_schema,
                "action_id": item.action_id,
                "selected_tool_id": item.selected_tool_id,
                "tools": tools,
            }
        if len(dynamic_queries) != len(compiled.dynamic_task_keys):
            raise PlaybookCatalogError("动态查询快照与编译任务数量不一致")
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
            "schema_version": "DBA_INVESTIGATION_EXECUTION_SNAPSHOT.v1",
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
            raise PlaybookCatalogError(
                f"Playbook {manifest.playbook_id} 未声明 Tool {tool_id} 的要求："
                f"{', '.join(sorted(missing))}"
            )
