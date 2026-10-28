"""把调查 Tool 与 Playbook 计划编译为可审计重放的冻结快照。"""

from __future__ import annotations

import re

from aiops_agent.application.top_sql_window import (
    TOP_SQL_REFERENCE_CONDITION, TOP_SQL_REFERENCE_KIND, TOP_SQL_WINDOW_TOOL,
    validate_top_sql_window,
)
from aiops_agent.diagnostics import DiagnosticRegistry, ResolvedDiagnosticTool
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
from .sampling import freeze_sampling, freeze_observation_schedule, validate_sampling_plan


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
            if not set(definition.required_capabilities) <= set(capabilities.target_capabilities):
                # 固定目录仅包含已离线验证的受控只读工具。未知访问能力可按需探测，
                # 但商业许可仍为硬门禁，实际权限由目标库执行反馈，绝不写成已具备。
                if "DB_READONLY" not in capabilities.target_capabilities:
                    continue
            if not set(definition.required_entitlements) <= set(
                capabilities.entitlements
            ):
                continue
            output_names = tuple(
                column.name for column in definition.output_columns
            )
            purpose = definition.description or (
                f"受控只读数据库观测：{definition.tool_id}"
            )
            card = {
                "tool_id": definition.tool_id,
                "version": definition.version,
                "tool_class": "ORACLE_SQL",
                "access_status": "UNKNOWN",
                "limit_semantics": definition.limit_semantics,
                "required_privileges": list(definition.required_privileges),
                "required_capabilities": list(definition.required_capabilities),
                "description": (
                    f"{purpose}；"
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
            if definition.discovery_tool_id:
                card["discovery_tool_id"] = definition.discovery_tool_id
            if definition.measurement_semantics is not None:
                card["measurement_semantics"] = definition.measurement_semantics
            if definition.cost_level == "LOW":
                card["observation_schedule_policy"] = {
                    "count_including_current_baseline": "2..4", "interval_seconds": "30..900",
                    "shares_one_group_per_round_with_sampling": True,
                    "execution": "PERSISTED_FUTURE_RUN_ONLY_AFTER_SUCCESSFUL_BASELINE",
                    "max_rows_per_point": 200, "max_bytes_per_point": 131072, "max_statement_seconds": 5,
                    "not_requested_historical_window": True,
                }
                card["sampling_policy"] = {
                    "count": "2..4", "interval_seconds": "2..10", "max_groups_per_round": 1,
                    "max_rows_per_sample": 200, "max_bytes_per_sample": 131072,
                    "semantics": "CURRENT_ACTIVITY or CUMULATIVE_SINCE_LOAD; not requested historical window",
                }
            discovered.append(card)
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
        validate_sampling_plan(actions)
        normalized = {}
        for action in actions:
            if action.tool_id == TOP_SQL_WINDOW_TOOL:
                normalized[action.action_id] = validate_top_sql_window(dict(action.input))
                continue
            resolved = self._resolve_direct_action(
                action=action,
                capabilities=capabilities,
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
            freeze_sampling(action, resolved.definition, parameters)
        return normalized

    def _resolve_direct_action(
        self,
        *,
        action: object,
        capabilities: DbaCapabilitySnapshot,
    ) -> ResolvedDiagnosticTool:
        """按当前能力发现结果冻结模型动作的真实 Tool 版本。"""
        matches = tuple(
            item
            for item in self.discover_tools(capabilities)
            if item["tool_id"] == action.tool_id
        )
        if len(matches) != 1:
            raise LookupError(
                f"诊断工具无法按当前能力唯一解析：{action.tool_id}"
            )
        resolved = self._tools.resolve(
            tool_id=action.tool_id,
            tool_version=str(matches[0]["version"]),
            db_type=str(capabilities.database_type),
            db_version=capabilities.database_version or "",
            capabilities=set(capabilities.target_capabilities) | set(matches[0].get("required_capabilities", ())),
            entitlements=set(capabilities.entitlements),
        )
        if resolved.definition.measurement_semantics is not None and str(action.measurement_semantics) != resolved.definition.measurement_semantics:
            raise ValueError(f"工具 {action.tool_id} 的观测口径必须使用目录声明：{resolved.definition.measurement_semantics}")
        return resolved

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
                if definition.measurement_semantics is not None and str(item.measurement_semantics) != definition.measurement_semantics:
                    raise PlaybookCatalogError(f"Playbook 观测口径与工具目录不一致：{definition.tool_id}")
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
        direct_actions = tuple(
            action
            for action in direct_actions
            if not getattr(action, "deferred", False)
        )
        direct_invocations = {}
        validate_sampling_plan(direct_actions)
        for action, task_key in zip(
            direct_actions, compiled.diagnostic_task_keys, strict=True
        ):
            if action.tool_id == TOP_SQL_WINDOW_TOOL:
                direct_invocations[task_key] = {
                    "action_id": action.action_id,
                    "question": action.question,
                    "measurement_semantics": action.measurement_semantics,
                    "presentation_kind": "TABLE",
                    "tool": {
                        "tool_id": TOP_SQL_WINDOW_TOOL,
                        "tool_version": "1.0.0",
                        "parameters": validate_top_sql_window(dict(action.input)),
                    },
                }
                continue
            resolved = self._resolve_direct_action(
                action=action,
                capabilities=capabilities,
            )
            definition = resolved.definition
            sampling = freeze_sampling(action, definition, self._tools.validate_parameters(resolved, dict(action.input)))
            schedule = freeze_observation_schedule(action)
            direct_invocations[task_key] = {
                "action_id": action.action_id,
                "question": action.question,
                "measurement_semantics": action.measurement_semantics,
                "presentation_kind": "TABLE",
                "catalog_hash": self._tools.catalog_hash,
                "sampling": sampling,
                "observation_schedule": schedule,
                "execution_condition": (
                    TOP_SQL_REFERENCE_CONDITION
                    if getattr(action, "optional", False)
                    and getattr(action, "expected_evidence_kind", "") == TOP_SQL_REFERENCE_KIND
                    else None
                ),
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
                            min(5, definition.timeout_seconds) if sampling or schedule else definition.timeout_seconds
                        ),
                        "max_result_rows": min(200, definition.max_rows) if sampling or schedule else definition.max_rows,
                        "max_result_bytes": min(131072, definition.max_bytes) if sampling or schedule else definition.max_bytes,
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
