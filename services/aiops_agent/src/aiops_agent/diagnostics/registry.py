"""不可变 Diagnostic Tool Catalog 加载与精确选择。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import DiagnosticToolDefinition
from .validation import validate_parameters, validate_readonly_template


DEFAULT_CATALOG_ROOT = Path(__file__).resolve().parent / "catalog"


@dataclass(frozen=True)
class ResolvedDiagnosticTool:
    definition: DiagnosticToolDefinition
    sql: str


def database_major_version(version_code: str) -> int:
    match = re.search(r"\d+", version_code)
    if not match:
        raise ValueError("数据库版本无法识别")
    return int(match.group(0))


class DiagnosticRegistry:
    def __init__(self, tools: tuple[ResolvedDiagnosticTool, ...]):
        identities = [
            (
                item.definition.tool_id,
                item.definition.version,
                item.definition.db_type,
                item.definition.variant,
            )
            for item in tools
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("诊断目录存在重复工具 Variant")
        self._tools = tools
        self.catalog_hash = hashlib.sha256(
            json.dumps(
                [
                    item.definition.model_dump(mode="json")
                    for item in sorted(
                        tools,
                        key=lambda tool: (
                            tool.definition.db_type,
                            tool.definition.tool_id,
                            tool.definition.version,
                            tool.definition.variant,
                        ),
                    )
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    @classmethod
    def load(cls, root: Path | None = None) -> "DiagnosticRegistry":
        catalog_root = root or DEFAULT_CATALOG_ROOT
        tools: list[ResolvedDiagnosticTool] = []
        for manifest_path in sorted(catalog_root.glob("*/manifest.json")):
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for raw in payload.get("tools", []):
                definition = DiagnosticToolDefinition.model_validate(raw)
                template_path = (manifest_path.parent / definition.template_ref).resolve()
                if not template_path.is_relative_to(manifest_path.parent.resolve()):
                    raise ValueError("诊断模板路径越界")
                sql_bytes = template_path.read_bytes()
                digest = hashlib.sha256(sql_bytes).hexdigest()
                if digest != definition.template_sha256:
                    raise ValueError(
                        f"诊断模板 Hash 不匹配：{definition.tool_id}"
                    )
                sql = sql_bytes.decode("utf-8")
                validate_readonly_template(sql, definition)
                tools.append(ResolvedDiagnosticTool(definition, sql))
        if not tools:
            raise ValueError("诊断目录为空")
        return cls(tuple(tools))

    @property
    def tools(self) -> tuple[ResolvedDiagnosticTool, ...]:
        return self._tools

    def resolve(
        self,
        *,
        tool_id: str,
        tool_version: str,
        db_type: str,
        db_version: str,
        capabilities: set[str],
        entitlements: set[str],
    ) -> ResolvedDiagnosticTool:
        major = database_major_version(db_version)
        candidates = [
            item
            for item in self._tools
            if item.definition.tool_id == tool_id
            and item.definition.version == tool_version
            and item.definition.db_type == db_type
            and item.definition.supported_version_min <= major
            < item.definition.supported_version_max_exclusive
            and set(item.definition.required_capabilities) <= capabilities
            and set(item.definition.required_entitlements) <= entitlements
        ]
        if len(candidates) != 1:
            raise LookupError("诊断工具没有唯一且精确匹配的 Variant")
        return candidates[0]

    def resolve_exact(
        self,
        *,
        tool_id: str,
        tool_version: str,
        db_type: str,
        variant: str,
        template_sha256: str,
    ) -> ResolvedDiagnosticTool:
        candidates = [
            item
            for item in self._tools
            if item.definition.tool_id == tool_id
            and item.definition.version == tool_version
            and item.definition.db_type == db_type
            and item.definition.variant == variant
            and item.definition.template_sha256 == template_sha256
        ]
        if len(candidates) != 1:
            raise LookupError("Executor 本地目录与 Grant 不匹配")
        return candidates[0]

    def validate_parameters(
        self, tool: ResolvedDiagnosticTool, values: dict[str, Any]
    ) -> dict[str, Any]:
        normalized = validate_parameters(tool.definition, values)
        self._validate_oracle_workload_report_parameters(
            tool.definition.tool_id, normalized
        )
        return normalized

    @staticmethod
    def _validate_oracle_workload_report_parameters(
        tool_id: str, parameters: dict[str, Any]
    ) -> None:
        """校验原生工作负载报告的跨字段边界，避免包调用接收歧义范围。"""
        if tool_id == "db.oracle.awr.report":
            if parameters["begin_snapshot_id"] >= parameters["end_snapshot_id"]:
                raise ValueError("AWR 报告起始快照必须早于结束快照")
        elif tool_id == "db.oracle.awr.diff_report":
            baseline_begin = parameters["baseline_begin_snapshot_id"]
            baseline_end = parameters["baseline_end_snapshot_id"]
            after_begin = parameters["after_begin_snapshot_id"]
            after_end = parameters["after_end_snapshot_id"]
            if (
                baseline_begin >= baseline_end
                or baseline_end > after_begin
                or after_begin >= after_end
            ):
                raise ValueError("AWR 对比报告的两段快照必须按时间先后且不重叠")
        elif tool_id == "db.oracle.ash.report":
            try:
                begin = datetime.fromisoformat(
                    parameters["begin_time"].replace("Z", "+00:00")
                )
                end = datetime.fromisoformat(
                    parameters["end_time"].replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError("ASH 报告时间必须为 ISO 8601 格式") from exc
            if begin.tzinfo is None or end.tzinfo is None:
                raise ValueError("ASH 报告时间必须包含 UTC 偏移")
            if begin >= end:
                raise ValueError("ASH 报告起始时间必须早于结束时间")
            if (end - begin).total_seconds() > 86_400:
                raise ValueError("ASH 报告时间范围不能超过 24 小时")
