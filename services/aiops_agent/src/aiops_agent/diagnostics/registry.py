"""不可变 Diagnostic Tool Catalog 加载与精确选择。"""

from __future__ import annotations

import hashlib
import json
import re
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
        return validate_parameters(tool.definition, values)
