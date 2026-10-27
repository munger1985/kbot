"""版本化功能入口目录、参数校验和审计文本生成。"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from aiops_agent.application.errors import validation_failed
from aiops_agent.application.implementation.inputs import (
    implementation_input_schema,
    normalize_implementation_parameters,
)
from platform_core.contracts.aiops import ImplementationProfile


class ConversationStarterCatalog:
    """加载受版本控制的目录，拒绝由浏览器决定内部规划语义。"""

    def __init__(self, path: Path | None = None) -> None:
        catalog_path = path or Path(__file__).with_name("catalog.json")
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.version = str(payload["catalog_version"])
        self._items = {
            str(item["starter_id"]): dict(item)
            for item in payload.get("starters", ())
        }

    def list_for_agent(
        self,
        *,
        agent_id,
        capabilities: set[str],
        targets: tuple[Any, ...],
        context_hint=None,
    ) -> dict[str, Any]:
        items = []
        for item in self._items.values():
            required = self._required_capabilities(item)
            if not set(required) <= capabilities:
                continue
            compatible_targets = tuple(
                target
                for target in targets
                if str(target.db_type) in item.get("supported_db_types", ())
            )
            if not compatible_targets:
                continue
            availability = [self._availability(target) for target in compatible_targets]
            status = next(
                (
                    candidate
                    for candidate in ("AVAILABLE", "LIMITED", "UNAVAILABLE")
                    if any(value[0] == candidate for value in availability)
                ),
                "UNAVAILABLE",
            )
            reason = next(
                (value[1] for value in availability if value[0] == status),
                None,
            )
            input_schema = self._input_schema(item)
            items.append(
                {
                    key: item[key]
                    for key in (
                        "starter_id",
                        "category",
                        "title",
                        "description",
                        "supported_db_types",
                        "execution_mode",
                        "sort_order",
                    )
                }
                | {
                    "required_capabilities": required,
                    "input_schema": input_schema,
                    "status": status,
                    "availability_reason": reason,
                }
            )
        return {
            "catalog_version": self.version,
            "agent_id": str(agent_id),
            "context_hint": (
                context_hint.model_dump(mode="json")
                if context_hint is not None
                else None
            ),
            "starters": sorted(items, key=lambda value: value["sort_order"]),
        }

    def freeze(
        self,
        *,
        selection: Any,
        capabilities: set[str],
        target: Any | None = None,
    ) -> dict[str, Any]:
        if str(selection.catalog_version) != self.version:
            raise validation_failed("功能目录已经更新，请重新打开功能菜单")
        item = self._items.get(str(selection.starter_id))
        if item is None:
            raise validation_failed("所选功能不存在或已经下线")
        required = set(self._required_capabilities(item))
        if not required <= capabilities:
            raise validation_failed("当前 Agent 未启用该功能所需能力")
        if target is not None:
            if str(target.db_type) not in item.get("supported_db_types", ()):
                raise validation_failed("所选功能不支持当前业务资源数据库类型")
            status, reason = self._availability(target)
            if status == "UNAVAILABLE":
                raise validation_failed(reason)
        return self._freeze_item(
            item=item,
            supplied_parameters=dict(selection.parameters),
            source="UI_ENTRY",
            required=required,
        )

    def resolver_candidates(
        self, *, supported_db_types: set[str]
    ) -> tuple[dict[str, Any], ...]:
        """向自然语言解析器公开语义和参数，不公开内部 Tool DAG。"""
        return tuple(
            {
                "capability_id": str(item["starter_id"]),
                "title": str(item["title"]),
                "description": str(item["description"]),
                "execution_mode": str(item["execution_mode"]),
                "supported_db_types": list(item["supported_db_types"]),
                "parameter_schema": list(self._input_schema(item)),
            }
            for item in sorted(
                self._items.values(), key=lambda value: value["sort_order"]
            )
            if supported_db_types
            & {str(value) for value in item.get("supported_db_types", ())}
        )

    def freeze_natural_language(
        self, *, capability_id: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        item = self._items.get(str(capability_id))
        if item is None:
            raise validation_failed("自然语言选择了目录外的运维能力")
        return self._freeze_item(
            item=item,
            supplied_parameters=parameters,
            source="NATURAL_LANGUAGE",
            required=set(self._required_capabilities(item)),
        )

    def _freeze_item(
        self,
        *,
        item: dict[str, Any],
        supplied_parameters: dict[str, Any],
        source: str,
        required: set[str],
    ) -> dict[str, Any]:
        input_schema = self._input_schema(item)
        try:
            if str(dict(item.get("planning") or {}).get("kind") or "") == "IMPLEMENTATION":
                profile = ImplementationProfile(
                    str(item["planning"]["implementation_profile"])
                )
                parameters = normalize_implementation_parameters(
                    profile, supplied_parameters
                )
            else:
                parameters = self._validate_parameters(
                    list(input_schema), supplied_parameters
                )
                if str(dict(item.get("planning") or {}).get("kind") or "") == "TOP_SQL_LIST":
                    if parameters.get("window") == "custom" and not (
                        parameters.get("begin_time") and parameters.get("end_time")
                    ):
                        raise ValueError("自定义 Top SQL 时间窗口需要开始时间和结束时间")
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc
        return {
            "starter_id": item["starter_id"],
            "catalog_version": self.version,
            "source": source,
            "category": item["category"],
            "title": item["title"],
            "execution_mode": item["execution_mode"],
            "required_capabilities": sorted(required),
            "parameters": parameters,
            "planning": dict(item["planning"]),
            "supported_db_types": list(item["supported_db_types"]),
            "user_message": self._user_message(
                item, parameters, input_schema=input_schema
            ),
        }

    @staticmethod
    def _required_capabilities(item: dict[str, Any]) -> tuple[str, ...]:
        declared = tuple(item.get("required_capabilities") or ())
        return declared or ("ACTIVE_DIAGNOSIS",)

    @staticmethod
    def _input_schema(item: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        """实施档案由代码目录动态提供参数，其他入口继续读取 JSON。"""
        planning = dict(item.get("planning") or {})
        if str(planning.get("kind") or "") != "IMPLEMENTATION":
            return tuple(dict(field) for field in item.get("input_schema", ()))
        profile = ImplementationProfile(str(planning["implementation_profile"]))
        return implementation_input_schema(profile)

    @staticmethod
    def _availability(target: Any) -> tuple[str, str | None]:
        if str(target.status) != "ENABLED":
            return "UNAVAILABLE", "Target 当前未启用"
        if not bool(getattr(target, "readonly_connection_enabled", False)):
            return "UNAVAILABLE", "该功能需要先配置数据库只读连接"
        connectivity = str(target.connectivity_status)
        if connectivity == "UNKNOWN":
            return "LIMITED", "连接状态尚未确认，执行时会实时核验"
        if connectivity not in {"CONNECTED", "DEGRADED"}:
            return "UNAVAILABLE", "Target 数据库连接当前不可用"
        return "AVAILABLE", None

    @classmethod
    def _validate_parameters(
        cls, schema: list[dict[str, Any]], supplied: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {str(field["name"]) for field in schema}
        unknown = sorted(set(supplied) - allowed)
        if unknown:
            raise validation_failed("功能参数包含未知字段：" + ", ".join(unknown))
        result: dict[str, Any] = {}
        for field in schema:
            name = str(field["name"])
            value = supplied.get(name, field.get("default"))
            if field.get("required") and (
                value is None or str(value).strip() == ""
            ):
                raise validation_failed(f"功能参数缺少：{field['label']}")
            if value is None:
                continue
            kind = str(field.get("type") or "text")
            if kind == "integer":
                try:
                    value = int(value)
                except (TypeError, ValueError) as exc:
                    raise validation_failed(f"{field['label']}必须是整数") from exc
                if value < int(field.get("min", value)) or value > int(
                    field.get("max", value)
                ):
                    raise validation_failed(f"{field['label']}超出允许范围")
            else:
                value = str(value).strip()
                max_length = field.get("max_length")
                if max_length is not None and len(value) > int(max_length):
                    raise validation_failed(f"{field['label']}长度超出允许范围")
                options = {
                    str(option["value"])
                    for option in field.get("options", ())
                    if isinstance(option, dict) and "value" in option
                }
                if options and value not in options:
                    raise validation_failed(f"{field['label']}不是允许的选项")
                if kind == "datetime":
                    try:
                        parsed = datetime.fromisoformat(
                            value.replace("Z", "+00:00")
                        )
                    except ValueError as exc:
                        raise validation_failed(f"{field['label']}不是有效时间") from exc
                    if parsed.tzinfo is None:
                        raise validation_failed(f"{field['label']}必须包含时区偏移")
                pattern = field.get("pattern")
                if pattern and not re.fullmatch(str(pattern), value):
                    raise validation_failed(f"{field['label']}格式不正确")
            result[name] = value
        cls._validate_time_ranges(result)
        return result

    @staticmethod
    def _validate_time_ranges(parameters: dict[str, Any]) -> None:
        def parsed(name: str) -> datetime:
            return datetime.fromisoformat(
                str(parameters[name]).replace("Z", "+00:00")
            )

        if {"begin_time", "end_time"} <= parameters.keys():
            if parsed("begin_time") >= parsed("end_time"):
                raise validation_failed("结束时间必须晚于开始时间")
        diff_fields = {
            "first_begin_time",
            "first_end_time",
            "second_begin_time",
            "second_end_time",
        }
        if diff_fields <= parameters.keys():
            first = parsed("first_end_time") - parsed("first_begin_time")
            second = parsed("second_end_time") - parsed("second_begin_time")
            if first.total_seconds() <= 0 or second.total_seconds() <= 0:
                raise validation_failed("两个对比区间都必须是正向时间区间")
            if first != second:
                raise validation_failed("AWR 对比的两个时间区间必须等长")

    @staticmethod
    def _user_message(
        item: dict[str, Any],
        parameters: dict[str, Any],
        *,
        input_schema: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    ) -> str:
        if not parameters:
            return f"执行功能：{item['title']}"
        labels = {
            str(field["name"]): str(field["label"])
            for field in (input_schema or item.get("input_schema", ()))
        }
        lines = [f"执行功能：{item['title']}"]
        lines.extend(
            f"{labels.get(key, key)}：{value}"
            for key, value in parameters.items()
        )
        return "\n".join(lines)
