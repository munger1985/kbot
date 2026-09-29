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

    def list_for_target(self, target: Any) -> dict[str, Any]:
        db_type = str(target.db_type)
        items = []
        for item in self._items.values():
            if db_type not in item.get("supported_db_types", ()):
                continue
            status, reason = self._availability(target)
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
                    "input_schema": input_schema,
                    "status": status,
                    "availability_reason": reason,
                }
            )
        return {
            "catalog_version": self.version,
            "target_id": str(target.target_id),
            "db_type": db_type,
            "starters": sorted(items, key=lambda value: value["sort_order"]),
        }

    def freeze(self, *, selection: Any, target: Any) -> dict[str, Any]:
        if str(selection.catalog_version) != self.version:
            raise validation_failed("功能目录已经更新，请重新打开功能菜单")
        item = self._items.get(str(selection.starter_id))
        if item is None:
            raise validation_failed("所选功能不存在或已经下线")
        if str(target.db_type) not in item.get("supported_db_types", ()):
            raise validation_failed("所选功能不支持当前 Target 数据库类型")
        status, reason = self._availability(target)
        if status == "UNAVAILABLE":
            raise validation_failed(reason)
        input_schema = self._input_schema(item)
        try:
            if str(dict(item.get("planning") or {}).get("kind") or "") == "IMPLEMENTATION":
                profile = ImplementationProfile(
                    str(item["planning"]["implementation_profile"])
                )
                parameters = normalize_implementation_parameters(
                    profile, dict(selection.parameters)
                )
            else:
                parameters = self._validate_parameters(
                    list(input_schema), dict(selection.parameters)
                )
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc
        return {
            "starter_id": item["starter_id"],
            "catalog_version": self.version,
            "category": item["category"],
            "title": item["title"],
            "execution_mode": item["execution_mode"],
            "parameters": parameters,
            "planning": dict(item["planning"]),
            "user_message": self._user_message(
                item, parameters, input_schema=input_schema
            ),
        }

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
