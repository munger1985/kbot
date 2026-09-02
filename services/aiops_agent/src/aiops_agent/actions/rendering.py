"""按动作登记的确定性 Renderer；不接受自由 SQL 或命令片段。"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Callable

from .contracts import RenderedAction, ResolvedActionTemplate
from .validation import validate_rendered_action


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_$#]{0,127}$")


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _oracle_identifier(value: object) -> str:
    parsed = str(value)
    if _IDENTIFIER.fullmatch(parsed) is None:
        raise ValueError("Oracle 标识符格式无效")
    return f'"{parsed}"'


def _database_object_ref(
    value: object, object_types: tuple[str, ...]
) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError("数据库对象引用必须是结构化对象")
    allowed_keys = {
        "container",
        "schema",
        "object_type",
        "object_name",
        "partition",
    }
    if not set(value).issubset(allowed_keys):
        raise ValueError("数据库对象引用包含未知字段")
    required = {"schema", "object_type", "object_name"}
    if not required.issubset(value):
        raise ValueError("数据库对象引用缺少必要字段")
    normalized = {
        key: str(item) for key, item in value.items() if item is not None
    }
    if normalized["object_type"] not in object_types:
        raise ValueError("数据库对象类型不在 Action Allowlist")
    for key in ("container", "schema", "object_name", "partition"):
        if key in normalized and _IDENTIFIER.fullmatch(normalized[key]) is None:
            raise ValueError("数据库对象引用中的标识符无效")
    return normalized


def _normalize(rule, value: object) -> Any:
    if rule.type in {"integer", "size", "duration"}:
        if isinstance(value, bool):
            raise ValueError("数值 Action 参数不能是布尔值")
        parsed = int(value)
        if rule.minimum is not None and parsed < rule.minimum:
            raise ValueError("Action 数值参数低于下限")
        if rule.maximum is not None and parsed > rule.maximum:
            raise ValueError("Action 数值参数超过上限")
        return parsed
    if rule.type == "boolean":
        if not isinstance(value, bool):
            raise ValueError("Action 布尔参数类型无效")
        return value
    if rule.type == "database_object_ref":
        return _database_object_ref(value, rule.object_types)
    if rule.type == "timestamp":
        if isinstance(value, datetime):
            return value.isoformat()
        parsed = str(value)
        datetime.fromisoformat(parsed.replace("Z", "+00:00"))
        return parsed
    parsed = str(value)
    if rule.type == "enum":
        if parsed not in rule.enum:
            raise ValueError("Action 枚举参数无效")
        return parsed
    if rule.max_length is not None and len(parsed) > rule.max_length:
        raise ValueError("Action 字符串参数过长")
    if rule.min_length is not None and len(parsed) < rule.min_length:
        raise ValueError("Action 字符串参数过短")
    if rule.pattern is not None and re.fullmatch(rule.pattern, parsed) is None:
        raise ValueError("Action 字符串参数格式无效")
    return parsed


def _strict_scalar_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    command = template.strip()
    rules = {item.name: item for item in definition.parameters}
    for name, value in parameters.items():
        rule = rules[name]
        if rule.type == "database_object_ref":
            ref = dict(value)
            rendered = (
                f'{_oracle_identifier(ref["schema"])}.'
                f'{_oracle_identifier(ref["object_name"])}'
            )
        elif rule.type == "identifier":
            rendered = _oracle_identifier(value)
        elif rule.type == "boolean":
            rendered = "TRUE" if value else "FALSE"
        else:
            rendered = str(value)
        command = command.replace(f"{{{{{name}}}}}", rendered)
    return command


def _oracle_index_rebuild_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    del definition
    ref = dict(parameters["index_ref"])
    index_name = (
        f'{_oracle_identifier(ref["schema"])}.'
        f'{_oracle_identifier(ref["object_name"])}'
    )
    online_clause = " ONLINE" if parameters["online"] else ""
    return template.strip().replace("{{index_ref}}", index_name).replace(
        "{{online}}", online_clause
    )


def _oracle_index_partition_rebuild_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    del definition
    ref = dict(parameters["index_ref"])
    index_name = (
        f'{_oracle_identifier(ref["schema"])}.'
        f'{_oracle_identifier(ref["object_name"])}'
    )
    partition_name = _oracle_identifier(parameters["partition_name"])
    if ref.get("partition") != parameters["partition_name"]:
        raise ValueError("索引分区引用与分区参数不一致")
    online_clause = " ONLINE" if parameters["online"] else ""
    return (
        template.strip()
        .replace("{{index_ref}}", index_name)
        .replace("{{partition_name}}", partition_name)
        .replace("{{online}}", online_clause)
    )


def _oracle_object_compile_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    del definition
    object_type = str(parameters["object_type"])
    ref = dict(parameters["object_ref"])
    if ref["object_type"] != object_type:
        raise ValueError("对象引用类型与编译类型不一致")
    object_name = (
        f'{_oracle_identifier(ref["schema"])}.'
        f'{_oracle_identifier(ref["object_name"])}'
    )
    return (
        template.strip()
        .replace("{{object_type}}", object_type)
        .replace("{{object_ref}}", object_name)
    )


def _oracle_table_statistics_gather_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    del definition
    ref = dict(parameters["table_ref"])
    table_arguments = (
        f"ownname => '{ref['schema']}', "
        f"tabname => '{ref['object_name']}'"
    )
    return template.strip().replace("{{table_ref}}", table_arguments)


def _oracle_scheduler_job_run_renderer(
    template: str, parameters: dict[str, Any], definition
) -> str:
    del definition
    ref = dict(parameters["job_ref"])
    qualified_name = (
        f'{_oracle_identifier(ref["schema"])}.'
        f'{_oracle_identifier(ref["object_name"])}'
    )
    job_argument = f"job_name => '{qualified_name}'"
    return template.strip().replace("{{job_ref}}", job_argument)


_RENDERERS: dict[str, Callable[[str, dict[str, Any], Any], str]] = {
    "strict-scalar.v2": _strict_scalar_renderer,
    "oracle-index-rebuild.v1": _oracle_index_rebuild_renderer,
    "oracle-index-partition-rebuild.v1": (
        _oracle_index_partition_rebuild_renderer
    ),
    "oracle-object-compile.v1": _oracle_object_compile_renderer,
    "oracle-table-statistics-gather.v1": (
        _oracle_table_statistics_gather_renderer
    ),
    "oracle-scheduler-job-run.v1": _oracle_scheduler_job_run_renderer,
}


class ActionRenderer:
    def render(
        self,
        template: ResolvedActionTemplate,
        parameters: dict[str, object],
    ) -> RenderedAction:
        definition = template.definition
        if (
            definition.execution_mode == "UNSUPPORTED"
            or template.command_template is None
        ):
            raise ValueError("不支持的 Action 不能渲染命令")
        expected = {item.name: item for item in definition.parameters}
        if set(parameters) != set(expected):
            raise ValueError("Action 参数集合与模板不一致")
        normalized = {
            name: _normalize(rule, parameters[name])
            for name, rule in expected.items()
        }
        renderer = _RENDERERS.get(definition.renderer_id)
        if renderer is None:
            raise ValueError("Action Renderer 未登记")
        command = renderer(template.command_template, normalized, definition)
        validate_rendered_action(command, definition=definition)
        parameters_hash = _sha256(normalized)
        return RenderedAction(
            action_template_id=definition.action_template_id,
            action_template_version=definition.version,
            variant=definition.variant,
            db_type=definition.db_type,
            renderer_version=definition.renderer_version,
            typed_parameters=normalized,
            parameters_hash=parameters_hash,
            command_text=command,
            command_hash=hashlib.sha256(command.encode()).hexdigest(),
            template_hash=template.template_hash,
            risk_level=definition.risk_level,
            action_family=definition.action_family,
            effect_class=definition.effect_class,
            execution_mode=definition.execution_mode,
            executor_kind=definition.executor_kind,
            precondition_tool_refs=definition.precondition_tool_refs,
            verification_tool_refs=definition.verification_tool_refs,
            expected_effects=definition.expected_effects,
            rollback_description=definition.rollback_description,
            statement_timeout_seconds=definition.statement_timeout_seconds,
            observation_delay_seconds=definition.observation_delay_seconds,
            idempotency_class=definition.idempotency_class,
            concurrency_key=definition.concurrency_key,
            lock_impact=definition.lock_impact,
            estimated_duration_seconds=definition.estimated_duration_seconds,
            cancellable=definition.cancellable,
        )
