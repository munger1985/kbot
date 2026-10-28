"""有界校验参数材料；不生成 SQL、不推定原解析环境或参数代表性。"""

import json
import re


MAX_MATERIAL_BYTES = 65536


class SqlPlanMaterialInvalid(ValueError):
    """错误不携带材料、原始解析错误或绑定原值。"""


def _reject():
    raise SqlPlanMaterialInvalid("SQL_PLAN_MATERIAL_INVALID")


def _pairs(entries):
    result = {}
    for key, value in entries:
        if key in result:
            _reject()
        result[key] = value
    return result


def _text(value, *, nullable=False, maximum=4096):
    if value is None and nullable:
        return
    if not isinstance(value, str) or len(value) > maximum or any(ord(char) < 32 for char in value):
        _reject()


def _value(value, depth=0):
    # 数值原值以十进制字符串传递，避免浏览器和 JSON 浮点转换改变绑定值。
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, str):
        if len(value) > 16384:
            _reject()
        return
    if isinstance(value, list) and depth < 4 and len(value) <= 256:
        for item in value:
            _value(item, depth + 1)
        return
    _reject()


def parse_sql_plan_material(raw):
    """原值只在专用保护服务内短暂使用，任何失败均返回固定错误。"""
    try:
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_MATERIAL_BYTES:
            _reject()
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=lambda _: _reject())
        if not isinstance(value, dict) or set(value) != {"schema_version", "parameters", "execution_context"}:
            _reject()
        if value["schema_version"] != "SQL_PLAN_PARAMETER_MATERIAL.v1":
            _reject()
        parameters = value["parameters"]
        if not isinstance(parameters, list) or len(parameters) > 64:
            _reject()
        for index, parameter in enumerate(parameters, start=1):
            if not isinstance(parameter, dict) or set(parameter) != {"position", "database_type_name", "value"}:
                _reject()
            if type(parameter["position"]) is not int or parameter["position"] != index:
                _reject()
            _text(parameter["database_type_name"], maximum=128)
            if not parameter["database_type_name"].strip():
                _reject()
            _value(parameter["value"])
        context = value["execution_context"]
        required = {"database_name", "schema_name", "role_name"}
        if (not isinstance(context, dict) or not required <= set(context)
            or set(context) - required - {"search_path", "settings"}):
            _reject()
        for name in required:
            _text(context[name], nullable=True, maximum=128)
        if "search_path" in context:
            if not isinstance(context["search_path"], list) or len(context["search_path"]) > 16:
                _reject()
            for entry in context["search_path"]:
                _text(entry, maximum=128)
        if "settings" in context:
            settings = context["settings"]
            if not isinstance(settings, dict) or len(settings) > 32:
                _reject()
            for name, setting in settings.items():
                if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.]{0,127}", name):
                    _reject()
                _text(setting, maximum=1024)
        return value
    except (ValueError, TypeError, RecursionError, OverflowError, UnicodeError):
        raise SqlPlanMaterialInvalid("SQL_PLAN_MATERIAL_INVALID") from None
