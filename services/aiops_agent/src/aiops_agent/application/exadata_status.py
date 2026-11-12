"""按审核原生类型匹配状态；NUMBER 的等值编码不依赖字符串或浮点近似。"""

from decimal import Decimal, InvalidOperation
import re


NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,6})?")


def match_status(definition, value):
    if definition.status is None or value is None:
        return None
    token = value.strip()
    if definition.native_value_type != "NUMBER":
        return next((item for raw, item in definition.status.values.items()
                     if raw.casefold() == token.casefold()), None)
    if not NUMBER.fullmatch(token):
        return None
    try:
        number = Decimal(token)
        return next((item for raw, item in definition.status.values.items()
                     if NUMBER.fullmatch(raw) and Decimal(raw) == number), None)
    except InvalidOperation:
        return None
