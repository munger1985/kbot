"""仅解码代码登记的合同类型，拒绝未知字段和隐式类型转换。"""

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
import math
from types import UnionType
from typing import get_args, get_origin, get_type_hints

from .errors import AlgorithmError, require


def encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, frozenset, list)):
        items = sorted(value) if isinstance(value, frozenset) else value
        return [encode(item) for item in items]
    if isinstance(value, dict):
        require(all(type(key) is str for key in value), "线协议只允许字符串映射键")
        return {key: encode(item) for key, item in value.items()}
    require(type(value) in {str, int, float, bool, type(None)}, "不支持的线协议类型")
    require(type(value) is not float or math.isfinite(value), "线协议数字必须有限")
    return value


def decode(contract: type, value: object) -> object:
    """contract 来自固定注册表或已知嵌套字段，不接受模块名/类名。"""
    origin, arguments = get_origin(contract), get_args(contract)
    if origin is UnionType:
        candidates = [item for item in arguments if type(value) is item]
        if not candidates:
            candidates = [item for item in arguments if value is not None and is_dataclass(item) and type(value) is dict]
        if not candidates:
            candidates = [item for item in arguments if get_origin(item) in {tuple, frozenset} and type(value) is list]
        if not candidates and type(value) is int and float in arguments:
            candidates = [float]
        require(len(candidates) == 1, "联合合同必须能按空值唯一确定类型")
        return decode(candidates[0], value)
    if origin in {tuple, frozenset}:
        require(type(value) is list, "线协议集合必须为数组")
        if len(arguments) == 2 and arguments[1] is Ellipsis or origin is frozenset:
            result = [decode(arguments[0], item) for item in value]
        else:
            require(len(arguments) == len(value), "定长数组形状无效")
            result = [decode(t, item) for t, item in zip(arguments, value)]
        if origin is frozenset:
            require(len(frozenset(result)) == len(result), "线协议集合不能包含重复项")
        return tuple(result) if origin is tuple else frozenset(result)
    if origin is dict:
        require(type(value) is dict and arguments[0] is str, "线协议映射无效")
        return {decode(str, key): decode(arguments[1], item) for key, item in value.items()}
    if isinstance(contract, type) and issubclass(contract, Enum):
        require(type(value) is str, "枚举必须为字符串")
        require(value in {item.value for item in contract}, "未登记的枚举值")
        return contract(value)
    if is_dataclass(contract):
        require(type(value) is dict, "合同必须为对象")
        field_map = {field.name: field for field in fields(contract)}
        require(set(value) <= set(field_map), "合同含未知字段")
        required = {name for name, field in field_map.items()
                    if field.default is MISSING and field.default_factory is MISSING}
        require(required <= set(value), "合同缺少必需字段")
        hints = get_type_hints(contract)
        try:
            return contract(**{name: decode(hints[name], item) for name, item in value.items()})
        except AlgorithmError:
            raise
        except (ValueError, TypeError, OverflowError):
            require(False, "合同字段值无效")
    if contract is float:
        require(type(value) in {float, int} and math.isfinite(value), "浮点字段无效")
        return float(value)
    require(contract in {str, int, bool, type(None)} and type(value) is contract, "标量类型无效")
    return value
