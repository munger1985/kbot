"""原生输出单位的精确投影；换算不使平均值、最大值或取整值变成可加计数器。"""

from decimal import localcontext

from aiops_agent.domain.evidence.observations import parse_numeric_sample


def canonical_native_value(field, observed_value):
    """只换算已返回输出，不还原 ROUND/CAST 丢失的原始计数器精度。"""
    number = parse_numeric_sample(observed_value)
    if number is None:
        return None
    with localcontext() as context:
        context.prec = max(80, len(number.as_tuple().digits) + abs(number.as_tuple().exponent) + 64)
        return number * field.scale_to_canonical
