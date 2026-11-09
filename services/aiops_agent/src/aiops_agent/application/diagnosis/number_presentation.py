"""数值的读者展示投影；不覆盖精确证据、计算口径或模型连续原文。"""

from decimal import Decimal, InvalidOperation

from aiops_agent.application.display_dictionary import display_field_name, display_metric_name, format_metric_value
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


NUMBER_PRESENTATION_POLICY = (
    "正文中的测量值优先使用对应 evidence.display_values 的服务端展示值；"
    "常规值最多保留两位小数，省略无意义尾零，容量和带宽使用适当的二进制单位，极小非零值保留有效数字或科学计数法，不能舍入成零。"
    "舍入值用‘约’表述，必须带原指标单位；latest 表述为最新采样值、average 为该采样窗口均值，不直接输出 latest=、average= 或大段原始小数。"
    "统计窗口和速率单位分开：一小时采样中的每秒速率仍是每秒，不能冒充一小时事务总量。"
    "display_values 只用于阅读，阈值比较、推理、证据字段、计算和结构化观测参数仍使用原始精确值；不能因展示舍入改变正常/异常判断或扩大结论。"
    "状态指标按 semantic_role 对应的 display_values 翻译，不输出‘可用性为1’等状态编码；"
    "可用性表示可用或不可用，备份作业结果表示成功或失败，不将所有0/1都解释为健康或异常。"
    "状态编码的平均值、分位数不是状态；缺失或未知值不能解释为正常。可用不等于全部健康，最新状态不证明整个窗口连续可用。"
    "SQL ID、数据库标识、版本、时间、IP、哈希和代码不是测量值，不得改写、舍入或单位换算；未知单位不得猜测。"
)

# 字段来自监控摘要合同，与指标名及数据库类型无关；速率不转累计量。
_SUMMARY_MEASUREMENTS = ("latest", "average", "minimum", "maximum", "first", "sample_p95")
_STATE_MEASUREMENTS = ("latest", "minimum", "maximum", "first")
# 以正式指标语义角色解释枚举；不能按单位或数值给未知状态猜含义。
_STATE_LABELS = {
    "AVAILABILITY": {Decimal(0): "不可用", Decimal(1): "可用"},
    "BACKUP_LAST_RUN_STATUS": {Decimal(0): "失败", Decimal(1): "成功"},
}


def _state_value(value, labels):
    if value is None or value == "":
        return "未取得有效状态"
    try:
        number = Decimal(int(value)) if isinstance(value, bool) else Decimal(str(value))
    except (InvalidOperation, ValueError):
        return "未知状态"
    return labels.get(number, "未知状态") if number.is_finite() else "未取得有效状态"


def monitoring_display_values(fact: TurnEvidenceFact) -> list[dict]:
    if not fact.series_computations:
        return []
    names = tuple(column["name"] for column in fact.columns)
    rows = []
    for trace in fact.series_computations:
        row = dict(zip(names, fact.rows[trace.row_index], strict=True))
        unit = str(row.get("unit") or "")
        role = str(row.get("semantic_role") or "")
        if unit == "state":
            labels = _STATE_LABELS.get(role)
            if labels is None or str(row.get("value_kind") or "") != "STATE":
                continue
            values = {name: _state_value(row[name], labels) for name in _STATE_MEASUREMENTS if name in row}
        elif unit == "unix_seconds" or unit.upper() in {"UNKNOWN", "UNSPECIFIED", ""}:
            continue
        else:
            # 只读当前行，不能用另一序列的摘要替换它；非重复摘要仍保留原证据。
            values = {name: format_metric_value(row[name], unit) for name in _SUMMARY_MEASUREMENTS
                      if name in row and row[name] is not None}
        if values:
            item = {"row_index": trace.row_index, "metric_name": display_metric_name(row.get("metric_code")),
                "values": values, "field_labels": {name: display_field_name(name) for name in values}}
            item["field_labels"]["latest"] = "最新采样状态" if unit == "state" else "最新采样值"
            if unit == "state":
                item["observed_states"] = list(dict.fromkeys(_state_value(value, labels) for _, value in trace.samples))
                item["state_policy"] = (
                    "按本指标语义解释状态；minimum/maximum 只表明窗口内出现过对应状态，不代表持续时长。"
                    "observed_states 仅描述已返回有效采样中的状态，不证明无采样间隔或整个窗口连续健康；"
                    "状态编码的 average/sample_p95 不作为业务状态，也不自动换算可用率。"
                )
            rows.append(item)
    return rows
