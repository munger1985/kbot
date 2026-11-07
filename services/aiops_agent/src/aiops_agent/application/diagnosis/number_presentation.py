"""数值的读者展示投影；不覆盖精确证据、计算口径或模型连续原文。"""

from aiops_agent.application.display_dictionary import format_metric_value
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


NUMBER_PRESENTATION_POLICY = (
    "正文中的测量值优先使用对应 evidence.display_values 的服务端展示值；"
    "常规值最多保留两位小数，省略无意义尾零，容量和带宽使用适当的二进制单位，极小非零值保留有效数字或科学计数法，不能舍入成零。"
    "舍入值用‘约’表述，必须带原指标单位；latest 表述为最新采样值、average 为该采样窗口均值，不直接输出 latest=、average= 或大段原始小数。"
    "统计窗口和速率单位分开：一小时采样中的每秒速率仍是每秒，不能冒充一小时事务总量。"
    "display_values 只用于阅读，阈值比较、推理、证据字段、计算和结构化观测参数仍使用原始精确值；不能因展示舍入改变正常/异常判断或扩大结论。"
    "SQL ID、数据库标识、版本、时间、IP、哈希和代码不是测量值，不得改写、舍入或单位换算；未知单位不得猜测。"
)

# 字段来自监控摘要合同，与指标名及数据库类型无关；速率不转累计量。
_SUMMARY_MEASUREMENTS = ("latest", "average", "minimum", "maximum", "first", "sample_p95")


def monitoring_display_values(fact: TurnEvidenceFact) -> list[dict]:
    if not fact.series_computations:
        return []
    names = tuple(column["name"] for column in fact.columns)
    rows = []
    for trace in fact.series_computations:
        row = dict(zip(names, fact.rows[trace.row_index], strict=True))
        unit = str(row.get("unit") or "")
        if unit in {"state", "unix_seconds"} or unit.upper() in {"UNKNOWN", "UNSPECIFIED", ""}:
            continue
        # 只读当前行，不能用另一序列的摘要替换它；非重复摘要仍保留原证据。
        values = {name: format_metric_value(row[name], unit) for name in _SUMMARY_MEASUREMENTS
                  if name in row and row[name] is not None}
        if values:
            rows.append({"row_index": trace.row_index, "values": values})
    return rows
