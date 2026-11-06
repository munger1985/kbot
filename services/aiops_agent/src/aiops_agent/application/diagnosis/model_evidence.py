"""模型只读取判断所需的数值与口径；完整样本和重放工件仍留在正式证据中。"""

from aiops_agent.contracts.turn_answer import TurnEvidenceFact


HTML_REPORT_TOOLS = frozenset({
    "db.oracle.awr.report", "db.oracle.awr.diff_report", "db.oracle.ash.report", "db.oracle.sql_monitor.report",
})


def evidence_model_payload(fact: TurnEvidenceFact) -> dict:
    """不删原始表格行、来源边界或质量标记，仅移除展示及重复的时序样本。"""
    payload = fact.model_dump(mode="json", exclude={"visualizations", "series_computations"})
    payload["series_computations"] = [
        item.model_dump(mode="json", exclude={"samples"}) for item in fact.series_computations
    ]
    if fact.tool_id in HTML_REPORT_TOOLS:
        payload["rows"] = []
        payload["columns"] = []
    computations = payload["series_computations"]
    if len(computations) > 1:
        # 仅上提完全相同的元数据；数值摘要、行号、原始引用和质量始终逐条保留。
        candidates = (
            "schema_version", "formula_version", "target_id", "source_id", "source_version",
            "binding_id", "unit", "value_kind", "window_start", "window_end",
            "effective_step_seconds", "forecast_horizon_days",
        )
        shared = {key: computations[0][key] for key in candidates
                  if all(item[key] == computations[0][key] for item in computations)}
        payload["series_computation_defaults"] = shared
        payload["series_computations"] = [
            {key: value for key, value in item.items() if key not in shared} for item in computations
        ]
        payload["series_computation_policy"] = "每条 series_computations 继承 series_computation_defaults；各行数值、质量和来源引用不变。原始样本仅用于服务端重放，不要求模型重复计算。"
    return payload


def compact_native_values(projection: dict) -> dict:
    """逐列共享不随行变化的单位与语义，不让模型重复阅读相同字段说明。"""
    fields = {}
    rows = []
    per_row = {"value", "quality", "native_origin_ids"}
    for row in projection["rows"]:
        values = {}
        for name, value in row["values"].items():
            metadata = {key: item for key, item in value.items() if key not in per_row}
            if name in fields and fields[name] != metadata:
                raise ValueError("原生换算列的共享语义不一致")
            fields[name] = metadata
            values[name] = {key: item for key, item in value.items() if key in per_row}
        rows.append({"row_index": row["row_index"], "values": values})
    return {
        **projection, "fields": fields, "rows": rows,
        "field_metadata_policy": "每行 values 的单位、种类和换算语义由同名 fields 项给出；行号、数值、质量和来源仍逐行保留。",
    }
