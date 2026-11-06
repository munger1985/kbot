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
