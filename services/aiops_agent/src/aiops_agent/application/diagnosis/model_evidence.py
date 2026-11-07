"""模型只读取判断所需的数值与口径；完整样本和重放工件仍留在正式证据中。"""

from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference
from aiops_agent.domain.evidence.physical_identity import identity_sample_extent


HTML_REPORT_TOOLS = frozenset({
    "db.oracle.awr.report", "db.oracle.awr.diff_report", "db.oracle.ash.report", "db.oracle.sql_monitor.report",
})


def evidence_model_payload(fact: TurnEvidenceFact) -> dict:
    """判断视图保留所有事实行；样本重放和重复数值不要求模型再次阅读。"""
    payload = fact.model_dump(mode="json", exclude={"visualizations", "series_computations", "source_identity_witnesses"})
    result_fields = EvidenceFieldReference.empty_result_fields(fact)
    if result_fields:
        # 只增加结果元数据的引用说明，原始零行、列定义和来源保持不变。
        payload["empty_result_reference"] = {
            "evidence_ref": fact.evidence_ref, "row_index": 0,
            "fields": result_fields,
            "scope": str(fact.measurement_semantics),
            "policy": "这是已返回的完整零行结果，不是虚构数据行；可引用结果行数或采集时间。是否查询成功仍须核对本轮gaps，不将权限探测、无指标采样或失败解释为未发现问题。",
        }
    computations = []
    for trace in fact.series_computations:
        item = trace.model_dump(mode="json", exclude={"samples"})
        row = dict(zip((column["name"] for column in fact.columns), payload["rows"][trace.row_index], strict=True))
        # 只移除与同一行完全一致的字段；未映射或不同的值仍显式保留，不能掩盖不一致。
        item["summary"] = {key: value for key, value in item["summary"].items()
                           if key not in row or row[key] != value}
        for key in ("metric_code", "unit", "value_kind", "forecast_horizon_days"):
            if key in row and item[key] == row[key]:
                del item[key]
        computations.append(item)
    payload["series_computations"] = computations
    payload["source_identity_witnesses"] = [
        {**item.model_dump(mode="json", exclude={"sample_times"}),
         "sample_time_extent": identity_sample_extent(item.sample_times)}
        for item in fact.source_identity_witnesses
    ]
    if fact.tool_id in HTML_REPORT_TOOLS:
        payload["rows"] = []
        payload["columns"] = []
    if computations:
        # 来源、窗口和计算口径可以共享；行号、原始引用和质量始终逐条保留。
        candidates = (
            "schema_version", "formula_version", "target_id", "source_id", "source_version",
            "binding_id", "window_start", "window_end",
            "effective_step_seconds", "forecast_horizon_days",
        )
        shared = {key: computations[0][key] for key in candidates
                  if key in computations[0] and all(key in item and item[key] == computations[0][key] for item in computations)}
        payload["series_computation_defaults"] = shared
        payload["series_computations"] = [
            {key: value for key, value in item.items() if key not in shared} for item in computations
        ]
        payload["series_computation_policy"] = "DIAGNOSIS_MODEL_VIEW.v1：每条计算继承 series_computation_defaults；同名数值及指标、单位、语义读取 row_index 指定的完整 rows 行，summary 只列未重复字段。逐行质量和 input_ref 保留。原始样本与完整计算链由服务端重放，模型不重新计算。身份采样范围不是连续覆盖证明。"
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
