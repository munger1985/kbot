"""诊断未完成时按原始证据逐项交付，不把空结果、采样或部分观测升级为健康结论。"""

from aiops_agent.application.display_dictionary import display_name, display_field_name
from aiops_agent.application.diagnosis.model_evidence import HTML_REPORT_TOOLS
from aiops_agent.contracts.turn_answer import TurnAnswerBlock, TurnEvidenceFact, TurnEvidenceGap


def observed_facts_markdown(evidence: tuple[TurnEvidenceFact, ...], gaps: tuple[TurnEvidenceGap, ...]) -> str:
    """只陈述取证时点、行数、口径和缺口；数值由表格直接呈现。"""
    lines = ["已核验观测（不等同于健康结论）：", ""]
    for fact in evidence:
        if fact.trust_level != "SOURCE_VERIFIED":
            continue
        readable_probe = any(not item.reused and item.state == "READABLE" for item in fact.object_access)
        detail = (
            "仅完成显式列读取探测，按设计未读取数据行"
            if readable_probe else f"返回 {fact.row_count} 行"
        )
        if not fact.row_count and not readable_probe:
            detail += "；仅表示本次查询未返回行，不能据此判定数据库整体正常或对象不存在"
        if fact.truncated:
            detail += "；结果已截断，不能代表完整范围"
        if fact.followup_run_id is not None:
            detail += "；这是后台历史观察，不代表当前状态"
        lines.append(
            f"- `{fact.tool_id}`：{fact.captured_at}，{detail}；口径：{display_name(fact.measurement_semantics)}。"
        )
        lines.extend(f"  - {warning}" for warning in fact.warnings)
        if fact.provenance is not None and fact.provenance.quality_flags:
            lines.append("  - 来源质量标记：" + "、".join(display_name(flag) for flag in fact.provenance.quality_flags))
    if gaps:
        lines.extend(("", "尚未完成或不可用的证据：", ""))
        lines.extend(f"- `{gap.code}`：{gap.detail}" for gap in gaps)
    return "\n".join(lines)


def observed_tables(evidence: tuple[TurnEvidenceFact, ...]) -> tuple[TurnAnswerBlock, ...]:
    """只输出已核验表格的有界预览，完整原始行仍保留在结果证据中。"""
    blocks = []
    for fact in evidence:
        if fact.trust_level != "SOURCE_VERIFIED" or not fact.rows or fact.sql_plan is not None:
            continue
        # 原生报告正文及访问探测不作为普通数据表展示。
        if fact.tool_id in HTML_REPORT_TOOLS or fact.object_access:
            continue
        rows = fact.model_dump(mode="json", include={"rows"})["rows"][:10]
        blocks.append(TurnAnswerBlock(
            block_type="TABLE", schema_version="AIOPS_TABLE_BLOCK.v1",
            payload={
                "title": display_name(fact.tool_id),
                "columns": [
                    {**column, "label": column.get("label") or (
                        display_field_name(column.get("name", ""))
                        if display_field_name(column.get("name", "")) != "观测字段" else column.get("name", "")
                    )}
                    for column in fact.columns
                ],
                "rows": rows, "row_count": fact.row_count,
                "truncated": fact.truncated or len(rows) < fact.row_count,
                "captured_at": fact.captured_at, "measurement_semantics": fact.measurement_semantics,
            },
            evidence_refs=(fact.evidence_ref,),
        ))
    return tuple(blocks)
