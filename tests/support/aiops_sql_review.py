"""把领域诊断夹具转换为模型实际返回的精简合同，不绕过生产展开与校验。"""

from aiops_agent.contracts.sql_diagnostic_response import SqlDiagnosticResponse


def diagnostic_response(draft, plan=None):
    data = draft.model_dump(mode="json")
    conclusions = data.pop("conclusions")
    data.pop("schema_version", None)
    data.pop("unresolved_questions", None)
    data["findings"] = [{key: value for key, value in item.items()
                         if key not in {"applicable_time", "alternatives"}} for item in conclusions]
    if plan is not None:
        data["plan_reviews"] = [{"evidence_ref": plan.evidence_ref, "node_ids": [1],
            "conclusion_id": conclusions[0]["conclusion_id"],
            "mechanism": "节点 1 读取关联对象并向上游提供输入，须结合原生行数和循环口径解释访问工作量。",
            "impact": "当前计划用于界定访问机制，不把单个节点或优化器成本直接判为性能故障。"}]
    return SqlDiagnosticResponse.model_validate(data)
