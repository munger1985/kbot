"""正式报告的冻结输入与状态展示，不签发执行授权或重新推断诊断结论。"""

from aiops_agent.application.configuration.common import sha256_json

def report_template_for_source(*, template, plan, source_kind):
    """生成和重放使用同一实际模板，不能比较请求模板后再静默替换。"""
    subject = dict(dict(plan.get("answer_context") or {}).get("task_frame") or {}).get("subject_ref") or {}
    prefix = {"MYSQL_DIGEST": "mysql", "POSTGRESQL_QUERY_ID": "postgresql"}.get(subject.get("subject_type"))
    if source_kind == "CHAT" and prefix:
        from aiops_agent.application.reporting import resolve_system_template
        from aiops_agent.application.errors import validation_failed
        resolved = resolve_system_template(f"system:{prefix}.sql_healthcheck")
        if resolved is None:
            raise validation_failed("SQL Healthcheck模板未注册")
        return resolved
    return template


def report_input_hash(*, source_artifact, source, template, period_kind="AD_HOC", period_start=None, period_end=None):
    """以确切来源、投影及模板判断重放，排除点击生成时的当前时钟。"""
    return sha256_json({
        "schema_version": "REPORT_INPUT.v1",
        "artifact": {"artifact_id": str(source_artifact.artifact_id),
            "content_hash": source_artifact.content_hash, "schema_version": source_artifact.schema_version},
        "source": source,
        "template": {"template_ref": template.template_ref, "version": template.version,
            "content_hash": template.content_hash},
        "period": {"kind": period_kind,
            "start": period_start if period_kind != "AD_HOC" else None,
            "end": period_end if period_kind != "AD_HOC" else None},
    })


def report_action_items(scope):
    """区分人工登记、动作后置条件与业务效果；没有证据不升级验证状态。"""
    items = [str(value) for value in scope.get("actions") or () if value]
    verification = dict(scope.get("verification") or {})
    human = scope.get("action_source") == "USER_PROVIDED" or verification.get("method") == "USER_REPORTED"
    prefix = "人工登记的处置（非平台执行证明）" if human else "已登记处置（执行来源以原始回执为准）"
    items.extend(f"{prefix}：{value}" for value in scope.get("executed_actions") or () if value)
    if not verification:
        return items
    if human:
        result = {"RESOLVED": "已报告恢复", "IMPROVED": "已报告改善", "NOT_ACHIEVED": "未达到预期",
            "ADVERSE": "出现不利影响", "NOT_VERIFIED": "未验证"}.get(verification.get("result"), "未确认")
        items.append(f"人工反馈结果：{result}；不代替受控测量或已验证业务效果。")
        return items
    labels = {
        "result": ("验证结果", {"RESOLVED": "症状恢复条件满足", "IMPROVED": "改善条件满足",
            "UNCHANGED": "未达到改善条件", "DEGRADED": "发生退化", "INCONCLUSIVE": "无法判断"}),
        "direct_effect": ("动作后置条件", {"ACHIEVED": "已达到", "NOT_ACHIEVED": "未达到",
            "ADVERSE": "出现不利影响", "UNKNOWN": "无法判断"}),
        "measured_effect": ("可比窗口业务效果", {"IMPROVED": "改善", "UNCHANGED": "未改善",
            "DEGRADED": "退化", "INCONCLUSIVE": "无法判断"}),
        "attribution_strength": ("措施效果归因", {"SUPPORTED": "有对照证据支持",
            "LIMITED": "观察性关联，归因有限", "UNKNOWN": "无法判断"}),
    }
    for key, (label, values) in labels.items():
        if key in verification:
            items.append(f"{label}：{values.get(verification[key], '无法判断')}。")
    for key, label in (("symptom_recovered", "目标症状恢复条件"), ("guardrails_passed", "护栏条件")):
        if key in verification:
            value = verification[key]
            items.append(f"{label}：{'满足' if value is True else '不满足' if value is False else '未确认'}。")
    return items


def report_goal_items(scope):
    """逐个保留原调查目标，不把报告生成、根因确认和业务恢复混为一谈。"""
    states = {"SATISFIED": "已回答", "PARTIAL": "部分回答", "BLOCKED": "受阻", "NOT_APPLICABLE": "不适用"}
    items = []
    for investigation in scope.get("investigation_goals") or ():
        turn = f"第 {investigation['turn_no']} 轮 · " if investigation.get("turn_no") else ""
        for goal in investigation.get("goals") or ():
            item = f"{turn}目标 {goal['goal_id']}：{states.get(goal['status'], '未确认')}"
            if goal.get("explanation"):
                item += "；" + goal["explanation"]
            items.append(item)
    return items
