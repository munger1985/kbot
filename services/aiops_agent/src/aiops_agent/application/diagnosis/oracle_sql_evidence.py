"""Oracle 专属证据续取：完整游标、单次执行、ASH 窗口与只读预算独立核验。"""

from datetime import UTC, datetime, timedelta

from aiops_agent.application.diagnosis.sql_evidence_answer import _subject, _row_dicts, _matches
from aiops_agent.domain.sql_subjects import ORACLE_CURSOR_KEYS as CURSOR_KEYS, normalize_sql_subject
from aiops_agent.contracts.sql_diagnosis import OracleSqlEvidenceRequest
from platform_core.contracts.aiops import InvestigationAction

XPLAN = "db.oracle.statement.xplan"
MONITOR = "db.oracle.statement.monitor"
MONITOR_PLAN = "db.oracle.statement.monitor_plan"
ASH = "db.oracle.statement.ash"
ORACLE_SQL_TOOLS = (XPLAN, MONITOR, MONITOR_PLAN, ASH)
REQUEST_TOOLS = {"XPLAN": XPLAN, "SQL_MONITOR": MONITOR, "ASH": ASH}


def authorized_oracle_sql_tools(snapshot):
    """使用与执行器一致的版本、只读能力和 Pack 门禁，不因查询可执行就推断许可。"""
    from pydantic import ValidationError
    from aiops_agent.diagnostics import DiagnosticRegistry
    from aiops_agent.playbooks import PlaybookRegistry
    from aiops_agent.tools import ToolExecutionSnapshotBuilder
    from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot

    try:
        capability = DbaCapabilitySnapshot.model_validate(snapshot)
    except (ValueError, TypeError, ValidationError):
        return ()
    if str(capability.database_type) != "ORACLE":
        return ()
    builder = ToolExecutionSnapshotBuilder(playbook_registry=PlaybookRegistry.load(),
                                          diagnostic_registry=DiagnosticRegistry.load())
    return tuple(item["tool_id"] for item in builder.discover_tools(capability)
                 if item["tool_id"] in ORACLE_SQL_TOOLS)


def _current(fact):
    return (fact.trust_level == "SOURCE_VERIFIED"
            and fact.source_id != "prior_turn.sql_observation" and fact.followup_run_id is None)


def _attempted(tool_id, evidence, gaps, prior_actions):
    # 包括零行、失败、已调度；不把同一工具的补证失败变成重复规划循环。
    return (any(fact.tool_id == tool_id and _current(fact) for fact in evidence)
            or any((gap.get("source_id") if isinstance(gap, dict) else gap.source_id) == tool_id for gap in gaps)
            or any(action.get("tool_id") == tool_id for action in prior_actions))


def _subjects(task_frame, evidence):
    from aiops_agent.application.diagnosis.sql_diagnosis import sql_scope_resolution

    spec = _subject(task_frame, "ORACLE")
    if spec is None:
        return ()
    subject, (_, identity, _) = spec
    resolution = sql_scope_resolution(subject, identity, "ORACLE", evidence)
    if resolution["status"] not in {"RESOLVED", "COMPARISON", "BOUNDED"}:
        return ()
    result = []
    for candidate in resolution["candidates"][:2]:
        normalized = normalize_sql_subject(candidate["subject_ref"], "ORACLE")
        if normalized and all(key in normalized for key in CURSOR_KEYS):
            result.append((candidate["subject_id"], {key: normalized[key] for key in CURSOR_KEYS}))
    return tuple(result)


def ash_window(task_frame, evidence):
    """以本轮游标原始观测时刻冻结窗口；不随重规划滚动，不解析自由文本时间。"""
    dates = []
    for fact in evidence:
        if fact.tool_id == "db.sql.cursor_details" and _current(fact):
            try:
                stamp = datetime.fromisoformat(fact.captured_at.replace("Z", "+00:00"))
                if stamp.tzinfo is not None:
                    dates.append(stamp.astimezone(UTC))
            except (ValueError, TypeError, AttributeError):
                continue
    seconds = task_frame.get("requested_window_seconds") or 900
    if not dates or isinstance(seconds, bool) or not isinstance(seconds, int) or not 60 <= seconds <= 86400:
        return None
    end = min(dates).replace(microsecond=0)
    return {"begin_time": (end - timedelta(seconds=seconds)).isoformat(),
            "end_time": end.isoformat()}


def oracle_evidence_policy(*, task_frame, evidence, available_tool_ids, gaps=(), prior_actions=()):
    return {
        "available_tools": [tool for tool in ORACLE_SQL_TOOLS if tool in available_tool_ids],
        "attempted_tools": [tool for tool in ORACLE_SQL_TOOLS if _attempted(tool, evidence, gaps, prior_actions)],
        "ash_window": ash_window(task_frame, evidence),
        "ash_window_source": "本轮游标观测时点之前的请求时长；未指定时默认15分钟，不代表历史故障时间",
        "rules": [
            "仅对当前已解析主体按需提出 oracle_evidence_requests，subject_id 必须来自主体清单；不提供SQL、绑定值或猜测执行标识。",
            "SQL Monitor 缺记录不是无问题；ASH是保留的活动样本，不是完整逐次执行记录；样本比例不等于精确耗时。",
            "只有改变当前判断且尚未尝试、当前许可允许的关键证据才申请；不把可选证据缺失作为阻断健康或问题判断的通用条件。",
            "用户明确历史区间时可在ASH请求成对提供带时区begin_time/end_time（最多24小时）；否则使用默认冻结窗口并如实说明，不冒充历史故障窗口。更广历史使用通用受控规划。",
        ],
    }


def pending_oracle_sql_actions(*, task_frame, database_type, evidence, available_tool_ids,
                               gaps=(), prior_actions=(), requests=(), remaining_seconds=None):
    """轻量默认补充 XPLAN/Monitor；只有诊断声明需要时读取 ASH，最多两个完整主体。"""
    if database_type != "ORACLE" or remaining_seconds is not None and remaining_seconds < 15:
        return ()
    subjects = _subjects(task_frame, evidence)
    if not subjects:
        return ()
    available = set(available_tool_ids)
    requested = {(item.kind, item.subject_id): item for item in (
        OracleSqlEvidenceRequest.model_validate(value) for value in requests)}
    actions = []

    def add(tool, parameters, question, semantics="CURRENT_ACTIVITY"):
        if tool not in available or _attempted(tool, evidence, gaps, prior_actions):
            return
        actions.append(InvestigationAction(action_id=f"a{len(actions) + 1}", tool_id=tool, input=parameters,
            question=question, expected_evidence_kind="ORACLE_SQL_DIAGNOSTIC_FACTS",
            measurement_semantics=semantics, optional=True))

    for subject_id, parameters in subjects:
        add(XPLAN, parameters, "补取精确本地游标的 DBMS_XPLAN 谓词、别名及 Outline；不执行原SQL、不读取绑定值")
        add(MONITOR, parameters, "发现精确游标仍保留的 SQL Monitor 单次执行；零行不表示健康")
        # 只能由本轮正式 Monitor 原始行绑定执行，不能借用用户文本或其他子游标。
        for fact in evidence:
            if fact.tool_id != MONITOR or not _current(fact):
                continue
            anchor = next((entry for entry in evidence if entry.tool_id == "db.sql.cursor_details"
                           and _current(entry) and any(_matches(row, parameters, "sql_id")
                               for row in _row_dicts(entry))), None)
            if (anchor is None or anchor.provenance is None or fact.provenance is None
                or not anchor.provenance.physical_identity
                or dict(anchor.provenance.physical_identity) != dict(fact.provenance.physical_identity)):
                continue
            for row in _row_dicts(fact):
                if not all(key in row for key in CURSOR_KEYS) or not _matches(row, parameters, "sql_id"):
                    continue
                try:
                    execution = int(str(row["sql_exec_id"]))
                    native = str(row["sql_exec_start_native"])
                    datetime.strptime(native, '%Y-%m-%dT%H:%M:%S')
                    if execution <= 0:
                        continue
                except (KeyError, ValueError, TypeError):
                    continue
                add(MONITOR_PLAN, {**parameters, "sql_exec_id": execution, "sql_exec_start_native": native},
                    "读取已观测执行的本实例进程行源、I/O和工作区；不把PX进程数据盲加为节点总量")
                break
            else:
                continue
            break
        window = ash_window(task_frame, evidence)
        request = requested.get(("ASH", subject_id))
        if request and request.begin_time is not None:
            window = {"begin_time": request.begin_time.astimezone(UTC).isoformat(timespec="seconds"),
                      "end_time": request.end_time.astimezone(UTC).isoformat(timespec="seconds")}
        if request and window:
            add(ASH, {**parameters, **window}, "按精确游标和冻结窗口读取ASH等待、ON CPU、计划行与阻塞样本",
                "HISTORICAL_SAMPLES")
    return tuple(actions[:6])

def oracle_runtime_context(observations, subjects):
    """只建立同一完整游标/原生计划行与单次执行关系，不汇总或替换 LAST 指标。"""
    result = []
    for subject in subjects:
        subject_id = subject["subject_id"]
        scoped = [(observation, row) for observation in observations for row in observation["records"]
                  if row["role"] == "SUBJECT" and subject_id in row["subject_ids"]
                  and observation["trust_level"] == "SOURCE_VERIFIED"]
        if not scoped:
            continue
        rows_by_tool = {tool: [(observation, row) for observation, row in scoped if observation["tool_id"] == tool]
                        for tool in ORACLE_SQL_TOOLS}
        def pointer(observation, row):
            return {"evidence_ref": observation["evidence_ref"], "row_index": row["row_index"]}
        executions = []
        for observation, row in rows_by_tool[MONITOR]:
            fields = row["fields"]
            key = (fields.get("sql_exec_id"), fields.get("sql_exec_start_native"))
            executions.append({**pointer(observation, row), "sql_exec_id": key[0], "sql_exec_start_native": key[1],
                "process_rows": [pointer(other, item) for other, item in rows_by_tool[MONITOR_PLAN]
                    if (item["fields"].get("sql_exec_id"), item["fields"].get("sql_exec_start_native")) == key]})
        line_links = []
        matching_plans = {item["evidence_ref"] for item in subject.get("plan_assessments", ())
                          if item["identity_match"] == item["physical_match"] == "SOURCE_MATCH"}
        for plan in observations:
            if (plan["tool_id"] != "db.oracle.statement.plan" or not plan.get("sql_plan")
                or plan["evidence_ref"] not in matching_plans):
                continue
            # 正式计划标准化会重排稀疏行号，按与计划投影相同的父子遍历恢复对应关系。
            native = {row["fields"].get("plan_line_id"): row for row in plan["records"]
                      if subject_id in row["subject_ids"]}
            ordered, visited = [], set()
            def visit(parent):
                for line, row in native.items():
                    if line not in visited and row["fields"].get("parent_id") == parent:
                        visited.add(line)
                        ordered.append(row)
                        visit(line)
            visit(None)
            if len(ordered) != len(plan["sql_plan"]["nodes"]):
                continue
            for node, raw in zip(plan["sql_plan"]["nodes"], ordered, strict=True):
                native_line = raw["fields"]["plan_line_id"]
                related = [pointer(observation, row) for tool in (MONITOR_PLAN, ASH)
                    for observation, row in rows_by_tool[tool]
                    if str(row["fields"].get("plan_line_id")) == str(native_line)]
                if related:
                    line_links.append({"plan_evidence_ref": plan["evidence_ref"], "node_id": node["node_id"],
                        "native_plan_line_id": native_line, "related_rows": related})
        result.append({
            "subject_id": subject_id,
            "xplan_rows": [pointer(observation, row) for observation, row in rows_by_tool[XPLAN]],
            "monitor_executions": executions,
            "ash_rows": [pointer(observation, row) for observation, row in rows_by_tool[ASH]],
            "plan_line_links": line_links,
            "limitations": [
                "XPLAN与LAST是采集时点缓存补充，不保证同一次执行；格式化文本不重建或覆盖结构化计划。",
                "Monitor为仍保留的具体执行，行源只覆盖所选实例及返回进程；PX计数、父子节点及累计指标不能直接相加。",
                "ASH是指定窗口内仍保留的活动样本；ON CPU不是精确CPU秒数，样本数不是执行次数，没有样本不证明无问题。",
                "计划行关系仅说明完整游标和计划哈希下的行号对应，不证明LAST与Monitor、ASH来自同一次执行。",
            ],
        })
    return result
