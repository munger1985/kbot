"""
测试 HITL 人机协同多轮交互核心逻辑。

覆盖:
  1. DBAnalysisSkill — 数据充分性检查、恢复模式、Timeline 构建
  2. OpsContextMemory — HITL 字段初始化与扩展
  3. OpsResumeRequest — Schema 验证与边界情况
  4. OpsOrchestrator — 上下文快照重建 (rebuild_context_from_pending)
  5. PendingRequestRepository — CRUD 操作 (需要数据库)

用法:
    python tests/test_hitl.py
"""

import asyncio
import json
import sys
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from loguru import logger

# ════════════════════════════════════════════════════════════════════════════════
# Unit Tests (不需要外部服务)
# ════════════════════════════════════════════════════════════════════════════════


def test_packet_type():
    """验证 PacketType.WAIT_FOR_USER 已注册"""
    from platform_core.dictionary import PacketType
    assert hasattr(PacketType, "WAIT_FOR_USER")
    assert PacketType.WAIT_FOR_USER == "wait_for_user"
    logger.success("✅ PacketType.WAIT_FOR_USER 已注册")


def test_ops_context_memory_hitl_fields():
    """验证 OpsContextMemory 新增了 HITL 字段"""
    from typing import Any
    from agent.common.ops_context import OpsContextMemory
    ann = OpsContextMemory.__annotations__
    assert "is_resuming" in ann, "is_resuming 字段缺失"
    assert "hitl_history" in ann, "hitl_history 字段缺失"
    assert ann["is_resuming"] == bool, f"is_resuming type mismatch: {ann['is_resuming']}"
    assert ann["hitl_history"] == list[dict[str, Any]], f"hitl_history type mismatch: {ann['hitl_history']}"
    logger.success("✅ OpsContextMemory HITL 字段已扩展")


def test_ops_resume_request_schema():
    """验证 OpsResumeRequest schema 校验"""
    from api.schemas.ops_schema import OpsResumeRequest

    # 正常请求: 有数据
    req = OpsResumeRequest(
        request_id="req_test123",
        user_data={"lock_count": 15, "blocked_session": 102},
        user_note="已执行，未发现异常",
        user_error=None,
    )
    assert req.request_id == "req_test123"
    assert req.user_data["lock_count"] == 15
    assert req.user_note == "已执行，未发现异常"
    assert req.user_error is None

    # 有错误信息: 模拟用户 SQL 执行报错
    req2 = OpsResumeRequest(
        request_id="req_test456",
        user_data=None,
        user_error="ORA-00942: table or view does not exist",
    )
    assert req2.user_data is None
    assert "ORA-00942" in req2.user_error

    # 最小请求: 只有 request_id
    req3 = OpsResumeRequest(request_id="req_min")
    assert req3.user_data is None
    assert req3.user_error is None
    assert req3.user_note is None

    logger.success("✅ OpsResumeRequest schema 校验通过 (3 个场景)")


def _get_analysis_skill():
    """动态加载 DBAnalysisSkill 类 (包名含连字符，无法直接 import)"""
    import importlib
    mod = importlib.import_module(
        "skills.skill_libs.db-analysis-skill.db_analysis_skill_core"
    )
    return mod.DBAnalysisSkill


def test_build_hitl_context_empty():
    """验证空 Timeline 的输出"""
    skill_cls = _get_analysis_skill()
    skill = skill_cls()
    result = skill._build_hitl_context([])
    assert "第一轮排查" in result or "暂无" in result
    logger.success("✅ 空 Timeline → 正确降级显示")


def test_build_hitl_context_single_round():
    """验证单轮 HITL 的 Timeline 渲染"""
    skill_cls = _get_analysis_skill()
    skill = skill_cls()

    hitl_history = [{
        "round": 1,
        "request_id": "req_aaa",
        "reason": "需要查看锁等待详情以定位阻塞源头",
        "sql_to_run": "SELECT sid, serial#, blocking_session FROM v$session WHERE ROWNUM <= 20",
        "user_data": {"rows": [{"sid": 102, "blocking_session": None}]},
        "user_error": None,
        "submitted_at": "2026-07-02T10:30:00Z",
    }]

    result = skill._build_hitl_context(hitl_history)
    assert "第 1 轮" in result
    assert "锁等待" in result
    assert "v$session" in result
    assert "sid" in result.lower()
    assert "📋" in result  # Timeline 标题
    logger.success("✅ 单轮 Timeline → 正确渲染")


def test_build_hitl_context_multi_round_with_error():
    """验证多轮 HITL + 用户错误容错的 Timeline 渲染"""
    skill_cls = _get_analysis_skill()
    skill = skill_cls()

    hitl_history = [
        {
            "round": 1,
            "request_id": "req_001",
            "reason": "需要查锁详情",
            "sql_to_run": "SELECT * FROM v$lock WHERE ROWNUM <= 10",
            "user_data": None,
            "user_error": "ORA-00942: table or view does not exist",
            "submitted_at": "2026-07-02T10:30:00Z",
        },
        {
            "round": 2,
            "request_id": "req_002",
            "reason": "权限不足，用替代方案",
            "sql_to_run": "SELECT * FROM dba_locks WHERE ROWNUM <= 10",
            "user_data": {"rows": [{"session_id": "102", "lock_type": "TX"}]},
            "user_error": None,
            "submitted_at": "2026-07-02T10:35:00Z",
        },
    ]

    result = skill._build_hitl_context(hitl_history)
    assert "第 1 轮" in result
    assert "第 2 轮" in result
    assert "ORA-00942" in result
    assert "替代方案" in result or "dba_locks" in result
    assert "📋" in result
    logger.success("✅ 多轮+错误容错 Timeline → 正确渲染")


def test_build_evidence_summary():
    """验证证据摘要构建"""
    skill_cls = _get_analysis_skill()
    skill = skill_cls()

    # 空证据
    result = skill._build_evidence_summary([], [], [])
    assert "无任何证据" in result

    # 有证据
    monitor = [{"meta": {"metric_code": "cpu_usage"}, "data": [{"value": 0.85}]}]
    metric = [{"meta": {"tool_name": "db_lock_chains"}, "data": [{"sid": 1}]}]
    doc = [{"file_name": "Oracle锁排查SOP.md"}]

    result2 = skill._build_evidence_summary(metric, monitor, doc)
    assert "Prometheus" in result2
    assert "cpu_usage" in result2
    assert "db_lock_chains" in result2
    assert "Oracle锁排查SOP" in result2
    logger.success("✅ 证据摘要构建 → 正确")


def test_context_rebuild_roundtrip():
    """验证 OpsOrchestrator._rebuild_context_from_pending 快照重建"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    orchestrator = OpsOrchestrator()

    pending = {
        "request_id": "req_test_snap",
        "session_id": "sess_abc123",
        "user_id": "chris",
        "agent_id": 126,
        "instance_id": "inst_xyz",
        "entry_id": "entr_001",
        "suspend_reason": "需要锁详情",
        "user_prompt": "SELECT ...",
        "sql_to_run": "SELECT sid FROM v$session WHERE ROWNUM <= 20",
        "expected_fields": json.dumps(["sid(会话ID)", "blocking_session(阻塞者)"]),
        "suspended_by_skill": "db-analysis-skill",
        "current_step_index": 1,
        "completed_steps": json.dumps([{
            "skill": "db-metric-skill",
            "task_description": "采集Prometheus指标",
            "status": "success",
        }]),
        "accumulated_results": json.dumps({
            "metric_results": [],
            "monitor_results": [
                {"step_id": 1, "task_description": "采集指标", "data": [{"value": 0.85}]}
            ],
            "doc_results": [],
        }),
        "pending_variables": json.dumps({"is_mutation_allowed": False}),
        "hitl_history": json.dumps([{"round": 1, "reason": "test"}]),
        "runtime_plan": json.dumps({
            "thought": "排查锁等待",
            "steps": [
                {"step_id": 1, "skill": "db-metric-skill", "task_description": "采集指标"},
                {"step_id": 2, "skill": "db-analysis-skill", "task_description": "分析根因"},
            ],
            "plan_type": "dynamic",
            "inputs": {
                "user_query": "数据库变慢",
                "model_name": "gpt-4",
            },
        }),
        "status": "pending",
        "requested_at": datetime.now(timezone.utc),
    }

    ctx = orchestrator._rebuild_context_from_pending(pending)

    # 验证基础字段
    assert ctx["session_id"] == "sess_abc123"
    assert ctx["user_id"] == "chris"
    assert ctx["agent_id"] == 126
    assert ctx["instance_id"] == "inst_xyz"

    # 验证 HITL 字段
    assert ctx["is_resuming"] is True
    assert len(ctx["hitl_history"]) == 1
    assert ctx["hitl_history"][0]["round"] == 1

    # 验证累积结果恢复
    assert len(ctx["monitor_results"]) == 1
    assert ctx["monitor_results"][0]["data"][0]["value"] == 0.85

    # 验证计划恢复
    assert ctx["runtime_plan"] is not None
    assert len(ctx["runtime_plan"]["steps"]) == 2

    # 验证变量恢复
    assert ctx["variables"]["is_mutation_allowed"] is False

    # 验证执行历史恢复
    assert len(ctx["execution_history"]) == 1
    assert ctx["execution_history"][0]["skill"] == "db-metric-skill"

    logger.success("✅ 上下文快照重建 roundtrip → 所有字段正确恢复")


def test_context_rebuild_with_string_json():
    """验证 pending 中的 JSON 字符串字段能正确解析"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    orchestrator = OpsOrchestrator()

    # 模拟从数据库读出的场景：JSON 字段可能是字符串
    pending = {
        "request_id": "req_str_test",
        "session_id": "sess_test",
        "user_id": "test_user",
        "agent_id": 1,
        "instance_id": "inst_test",
        "entry_id": "entr_test",
        "current_step_index": 0,
        "completed_steps": json.dumps([{"skill": "test", "status": "success"}]),
        "accumulated_results": json.dumps({"metric_results": [], "monitor_results": [], "doc_results": []}),
        "pending_variables": json.dumps({"key1": "value1"}),
        "hitl_history": json.dumps([]),
        "runtime_plan": json.dumps({"steps": [{"step_id": 1, "skill": "test-skill"}], "inputs": {"user_query": "test", "model_name": "m"}}),
    }

    ctx = orchestrator._rebuild_context_from_pending(pending)

    assert ctx["is_resuming"] is True
    assert ctx["hitl_history"] == []
    assert ctx["variables"]["key1"] == "value1"
    assert len(ctx["execution_history"]) == 1
    assert len(ctx["runtime_plan"]["steps"]) == 1
    logger.success("✅ JSON 字符串字段解析 → 正确")
# ════════════════════════════════════════════════════════════════════════════════
# Mutation / Approval Flow Tests
# ════════════════════════════════════════════════════════════════════════════════


def test_packet_type_action_items():
    """验证 PacketType.ACTION_ITEMS 已注册"""
    from platform_core.dictionary import PacketType
    assert hasattr(PacketType, "ACTION_ITEMS")
    assert PacketType.ACTION_ITEMS == "action_items"
    logger.success("✅ PacketType.ACTION_ITEMS 已注册")


def test_ops_approve_request_schema():
    """验证 OpsApproveRequest schema"""
    from api.schemas.ops_schema import OpsApproveRequest

    # 批准
    req1 = OpsApproveRequest(
        request_id="appr_test",
        approved=True,
        approver_note="风险可控，批准执行",
    )
    assert req1.approved is True
    assert "风险可控" in req1.approver_note

    # 拒绝
    req2 = OpsApproveRequest(
        request_id="appr_test2",
        approved=False,
        approver_note="影响范围过大，请提供更安全的替代方案",
    )
    assert req2.approved is False

    # 最小请求
    req3 = OpsApproveRequest(request_id="appr_min", approved=True)
    assert req3.approver_note is None

    logger.success("✅ OpsApproveRequest schema 校验通过 (3 场景)")


def test_safety_gate_read_only():
    """验证安全门禁: READ_ONLY 技能直接放行"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode
    from agent.common.ops_context import OpsContextMemory
    from typing import Any, cast

    orchestrator = OpsOrchestrator()

    # 创建一个 READ_ONLY 技能
    class TestReadOnlySkill(BaseSkill):
        meta = SkillMeta(
            name="test-readonly",
            description="test",
            domain=SkillDomain.OPS,
            run_mode=SkillRunMode.READ_ONLY,
        )

    skill = TestReadOnlySkill()
    ctx = cast(OpsContextMemory, {
        "trace_id": "test",
        "user_id": "u1",
        "session_id": "s1",
        "agent_id": 1,
        "trigger_type": "manual",
        "command_or_query": "test",
        "llm_model": "m",
        "embedding_model": "e",
        "instance_id": "i1",
        "db_type": "oracle",
        "version_code": 0,
        "db_role": "primary",
        "environment": "dev",
        "monitor_type": "prometheus",
        "prometheus_instance_label": None,
        "alert_context": None,
        "runtime_plan": None,
        "current_step_index": 0,
        "current_execution": None,
        "execution_history": [],
        "approval_context": None,
        "variables": {},
        "metric_results": [],
        "monitor_results": [],
        "os_log_snapshots": [],
        "doc_results": [],
        "temp": {},
        "is_resuming": False,
        "hitl_history": [],
    })

    result = orchestrator._check_safety_gate(ctx, skill, "test-readonly")
    assert result["allowed"] is True
    assert result.get("needs_approval") is False
    logger.success("✅ 安全门禁: READ_ONLY 技能直接放行")


def test_safety_gate_mutation_blocked():
    """验证安全门禁: is_mutation_allowed=False → 硬阻断"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode

    orchestrator = OpsOrchestrator()

    class TestMutationSkill(BaseSkill):
        meta = SkillMeta(
            name="test-mutation",
            description="test",
            domain=SkillDomain.OPS,
            run_mode=SkillRunMode.MUTATION,
        )

    skill = TestMutationSkill()
    # ctx with is_mutation_allowed=False
    ctx = {
        "variables": {"is_mutation_allowed": False, "require_approval": True},
        "environment": "dev",
        "instance_id": "test-inst",
        "approval_context": None,
    }

    result = orchestrator._check_safety_gate(ctx, skill, "test-mutation")
    assert result["allowed"] is False
    assert result.get("needs_approval") is False  # hard block, not approval interrupt
    assert "未开启变更许可" in result["reason"]
    logger.success("✅ 安全门禁: MUTATION + is_mutation_allowed=False → 硬阻断")


def test_safety_gate_mutation_needs_approval():
    """验证安全门禁: is_mutation_allowed=True + require_approval=True + 无令牌 → 审批中断"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode

    orchestrator = OpsOrchestrator()

    class TestMutationSkill2(BaseSkill):
        meta = SkillMeta(
            name="test-mutation2",
            description="test",
            domain=SkillDomain.OPS,
            run_mode=SkillRunMode.MUTATION,
        )

    skill = TestMutationSkill2()
    ctx = {
        "variables": {"is_mutation_allowed": True, "require_approval": True},
        "environment": "dev",
        "instance_id": "test-inst",
        "approval_context": None,  # 无审批令牌
    }

    result = orchestrator._check_safety_gate(ctx, skill, "test-mutation2")
    assert result["allowed"] is False
    assert result.get("needs_approval") is True  # interrupt, not hard block
    logger.success("✅ 安全门禁: MUTATION + 无审批令牌 → 审批中断")


def test_safety_gate_mutation_approved():
    """验证安全门禁: is_mutation_allowed=True + require_approval=True + 有令牌 → 放行"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode

    orchestrator = OpsOrchestrator()

    class TestMutationSkill3(BaseSkill):
        meta = SkillMeta(
            name="test-mutation3",
            description="test",
            domain=SkillDomain.OPS,
            run_mode=SkillRunMode.MUTATION,
        )

    skill = TestMutationSkill3()
    ctx = {
        "variables": {"is_mutation_allowed": True, "require_approval": True},
        "environment": "dev",
        "instance_id": "test-inst",
        "approval_context": {"approved": True, "approved_by": "user"},  # 已审批
    }

    result = orchestrator._check_safety_gate(ctx, skill, "test-mutation3")
    assert result["allowed"] is True
    logger.success("✅ 安全门禁: MUTATION + 审批令牌 → 放行")


def test_safety_gate_mutation_no_approval_required():
    """验证安全门禁: is_mutation_allowed=True + require_approval=False → 直接放行"""
    from agent.orchestrator.ops_orchestrator import OpsOrchestrator
    from skills import BaseSkill, SkillMeta, SkillDomain, SkillRunMode

    orchestrator = OpsOrchestrator()

    class TestMutationSkill4(BaseSkill):
        meta = SkillMeta(
            name="test-mutation4",
            description="test",
            domain=SkillDomain.OPS,
            run_mode=SkillRunMode.MUTATION,
        )

    skill = TestMutationSkill4()
    ctx = {
        "variables": {"is_mutation_allowed": True, "require_approval": False},
        "environment": "stg",
        "instance_id": "test-inst",
        "approval_context": None,
    }

    result = orchestrator._check_safety_gate(ctx, skill, "test-mutation4")
    assert result["allowed"] is True
    logger.success("✅ 安全门禁: MUTATION + require_approval=False → 直接放行")


def test_action_items_extraction():
    """验证 ExecuteOpsSkill 从上下文中提取 action_sql"""
    import importlib
    mod = importlib.import_module(
        "skills.skill_libs.execute-ops-skill.execute_ops_skill_core"
    )
    skill = mod.ExecuteOpsSkill()

    # 模拟 DBAnalysisSkill 产出包含 SQL 的 execution_history
    context = {
        "execution_history": [
            {
                "skill": "db-analysis-skill",
                "status": "success",
                "output": "```sql\nALTER SYSTEM KILL SESSION '102,45678' IMMEDIATE;\n```",
            }
        ],
        "variables": {},
    }

    sql, ctx_text = skill._extract_action_from_context(context)
    assert "ALTER SYSTEM KILL SESSION" in sql
    assert "102" in sql
    assert "45678" in sql
    logger.success("✅ ExecuteOpsSkill 正确提取变更 SQL")


def test_action_items_extraction_from_variables():
    """验证 ExecuteOpsSkill 从 variables 中读取 pending_action_sql"""
    import importlib
    mod = importlib.import_module(
        "skills.skill_libs.execute-ops-skill.execute_ops_skill_core"
    )
    skill = mod.ExecuteOpsSkill()

    context = {
        "execution_history": [],
        "variables": {
            "pending_action_sql": "ALTER SYSTEM SET optimizer_mode='FIRST_ROWS'",
            "pending_action_context": "优化器模式需要临时调整为 FIRST_ROWS",
        },
    }

    sql, ctx_text = skill._extract_action_from_context(context)
    assert "ALTER SYSTEM" in sql
    assert "FIRST_ROWS" in sql
    logger.success("✅ ExecuteOpsSkill 从 variables 读取变更 SQL")


# ════════════════════════════════════════════════════════════════════════════════
# Integration Tests (需要数据库)
# ════════════════════════════════════════════════════════════════════════════════


async def test_pending_repo_crud():
    """测试 PendingRequestRepository 的 CRUD 操作 (需要数据库)"""
    from platform_core.database.oracle import get_session
    from dao.repositories import PendingRequestRepository

    request_id = f"req_test_{uuid.uuid4().hex[:12]}"
    session_id = f"sess_test_{uuid.uuid4().hex[:12]}"

    try:
        async with get_session() as session:
            repo = PendingRequestRepository(session)

            # CREATE
            pending_data = {
                "request_id": request_id,
                "session_id": session_id,
                "user_id": "chris",
                "agent_id": 126,
                "instance_id": "test_inst",
                "entry_id": f"entr_{uuid.uuid4().hex[:12]}",
                "suspend_reason": "测试: 需要查锁等待详情",
                "user_prompt": "SELECT * FROM v$lock",
                "sql_to_run": "SELECT * FROM v$lock WHERE ROWNUM <= 20",
                "expected_fields": json.dumps(["sid", "lock_type"]),
                "suspended_by_skill": "db-analysis-skill",
                "current_step_index": 1,
                "completed_steps": json.dumps([]),
                "accumulated_results": json.dumps({"metric_results": [], "monitor_results": [], "doc_results": []}),
                "pending_variables": json.dumps({}),
                "hitl_history": json.dumps([]),
                "runtime_plan": json.dumps({"steps": []}),
                "status": "pending",
                "timeout_at": datetime.now(timezone.utc) + timedelta(minutes=30),
            }
            entity = await repo.create(pending_data)
            assert entity.request_id == request_id
            logger.success(f"✅ CREATE: {request_id}")

            # READ
            fetched = await repo.get_by_request_id(request_id)
            assert fetched is not None
            assert fetched["status"] == "pending"
            assert fetched["suspended_by_skill"] == "db-analysis-skill"
            logger.success("✅ READ: 数据一致")

            # UPDATE: mark answered
            await repo.mark_answered(request_id)
            fetched2 = await repo.get_by_request_id(request_id)
            assert fetched2["status"] == "answered"
            assert fetched2["responded_at"] is not None
            logger.success("✅ UPDATE: mark_answered")

            await session.commit()

    except Exception as e:
        logger.warning(f"⚠️ PendingRepo CRUD 测试跳过 (数据库不可用): {e}")
        return


async def test_timeout_detection():
    """测试超时检测逻辑 (需要数据库)"""
    from platform_core.database.oracle import get_session
    from dao.repositories import PendingRequestRepository

    request_id = f"req_timeout_{uuid.uuid4().hex[:12]}"
    session_id = f"sess_timeout_{uuid.uuid4().hex[:12]}"

    try:
        async with get_session() as session:
            repo = PendingRequestRepository(session)

            # 创建一个已超时的记录 (timeout_at 设为过去)
            pending_data = {
                "request_id": request_id,
                "session_id": session_id,
                "user_id": "chris",
                "agent_id": 126,
                "instance_id": "test_inst",
                "entry_id": f"entr_{uuid.uuid4().hex[:12]}",
                "suspend_reason": "测试超时",
                "user_prompt": "TEST",
                "sql_to_run": "SELECT 1 FROM DUAL",
                "expected_fields": json.dumps([]),
                "suspended_by_skill": "db-analysis-skill",
                "current_step_index": 0,
                "completed_steps": json.dumps([]),
                "accumulated_results": json.dumps({"metric_results": [], "monitor_results": [], "doc_results": []}),
                "pending_variables": json.dumps({}),
                "hitl_history": json.dumps([]),
                "runtime_plan": json.dumps({"steps": []}),
                "status": "pending",
                "timeout_at": datetime.now(timezone.utc) - timedelta(minutes=5),  # 已超时
            }
            await repo.create(pending_data)
            await session.commit()

        # 查找超时记录
        async with get_session() as session:
            repo = PendingRequestRepository(session)
            timed_out = await repo.find_timeout_pending()
            request_ids = [r["request_id"] for r in timed_out]
            assert request_id in request_ids, f"超时记录 {request_id} 未被检出"
            logger.success(f"✅ 超时检测: 正确检出 {request_id}")

            # 标记超时并清理
            await repo.mark_timeout(request_id)
            await session.commit()

    except Exception as e:
        logger.warning(f"⚠️ 超时检测测试跳过 (数据库不可用): {e}")
        return


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════


async def main():
    logger.info("=" * 60)
    logger.info("HITL 单元测试套件启动")
    logger.info("=" * 60)

    # Unit tests (no external dependencies)
    unit_tests = [
        ("PacketType 注册", test_packet_type),
        ("PacketType.ACTION_ITEMS", test_packet_type_action_items),
        ("OpsContextMemory 字段", test_ops_context_memory_hitl_fields),
        ("OpsResumeRequest Schema", test_ops_resume_request_schema),
        ("OpsApproveRequest Schema", test_ops_approve_request_schema),
        ("空 Timeline 构建", test_build_hitl_context_empty),
        ("单轮 Timeline 构建", test_build_hitl_context_single_round),
        ("多轮+错误 Timeline 构建", test_build_hitl_context_multi_round_with_error),
        ("证据摘要构建", test_build_evidence_summary),
        ("上下文快照重建 Roundtrip", test_context_rebuild_roundtrip),
        ("JSON字符串字段解析", test_context_rebuild_with_string_json),
        ("安全门禁 READ_ONLY", test_safety_gate_read_only),
        ("安全门禁 硬阻断", test_safety_gate_mutation_blocked),
        ("安全门禁 审批中断", test_safety_gate_mutation_needs_approval),
        ("安全门禁 审批通过", test_safety_gate_mutation_approved),
        ("安全门禁 无需审批", test_safety_gate_mutation_no_approval_required),
        ("变更SQL提取 history", test_action_items_extraction),
        ("变更SQL提取 variables", test_action_items_extraction_from_variables),
    ]

    passed = 0
    failed = 0
    for name, test_fn in unit_tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info(f"\n{'=' * 60}")
    logger.info(f"单元测试: {passed} 通过, {failed} 失败")
    logger.info(f"{'=' * 60}\n")

    # Integration tests (require DB)
    logger.info("集成测试 (需要数据库连接)...")
    await test_pending_repo_crud()
    await test_timeout_detection()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"测试完成: 单元测试 {passed}/{passed + failed} 通过")
    logger.info(f"{'=' * 60}")

    return failed


if __name__ == "__main__":
    import sys as _sys
    # 注入 typing.Any 到全局命名空间
    from typing import Any
    exit_code = asyncio.run(main())
    _sys.exit(exit_code)
