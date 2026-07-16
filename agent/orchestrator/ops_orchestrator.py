# agent/orchestrator/ops_orchestrator.py

import json
import uuid
from loguru import logger
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator, cast
from fastapi import BackgroundTasks

from core.dictionary import PacketType
from core.database import db_instance
from agent.memory import MemoryService
from skills.skill_manager import SkillManager
from skills import SkillRuntime, SkillRunMode
from agent.planner.ops_planner import OpsTaskPlanner
from agent.orchestrator import DocOrchestrator
from services.basic import AgentService
from services.basic import OpsAgentConfService
from agent.common.ops_context import OpsContextMemory
from agent.common.skill_context import ExecutionPlan
from utils.clients import OpsDBExecutor
from utils.monitor import PrometheusClient, ZabbixProvider, UnifiedMetricRegistry
from dao.repositories import PendingRequestRepository


DISPLAY_PACKET_TYPES = {
    PacketType.THOUGHT,
    PacketType.ANSWER,
    PacketType.ERROR,
    PacketType.DOC_RESULTS,
    PacketType.MONITOR_RESULTS,
    PacketType.METRIC_RESULTS,
    PacketType.WARNING,
    PacketType.REQUIRE_APPROVAL,
    PacketType.ACTION_ITEMS,
    PacketType.CALL,
    PacketType.WAIT_FOR_USER,
    PacketType.DONE
}


class OpsOrchestrator:
    """
    智能故障自愈核心流水线编排器 (UI 强管控·精准定靶版) v3 — 支持 HITL 人机协同
    """

    def __init__(self):
        self.skill_manager = SkillManager()
        self.memory_service = MemoryService()
        self.agent_service = AgentService()
        self.ops_agent_conf_service = OpsAgentConfService()
        self.doc_orchestrator = DocOrchestrator()

        # --- 监控数据源基础设施 ---
        self.metric_registry = UnifiedMetricRegistry()
        self.prometheus_client = PrometheusClient()
        self.zabbix_client = ZabbixProvider()
        self.ops_db_executor = OpsDBExecutor()

        self.planner = OpsTaskPlanner(
            skill_manager=self.skill_manager,
            doc_orchestrator=self.doc_orchestrator,
            metric_registry=self.metric_registry,
        )

    async def execute_ops_stream_pipeline(
        self,
        background_tasks: BackgroundTasks,
        user_id: str,
        session_id: str,
        agent_id: int,
        question: str,
        instance_id: str,
        trigger_type: str = "manual",
        client_time: str | None = None,
        client_tz: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:

        # --- 0. 全周期元数据计算 ---
        start_time = datetime.now(timezone.utc)
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        entry_id = str(uuid.uuid4())

        # 从底座获取模型参数
        model_params = await self.agent_service.get_agent_model_params(agent_id)
        llm_model = model_params.llm_model
        embedding_model = model_params.txt_embedding_model

        # 确保缓存层就绪
        await self.memory_service.ensure_session_exists(
            session_id=session_id, user_id=user_id, agent_id=agent_id, question=question
        )

        # --- 1. 严格按照 OpsContextMemory 定义实例化强类型总线 ---
        ctx: OpsContextMemory = {
            "trace_id": f"trace-{uuid.uuid4().hex[:12]}",
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "trigger_type": cast(Any, trigger_type),
            "command_or_query": question,
            "client_time": client_time or "",
            "client_tz": client_tz or "",
            "llm_model": llm_model,
            "embedding_model": embedding_model,

            "instance_id": instance_id,
            "db_type": "oracle",
            "version_code": 0,
            "db_role": "primary",
            "environment": "dev",
            "monitor_type": "prometheus",
            "prometheus_instance_label": None,
            "zabbix_host_name": None,

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

            # HITL 人机协同
            "is_resuming": False,
            "hitl_history": [],
        }

        # --- 1.2 拓扑资产锁定与安全策略注入 ---
        logger.info(f"[{ctx['trace_id']}] 自愈流水线启动 | 用户: {user_id} | 实例: {instance_id} | 问题: {question[:80]}")
        yield {"type": PacketType.THOUGHT, "content": "正在锁定指定物理实例资产并注入安全边界策略...\n"}

        try:
            target_db = await self.ops_agent_conf_service.get_instance_detail_by_id(instance_id)

            if target_db:
                ctx["db_type"] = target_db["db_type"]
                ctx["version_code"] = target_db["version_code"]
                ctx["db_role"] = target_db["db_role"]
                ctx["environment"] = cast(Any, target_db["environment"])
                ctx["monitor_type"] = target_db.get("monitor_type", "prometheus")
                ctx["prometheus_instance_label"] = target_db.get("prometheus_instance_label")
                ctx["zabbix_host_name"] = target_db.get("zabbix_host_name")

                ctx["variables"]["is_mutation_allowed"] = target_db["is_mutation_allowed"]
                ctx["variables"]["require_approval"] = target_db["require_approval"]
                ctx["variables"]["max_daily_execution"] = target_db["max_daily_execution"]
                ctx["variables"]["security_level"] = target_db["security_level"]

                logger.success(f"[{ctx['trace_id']}] 资产网关锁定成功 | 实例: {ctx['instance_id']} ({ctx['db_type']})")
                yield {
                    "type": PacketType.THOUGHT,
                    "content": f"🎯 已成功锁定目标实例: `{target_db['instance_name']}` | 环境: `{ctx['environment'].upper()}` | 引擎: `{ctx['db_type'].upper()}`\n"
                }
            else:
                yield {
                    "type": PacketType.ERROR,
                    "content": "❌ 安全熔断: 未在系统中检索到该指定的物理数据库资产, 流水线强制退出。"
                }
                yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}
                return

        except Exception as asset_err:
            logger.error(f"[Orchestrator] 资产中心上下文注入阶段发生崩溃: {str(asset_err)}")
            yield {"type": PacketType.ERROR, "content": "⚠️ 运维自愈总线连接资产中心失败, 部分深度内核诊断指标可能无法加载。"}

        # --- 2. 接入网关改写层（Rewrite Gateway） ---
        try:
            async for plan_packet in self.planner.generate_plan(ctx=ctx):
                p_type = plan_packet.get("type")
                if p_type in DISPLAY_PACKET_TYPES:
                    yield plan_packet

            logger.success(f"[{ctx['trace_id']}] 任务编排蓝图规划成功, 准备驱动原子技能流水线。")

        except Exception as plan_err:
            logger.error(f"[Orchestrator] 核心规划控制面发生崩溃: {plan_err}")
            yield {
                "type": PacketType.ERROR,
                "content": "❌ 安全熔断: 核心大脑规划任务失败, 自愈总线终止执行。"
            }
            yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}
            return

        # --- 注入监控与诊断基础设施引用 ---
        ctx["variables"]["_prometheus_client"] = self.prometheus_client
        ctx["variables"]["_zabbix_client"] = self.zabbix_client
        ctx["variables"]["_metric_registry"] = self.metric_registry
        ctx["variables"]["_ops_db_executor"] = self.ops_db_executor

        # --- 3. 驱动强类型状态机执行线性原子技能 ---
        plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
        final_answer_accumulator = ""

        for idx, step in enumerate(plan_steps):
            ctx["current_step_index"] = idx

            runtime = SkillRuntime(context=ctx)
            exec_info = runtime.create_execution_context(step_config=step)
            skill_name = exec_info["skill"]

            ctx["current_execution"] = cast(Any, exec_info)

            yield {
                "type": PacketType.CALL,
                "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}
            }

            skill_instance = self.skill_manager.get_skill_instance(skill_name)
            if not skill_instance:
                exec_info.update({"status": "failed", "error": f"组件 {skill_name} 未激活"})
                ctx["execution_history"].append(cast(Any, exec_info))
                yield {"type": PacketType.ERROR, "content": f"⚠️ 关键自愈组件 [{skill_name}] 离线, 本步骤跳过。"}
                continue

            # --- 🔒 安全熔断门禁 (v2: 支持审批中断) ---
            gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
            if not gate_result["allowed"]:
                if gate_result.get("needs_approval"):
                    # ──── 审批中断 ────
                    approval_request_id = str(uuid.uuid4())
                    action_sql = ctx["variables"].get("pending_action_sql", "")
                    action_impact = ctx["variables"].get("pending_action_impact", "")
                    action_rollback = ctx["variables"].get("pending_action_rollback", "")
                    action_risk = ctx["variables"].get("pending_action_risk_level", "medium")

                    logger.info(
                        f"[{ctx['trace_id']}] 🔴 审批中断触发 | "
                        f"Skill: {skill_name} | Step: {idx} | "
                        f"ApprovalID: {approval_request_id}"
                    )

                    yield {
                        "type": PacketType.REQUIRE_APPROVAL,
                        "content": {
                            "request_id": approval_request_id,
                            "skill_name": skill_name,
                            "reason": gate_result["reason"],
                            "action_sql": action_sql,
                            "impact": action_impact,
                            "rollback_sql": action_rollback,
                            "risk_level": action_risk,
                            "instance_id": ctx["instance_id"],
                            "environment": ctx["environment"],
                        }
                    }

                    await self._suspend_for_approval(
                        ctx=ctx,
                        approval_request_id=approval_request_id,
                        current_step_index=idx,
                        entry_id=entry_id,
                        action_sql=action_sql,
                        action_impact=action_impact,
                        action_rollback=action_rollback,
                        skill_name=skill_name,
                    )

                    yield {
                        "type": PacketType.DONE,
                        "content": {
                            "entry_id": entry_id,
                            "status": "awaiting_approval",
                            "request_id": approval_request_id,
                        }
                    }
                    return
                else:
                    # Hard block
                    exec_info.update({"status": "blocked", "error": gate_result["reason"]})
                    ctx["execution_history"].append(cast(Any, exec_info))
                    yield {"type": PacketType.ERROR, "content": f"🚫 安全熔断: {gate_result['reason']}"}
                    continue

            try:
                _monitor_snapshot = len(ctx.get("monitor_results", []))
                _metric_snapshot = len(ctx.get("metric_results", []))

                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")

                    # 数据沉淀区
                    if p_type == PacketType.MONITOR_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["monitor_results"].append({
                                "step_id": step.get("step_id"),
                                "task_description": step.get("task_description") or exec_info.get("resolved_input"),
                                "data": content["data"],
                                "meta": content.get("meta", {})
                            })
                    elif p_type == PacketType.METRIC_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["metric_results"].append({
                                "step_id": step.get("step_id"),
                                "task_description": step.get("task_description") or exec_info.get("resolved_input"),
                                "data": content["data"],
                                "meta": content.get("meta", {})
                            })

                    # ──── HITL: 中断检测 ────
                    if p_type == PacketType.WAIT_FOR_USER:
                        if not isinstance(content, dict):
                            continue
                        suspend_ctx = content
                        request_id = suspend_ctx["request_id"]

                        logger.info(
                            f"[{ctx['trace_id']}] 🔴 HITL 中断触发 | "
                            f"Skill: {skill_name} | Step: {idx} | "
                            f"RequestID: {request_id}"
                        )

                        await self._suspend_execution(
                            ctx=ctx,
                            suspend_ctx=suspend_ctx,
                            request_id=request_id,
                            current_step_index=idx,
                            entry_id=entry_id,
                            start_time=start_time,
                        )

                        # 将中断的步骤记录到 execution_history，供恢复时下游技能读取其产出
                        exec_info["status"] = "suspended"
                        if final_answer_accumulator:
                            exec_info["answer"] = final_answer_accumulator.strip()
                        ctx["execution_history"].append(cast(Any, exec_info))
                        ctx["current_execution"] = None

                        yield packet
                        yield {
                            "type": PacketType.DONE,
                            "content": {
                                "entry_id": entry_id,
                                "status": "suspended",
                                "request_id": request_id,
                            }
                        }
                        return

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                exec_info.update({"status": "success"})
                # 检测执行类技能是否失败
                action_result = ctx.get("variables", {}).get("action_result", {})
                if isinstance(action_result, dict) and action_result.get("status") in ("failed", "error"):
                    exec_info["status"] = "failed"
                    exec_info["error"] = action_result.get("error", "执行失败")
                    ctx["execution_history"].append(cast(Any, exec_info))
                    ctx["current_execution"] = None
                    break
                output_var = exec_info.get("output_var")
                if output_var:
                    new_monitor = ctx.get("monitor_results", [])[_monitor_snapshot:]
                    new_metric = ctx.get("metric_results", [])[_metric_snapshot:]
                    step_data = {"monitor": new_monitor, "metric": new_metric}
                    ctx["variables"][output_var] = json.dumps(step_data, ensure_ascii=False, default=str)
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None

            except Exception as e:
                logger.error(f"[Orchestrator] 执行自愈组件 [{skill_name}] 发生非致命中断: {e}")
                exec_info.update({"status": "failed", "error": str(e)})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None
                continue

        # --- 4. 闭环落库 ---
        action_result = ctx.get("variables", {}).get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("status") == "failed":
            final_answer_accumulator = f"❌ 变更执行失败: {action_result.get('error', '未知错误')}"
            yield {
                "type": PacketType.ANSWER,
                "content": final_answer_accumulator,
            }
        response_time = datetime.now(timezone.utc)
        plan_skills_trace = [s.get("skill") for s in plan_steps] if plan_steps else []

        safe_variables = {
            k: (
                json.loads(v) if isinstance(v, str) and len(v) > 0
                and v[0] in ('{', '[') else v
            )
            for k, v in ctx["variables"].items()
            if not k.startswith("_") and not hasattr(v, '__dict__')
        }

        prepared_data_payload = {
            "standalone_query": ctx["command_or_query"],
            "search_keywords": ctx["command_or_query"],
            "turn_type": "task_oriented",
            "turn_entities": safe_variables,
            "new_state": safe_variables,
            "active_topic": "AIOps内核指标探测与故障自愈",
            "current_plan": {"skill_sequence": plan_skills_trace, "total_steps": len(plan_skills_trace)},
            "thought": ctx["runtime_plan"]["thought"] if ctx["runtime_plan"] else "",
            "metric_results_snapshot": ctx.get("metric_results", []),
            "doc_results_snapshot": ctx.get("doc_results", []),
        }

        background_tasks.add_task(
            self.memory_service.persist_and_reflect_memory,
            session_id=session_id,
            user_id=user_id,
            entry_id=entry_id,
            raw_question=question,
            answer=final_answer_accumulator.strip() or "自动化自愈 SOP 链路安全执行完毕。",
            model_params=model_params,
            prepared_data=prepared_data_payload,
            context_memory=cast(Any, ctx),
            request_time=start_time,
            response_time=response_time
        )

        logger.success(f"[Orchestrator] 运维自愈强类型流水线圆满结束, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ==================================================================
    # 安全门禁 v2
    # ==================================================================

    def _check_safety_gate(
        self,
        ctx: OpsContextMemory,
        skill_instance: Any,
        skill_name: str
    ) -> dict[str, Any]:
        """
        🔒 运维安全熔断门禁 (v2 — 支持审批中断):
        三重校验: 变更许可(硬阻断) → 审批门禁(软中断) → 生产环境警告
        """
        skill_meta = getattr(skill_instance, "meta", None)
        skill_run_mode = getattr(skill_meta, "run_mode", SkillRunMode.READ_ONLY) if skill_meta else SkillRunMode.READ_ONLY

        if skill_run_mode == SkillRunMode.READ_ONLY:
            return {"allowed": True}

        variables = ctx.get("variables", {})
        environment = ctx.get("environment", "prod")
        instance_id = ctx.get("instance_id", "unknown")

        # 1. 变更许可检查 (硬阻断)
        is_mutation_allowed = variables.get("is_mutation_allowed", False)
        if not is_mutation_allowed:
            logger.warning(
                f"[SafetyGate] 拦截高危自愈动作 | Skill: {skill_name} | "
                f"实例: {instance_id} | 原因: 该实例未开启变更许可"
            )
            return {
                "allowed": False,
                "needs_approval": False,
                "reason": (
                    f"自愈组件 [{skill_name}] 属于变更类高危操作, 但实例 `{instance_id}` "
                    f"未开启变更许可 (is_mutation_allowed=False)。请联系 DBA 管理员开启。"
                )
            }

        # 2. 审批门禁检查 (软中断)
        require_approval = variables.get("require_approval", True)
        if require_approval:
            approval_ctx = ctx.get("approval_context")
            if not approval_ctx or not approval_ctx.get("approved"):
                logger.warning(
                    f"[SafetyGate] 需要审批 | Skill: {skill_name} | "
                    f"实例: {instance_id} | 触发审批中断"
                )
                return {
                    "allowed": False,
                    "needs_approval": True,
                    "reason": (
                        f"自愈组件 [{skill_name}] 即将执行高危变更操作。\n"
                        f"实例: `{instance_id}` | 环境: `{environment}`\n"
                        f"请在前端确认风险后授权执行。"
                    )
                }

        # 3. 生产环境警告
        if environment == "prod":
            logger.warning(
                f"[SafetyGate] ⚠️ 生产环境高危动作即将执行 | Skill: {skill_name} | "
                f"实例: {instance_id} | 审批令牌已校验通过"
            )

        logger.info(
            f"[SafetyGate] ✅ 安全门禁通过 | Skill: {skill_name} | "
            f"实例: {instance_id} | 环境: {environment}"
        )
        return {"allowed": True, "needs_approval": False}

    # ==================================================================
    # HITL: 挂起与恢复
    # ==================================================================

    async def _suspend_execution(
        self,
        ctx: OpsContextMemory,
        suspend_ctx: dict[str, Any],
        request_id: str,
        current_step_index: int,
        entry_id: str,
        start_time: datetime,
    ) -> None:
        """持久化完整执行快照到 kbot_ops_pending_request"""
        timeout_at = datetime.now(timezone.utc) + timedelta(minutes=30)

        pending_data = {
            "request_id": request_id,
            "session_id": ctx["session_id"],
            "user_id": ctx["user_id"],
            "agent_id": ctx["agent_id"],
            "instance_id": ctx["instance_id"],
            "entry_id": entry_id,
            "suspend_reason": suspend_ctx.get("reason", ""),
            "user_prompt": suspend_ctx.get("sql_to_run", ""),
            "sql_to_run": suspend_ctx.get("sql_to_run", ""),
            "expected_fields": json.dumps(
                suspend_ctx.get("expected_fields", []), ensure_ascii=False
            ),
            "suspended_by_skill": suspend_ctx.get("suspended_by", "unknown"),
            "current_step_index": current_step_index,
            "completed_steps": json.dumps(
                ctx.get("execution_history", []), default=str, ensure_ascii=False
            ),
            "accumulated_results": json.dumps({
                "metric_results": ctx.get("metric_results", []),
                "monitor_results": ctx.get("monitor_results", []),
                "doc_results": ctx.get("doc_results", []),
            }, default=str, ensure_ascii=False),
            "pending_variables": json.dumps({
                k: v for k, v in ctx["variables"].items()
                if not k.startswith("_")
            }, default=str, ensure_ascii=False),
            "hitl_history": json.dumps(
                ctx.get("hitl_history", []), default=str, ensure_ascii=False
            ),
            "runtime_plan": json.dumps(
                ctx.get("runtime_plan"), default=str, ensure_ascii=False
            ),
            "status": "pending",
            "timeout_at": timeout_at,
        }

        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            await repo.create(pending_data)

        logger.info(
            f"[HITL Suspend] request_id={request_id} | "
            f"step={current_step_index} | 快照已持久化"
        )

    async def _suspend_for_approval(
        self,
        ctx: OpsContextMemory,
        approval_request_id: str,
        current_step_index: int,
        entry_id: str,
        action_sql: str,
        action_impact: str,
        action_rollback: str,
        skill_name: str,
    ) -> None:
        """审批挂起: 持久化执行状态，等待用户审批"""
        timeout_at = datetime.now(timezone.utc) + timedelta(minutes=30)

        pending_data = {
            "request_id": approval_request_id,
            "session_id": ctx["session_id"],
            "user_id": ctx["user_id"],
            "agent_id": ctx["agent_id"],
            "instance_id": ctx["instance_id"],
            "entry_id": entry_id,
            "suspend_reason": f"等待审批: {skill_name}",
            "user_prompt": (
                f"## 待审批的变更操作\n\n"
                f"**SQL**:\n```sql\n{action_sql}\n```\n\n"
                f"**影响**: {action_impact}\n\n"
                f"**回滚**: {action_rollback}\n"
            ),
            "sql_to_run": action_sql,
            "expected_fields": json.dumps({
                "type": "approval",
                "action_sql": action_sql,
                "impact": action_impact,
                "rollback_sql": action_rollback,
            }, ensure_ascii=False),
            "suspended_by_skill": skill_name,
            "current_step_index": current_step_index,
            "completed_steps": json.dumps(
                ctx.get("execution_history", []), default=str, ensure_ascii=False
            ),
            "accumulated_results": json.dumps({
                "metric_results": ctx.get("metric_results", []),
                "monitor_results": ctx.get("monitor_results", []),
                "doc_results": ctx.get("doc_results", []),
            }, default=str, ensure_ascii=False),
            "pending_variables": json.dumps({
                k: v for k, v in ctx["variables"].items()
                if not k.startswith("_")
            }, default=str, ensure_ascii=False),
            "hitl_history": json.dumps(
                ctx.get("hitl_history", []), default=str, ensure_ascii=False
            ),
            "runtime_plan": json.dumps(
                ctx.get("runtime_plan"), default=str, ensure_ascii=False
            ),
            "status": "pending",
            "timeout_at": timeout_at,
        }

        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            await repo.create(pending_data)

        logger.info(
            f"[HITL Approval] approval_id={approval_request_id} | "
            f"skill={skill_name} | 审批挂起已持久化"
        )

    async def resume_ops_stream_pipeline(
        self,
        background_tasks: BackgroundTasks,
        request_id: str,
        user_data: dict[str, Any] | None,
        user_note: str | None,
        user_error: str | None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """从挂起状态恢复执行（多轮 HITL 数据回填）"""
        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            pending = await repo.get_by_request_id(request_id)

            if not pending:
                yield {"type": PacketType.ERROR,
                       "content": f"❌ 挂起请求 {request_id} 不存在或已过期"}
                yield {"type": PacketType.DONE,
                       "content": {"entry_id": "N/A", "status": "error"}}
                return

            if pending["status"] != "pending":
                yield {"type": PacketType.ERROR,
                       "content": f"❌ 挂起请求 {request_id} 状态为 {pending['status']}，不可恢复"}
                yield {"type": PacketType.DONE,
                       "content": {"entry_id": pending.get("entry_id", "N/A"), "status": "already_handled"}}
                return

            logger.info(
                f"[HITL Resume] request_id={request_id} | "
                f"session={pending['session_id']} | "
                f"has_data={user_data is not None} | "
                f"has_error={user_error is not None}"
            )

            # 标记为已处理
            await repo.mark_answered(request_id)

        # 重建上下文
        ctx = self._rebuild_context_from_pending(pending)

        # 恢复基础设施引用
        ctx["variables"]["_prometheus_client"] = self.prometheus_client
        ctx["variables"]["_metric_registry"] = self.metric_registry
        ctx["variables"]["_ops_db_executor"] = self.ops_db_executor

        # HITL: 追加本轮到 Timeline
        hitl_history: list[dict] = ctx.get("hitl_history", [])
        round_num = len(hitl_history) + 1

        hitl_history.append({
            "round": round_num,
            "request_id": request_id,
            "reason": pending.get("suspend_reason", ""),
            "sql_to_run": pending.get("sql_to_run", ""),
            "user_data": user_data,
            "user_error": user_error,
            "user_note": user_note,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
        ctx["hitl_history"] = hitl_history
        ctx["is_resuming"] = True

        # 从断点继续
        plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
        current_step_index = pending["current_step_index"]
        entry_id = pending["entry_id"]
        start_time = pending.get("requested_at", datetime.now(timezone.utc))

        logger.info(
            f"[HITL Resume] 从 Step {current_step_index} 恢复 | "
            f"总步骤: {len(plan_steps)} | HITL 轮次: {len(hitl_history)}"
        )

        # 继续步骤循环
        final_answer_accumulator = ""
        for idx in range(current_step_index, len(plan_steps)):
            ctx["current_step_index"] = idx
            step = plan_steps[idx]

            runtime = SkillRuntime(context=ctx)
            exec_info = runtime.create_execution_context(step_config=step)
            skill_name = exec_info["skill"]
            ctx["current_execution"] = cast(Any, exec_info)

            yield {
                "type": PacketType.CALL,
                "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}
            }

            skill_instance = self.skill_manager.get_skill_instance(skill_name)
            if not skill_instance:
                exec_info.update({"status": "failed", "error": f"组件 {skill_name} 未激活"})
                ctx["execution_history"].append(cast(Any, exec_info))
                yield {"type": PacketType.ERROR,
                       "content": f"⚠️ 关键自愈组件 [{skill_name}] 离线, 本步骤跳过。"}
                continue

            gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
            if not gate_result["allowed"]:
                if gate_result.get("needs_approval"):
                    approval_request_id = str(uuid.uuid4())
                    action_sql = ctx["variables"].get("pending_action_sql", "")
                    action_impact = ctx["variables"].get("pending_action_impact", "")
                    action_rollback = ctx["variables"].get("pending_action_rollback", "")
                    action_risk = ctx["variables"].get("pending_action_risk_level", "medium")

                    yield {
                        "type": PacketType.REQUIRE_APPROVAL,
                        "content": {
                            "request_id": approval_request_id,
                            "skill_name": skill_name,
                            "reason": gate_result["reason"],
                            "action_sql": action_sql,
                            "impact": action_impact,
                            "rollback_sql": action_rollback,
                            "risk_level": action_risk,
                            "instance_id": ctx["instance_id"],
                            "environment": ctx["environment"],
                        }
                    }

                    await self._suspend_for_approval(
                        ctx=ctx, approval_request_id=approval_request_id,
                        current_step_index=idx, entry_id=entry_id,
                        action_sql=action_sql, action_impact=action_impact,
                        action_rollback=action_rollback, skill_name=skill_name,
                    )

                    yield {
                        "type": PacketType.DONE,
                        "content": {"entry_id": entry_id, "status": "awaiting_approval",
                                    "request_id": approval_request_id}
                    }
                    return
                else:
                    exec_info.update({"status": "blocked", "error": gate_result["reason"]})
                    ctx["execution_history"].append(cast(Any, exec_info))
                    yield {"type": PacketType.ERROR,
                           "content": f"🚫 安全熔断: {gate_result['reason']}"}
                    continue

            try:
                _monitor_snapshot = len(ctx.get("monitor_results", []))
                _metric_snapshot = len(ctx.get("metric_results", []))

                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.WAIT_FOR_USER:
                        if not isinstance(content, dict):
                            continue
                        suspend_ctx = content
                        new_request_id = suspend_ctx["request_id"]
                        await self._suspend_execution(
                            ctx=ctx, suspend_ctx=suspend_ctx,
                            request_id=new_request_id, current_step_index=idx,
                            entry_id=entry_id, start_time=start_time,
                        )
                        exec_info["status"] = "suspended"
                        if final_answer_accumulator:
                            exec_info["answer"] = final_answer_accumulator.strip()
                        ctx["execution_history"].append(cast(Any, exec_info))
                        ctx["current_execution"] = None
                        yield packet
                        yield {
                            "type": PacketType.DONE,
                            "content": {"entry_id": entry_id, "status": "suspended",
                                        "request_id": new_request_id}
                        }
                        return

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")
                    if p_type == PacketType.MONITOR_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["monitor_results"].append({
                                "step_id": step.get("step_id"),
                                "task_description": step.get("task_description") or exec_info.get("resolved_input"),
                                "data": content["data"], "meta": content.get("meta", {})
                            })
                    elif p_type == PacketType.METRIC_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["metric_results"].append({
                                "step_id": step.get("step_id"),
                                "task_description": step.get("task_description") or exec_info.get("resolved_input"),
                                "data": content["data"], "meta": content.get("meta", {})
                            })

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                exec_info.update({"status": "success"})
                # 检测执行类技能是否失败（ExecuteOpsSkill 通过 ctx 传递结果）
                action_result = ctx.get("variables", {}).get("action_result", {})
                if isinstance(action_result, dict) and action_result.get("status") in ("failed", "error"):
                    exec_info["status"] = "failed"
                    exec_info["error"] = action_result.get("error", "执行失败")
                    ctx["execution_history"].append(cast(Any, exec_info))
                    ctx["current_execution"] = None
                    break  # 停止后续步骤，不再继续
                output_var = exec_info.get("output_var")
                if output_var:
                    new_monitor = ctx.get("monitor_results", [])[_monitor_snapshot:]
                    new_metric = ctx.get("metric_results", [])[_metric_snapshot:]
                    step_data = {"monitor": new_monitor, "metric": new_metric}
                    ctx["variables"][output_var] = json.dumps(step_data, ensure_ascii=False, default=str)
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None

            except Exception as e:
                logger.error(f"[Orchestrator Resume] Skill [{skill_name}] 异常: {e}")
                exec_info.update({"status": "failed", "error": str(e)})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None
                continue

        # 闭环落库 — 跳过纯成功信息，若有失败则汇总
        response_time = datetime.now(timezone.utc)
        plan_skills_trace = [s.get("skill") for s in plan_steps] if plan_steps else []
        safe_variables = {
            k: (
                json.loads(v) if isinstance(v, str) and len(v) > 0
                and v[0] in ('{', '[') else v
            )
            for k, v in ctx["variables"].items()
            if not k.startswith("_") and not hasattr(v, '__dict__')
        }

        # 汇总执行结果（失败时向用户展示明确信息）
        action_result = ctx.get("variables", {}).get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("status") == "failed":
            final_answer_accumulator = f"❌ 变更执行失败: {action_result.get('error', '未知错误')}"
            yield {
                "type": PacketType.ANSWER,
                "content": final_answer_accumulator,
            }

        model_params = await self.agent_service.get_agent_model_params(ctx["agent_id"])

        prepared_data_payload = {
            "standalone_query": ctx["command_or_query"],
            "search_keywords": ctx["command_or_query"],
            "turn_type": "task_oriented",
            "turn_entities": safe_variables,
            "new_state": safe_variables,
            "active_topic": "AIOps内核指标探测与故障自愈",
            "current_plan": {"skill_sequence": plan_skills_trace, "total_steps": len(plan_skills_trace)},
            "thought": ctx["runtime_plan"]["thought"] if ctx["runtime_plan"] else "",
            "metric_results_snapshot": ctx.get("metric_results", []),
            "doc_results_snapshot": ctx.get("doc_results", []),
        }

        background_tasks.add_task(
            self.memory_service.persist_and_reflect_memory,
            session_id=ctx["session_id"], user_id=ctx["user_id"],
            entry_id=entry_id, raw_question=ctx["command_or_query"],
            answer=final_answer_accumulator.strip() or "自动化自愈 SOP 链路安全执行完毕。",
            model_params=model_params, prepared_data=prepared_data_payload,
            context_memory=cast(Any, ctx), request_time=start_time, response_time=response_time
        )

        logger.success(f"[Orchestrator Resume] 运维自愈流水线圆满结束, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    async def resume_with_approval(
        self,
        background_tasks: BackgroundTasks,
        request_id: str,
        approved: bool,
        approver_note: str | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """审批恢复执行: 用户审批后从挂起点恢复"""
        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            pending = await repo.get_by_request_id(request_id)

            if not pending:
                yield {"type": PacketType.ERROR,
                       "content": f"❌ 审批请求 {request_id} 不存在或已过期"}
                yield {"type": PacketType.DONE,
                       "content": {"entry_id": "N/A", "status": "error"}}
                return

            if pending["status"] != "pending":
                yield {"type": PacketType.ERROR,
                       "content": f"❌ 审批请求 {request_id} 状态为 {pending['status']}，不可重复处理"}
                yield {"type": PacketType.DONE,
                       "content": {"entry_id": pending.get("entry_id", "N/A"), "status": "already_handled"}}
                return

            await repo.mark_answered(request_id)

        if not approved:
            logger.warning(
                f"[HITL Approval] 审批拒绝 | request_id={request_id} | note={approver_note}"
            )
            yield {"type": PacketType.WARNING,
                   "content": f"⚠️ 变更操作已被拒绝。\n审批人备注: {approver_note or '无'}\n诊断流程结束。\n"}
            yield {"type": PacketType.DONE,
                   "content": {"entry_id": pending.get("entry_id", ""), "status": "rejected"}}
            return

        logger.info(f"[HITL Approval] 审批通过 | request_id={request_id}")

        ctx = self._rebuild_context_from_pending(pending)
        ctx["approval_context"] = {
            "approved": True,
            "approved_by": "user",
            "approved_at": datetime.now(timezone.utc).isoformat(),
            "approver_note": approver_note or "",
        }
        ctx["variables"]["_prometheus_client"] = self.prometheus_client
        ctx["variables"]["_metric_registry"] = self.metric_registry
        ctx["variables"]["_ops_db_executor"] = self.ops_db_executor
        ctx["is_resuming"] = True

        plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
        current_step_index = pending["current_step_index"]
        entry_id = pending["entry_id"]
        start_time = pending.get("requested_at", datetime.now(timezone.utc))

        yield {"type": PacketType.THOUGHT,
               "content": "✅ 审批已通过，正在执行自愈变更操作...\n"}

        final_answer_accumulator = ""
        for idx in range(current_step_index, len(plan_steps)):
            ctx["current_step_index"] = idx
            step = plan_steps[idx]
            runtime = SkillRuntime(context=ctx)
            exec_info = runtime.create_execution_context(step_config=step)
            skill_name = exec_info["skill"]
            ctx["current_execution"] = cast(Any, exec_info)

            yield {"type": PacketType.CALL,
                   "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}}

            skill_instance = self.skill_manager.get_skill_instance(skill_name)
            if not skill_instance:
                exec_info.update({"status": "failed", "error": f"组件 {skill_name} 未激活"})
                ctx["execution_history"].append(cast(Any, exec_info))
                continue

            gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
            if not gate_result["allowed"]:
                exec_info.update({"status": "blocked", "error": gate_result["reason"]})
                ctx["execution_history"].append(cast(Any, exec_info))
                yield {"type": PacketType.ERROR,
                       "content": f"🚫 安全熔断: {gate_result['reason']}"}
                continue

            try:
                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")
                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")
                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet
                exec_info.update({"status": "success"})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None
            except Exception as e:
                logger.error(f"[Approval Resume] Skill [{skill_name}] 异常: {e}")
                exec_info.update({"status": "failed", "error": str(e)})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None
                continue

        response_time = datetime.now(timezone.utc)
        model_params = await self.agent_service.get_agent_model_params(ctx["agent_id"])

        background_tasks.add_task(
            self.memory_service.persist_and_reflect_memory,
            session_id=ctx["session_id"], user_id=ctx["user_id"],
            entry_id=entry_id, raw_question=ctx["command_or_query"],
            answer=final_answer_accumulator.strip() or "自愈变更执行完毕。",
            model_params=model_params,
            prepared_data={
                "standalone_query": ctx["command_or_query"],
                "search_keywords": ctx["command_or_query"],
                "turn_type": "task_oriented",
                "turn_entities": {}, "new_state": {},
                "active_topic": "AIOps自愈变更执行",
                "current_plan": {},
                "thought": ctx["runtime_plan"]["thought"] if ctx["runtime_plan"] else "",
                "metric_results_snapshot": ctx.get("metric_results", []),
                "doc_results_snapshot": ctx.get("doc_results", []),
            },
            context_memory=cast(Any, ctx),
            request_time=start_time, response_time=response_time
        )

        logger.success(f"[HITL Approval] 变更执行完成, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    def _rebuild_context_from_pending(
        self, pending: dict[str, Any]
    ) -> OpsContextMemory:
        """从持久化快照重建 OpsContextMemory"""
        runtime_plan = pending.get("runtime_plan") or {}
        variables = pending.get("pending_variables") or {}
        accumulated = pending.get("accumulated_results") or {}
        hitl_history = pending.get("hitl_history") or []
        completed_steps = pending.get("completed_steps") or []

        if isinstance(runtime_plan, str):
            runtime_plan = json.loads(runtime_plan)
        if isinstance(variables, str):
            variables = json.loads(variables)
        if isinstance(accumulated, str):
            accumulated = json.loads(accumulated)
        if isinstance(hitl_history, str):
            hitl_history = json.loads(hitl_history)
        if isinstance(completed_steps, str):
            completed_steps = json.loads(completed_steps)

        ctx: OpsContextMemory = {
            "trace_id": f"trace-resume-{uuid.uuid4().hex[:12]}",
            "user_id": pending.get("user_id", ""),
            "session_id": pending.get("session_id", ""),
            "agent_id": pending.get("agent_id", ""),
            "trigger_type": cast(Any, "manual"),
            "client_time": "",
            "client_tz": "",
            "command_or_query": (
                runtime_plan.get("inputs", {}).get("user_query", "")
                if isinstance(runtime_plan, dict) else ""
            ),
            "llm_model": (
                runtime_plan.get("inputs", {}).get("model_name", "")
                if isinstance(runtime_plan, dict) else ""
            ),
            "embedding_model": "",
            "instance_id": pending.get("instance_id", ""),
            "db_type": "",
            "version_code": 0,
            "db_role": "primary",
            "environment": "dev",
            "monitor_type": "prometheus",
            "prometheus_instance_label": None,
            "zabbix_host_name": None,
            "alert_context": None,
            "runtime_plan": cast(ExecutionPlan, runtime_plan),
            "current_step_index": pending.get("current_step_index", 0),
            "current_execution": None,
            "execution_history": cast(Any, completed_steps),
            "approval_context": None,
            "variables": variables,
            "metric_results": (
                accumulated.get("metric_results", [])
                if isinstance(accumulated, dict) else []
            ),
            "monitor_results": (
                accumulated.get("monitor_results", [])
                if isinstance(accumulated, dict) else []
            ),
            "os_log_snapshots": [],
            "doc_results": (
                accumulated.get("doc_results", [])
                if isinstance(accumulated, dict) else []
            ),
            "temp": {},
            "is_resuming": True,
            "hitl_history": hitl_history,
        }
        return ctx

    async def check_pending_timeouts(self) -> list[dict[str, Any]]:
        """HITL 超时检测: 扫描超时 pending 记录"""
        timed_out_requests = []
        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            pending_list = await repo.find_timeout_pending()

            for pending in pending_list:
                request_id = pending["request_id"]
                logger.warning(
                    f"[HITL Timeout] request_id={request_id} | "
                    f"session={pending['session_id']} | 已超时"
                )
                await repo.mark_timeout(request_id)
                timed_out_requests.append({
                    "request_id": request_id,
                    "session_id": pending.get("session_id", ""),
                    "user_id": pending.get("user_id", ""),
                    "instance_id": pending.get("instance_id", ""),
                    "requested_at": str(pending.get("requested_at", "")),
                })

            if timed_out_requests:
                logger.info(f"[HITL Timeout] 标记 {len(timed_out_requests)} 个超时请求")

        return timed_out_requests
