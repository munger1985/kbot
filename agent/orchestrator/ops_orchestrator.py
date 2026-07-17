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
from agent.common.skill_context import ExecutionPlan, SkillExecutionContext, TaskStep
from agent.common.ops_verifier import OpsVerifier
from agent.common.ops_reporter import OpsReporter
from utils.clients import OpsDBExecutor
from utils.monitor import PrometheusClient, ZabbixProvider, UnifiedMetricRegistry
from dao.repositories import PendingRequestRepository, OpsExecutionReportRepository


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
    PacketType.VERIFICATION_RESULTS,
    PacketType.CONFIRM_ACTION,
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

    # ==================================================================
    # 主入口: execute_ops_stream_pipeline
    # ==================================================================

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

        model_params = await self.agent_service.get_agent_model_params(agent_id)
        llm_model = model_params.llm_model
        embedding_model = model_params.txt_embedding_model

        await self.memory_service.ensure_session_exists(
            session_id=session_id, user_id=user_id, agent_id=agent_id, question=question
        )

        # --- 1. 严格按照 OpsContextMemory 定义实例化强类型总线 ---
        ctx = self._build_initial_context(
            user_id=user_id, session_id=session_id, agent_id=agent_id,
            question=question, instance_id=instance_id, trigger_type=trigger_type,
            client_time=client_time, client_tz=client_tz,
            llm_model=llm_model, embedding_model=embedding_model,
        )

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
                ctx["instance_name"] = target_db["instance_name"]

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

        # 采集修复前的指标快照
        self._collect_pre_snapshot(ctx)

        # --- 3. 驱动强类型状态机执行线性原子技能 ---
        plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
        final_answer_accumulator = ""

        for idx, step in enumerate(plan_steps):
            ctx["current_step_index"] = idx

            result = self._prepare_step(ctx, step)
            if result is None:
                exec_info = ctx.get("current_execution") or {}
                status = exec_info.get("status", "failed")
                if status == "failed":
                    yield {"type": PacketType.ERROR, "content": f"⚠️ 关键自愈组件 [{exec_info.get('skill', 'unknown')}] 离线, 本步骤跳过。"}
                else:
                    yield {"type": PacketType.ERROR, "content": f"🚫 安全熔断: {exec_info.get('error', '未知原因')}"}
                ctx["current_execution"] = None
                continue

            runtime, exec_info, skill_instance, skill_name = result

            yield {
                "type": PacketType.CALL,
                "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}
            }

            try:
                _monitor_snapshot = len(ctx.get("monitor_results", []))
                _metric_snapshot = len(ctx.get("metric_results", []))

                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")
                    if content is None:
                        continue

                    # ──── CONFIRM_ACTION: 逐命令确认截停 ────
                    if p_type == PacketType.CONFIRM_ACTION:
                        confirm_request_id = str(uuid.uuid4())

                        yield {
                            "type": PacketType.CONFIRM_ACTION,
                            "content": {
                                "request_id": confirm_request_id,
                                **(content if isinstance(content, dict) else {}),
                            }
                        }

                        await self._persist_pending(
                            ctx,
                            request_id=confirm_request_id,
                            current_step_index=idx,
                            entry_id=entry_id,
                            suspend_reason=f"等待确认: {skill_name} (第{content.get('round', 1)}轮)",
                            suspend_type="confirm_action",
                            user_prompt=(
                                f"## 待确认的变更操作\n\n"
                                f"**SQL**:\n```sql\n{content.get('sql', '')}\n```\n\n"
                                f"**影响**: {content.get('impact', '')}\n\n**回滚**: {content.get('rollback_sql', '')}\n"
                            ),
                            sql_to_run=content.get("sql", ""),
                            expected_fields={
                                "type": "confirm_action",
                                "action_sql": content.get("sql", ""),
                                "impact": content.get("impact", ""),
                                "rollback_sql": content.get("rollback_sql", ""),
                            },
                            suspended_by_skill=skill_name,
                        )

                        yield {"type": PacketType.DONE,
                               "content": {"entry_id": entry_id,
                                           "status": "awaiting_confirm",
                                           "request_id": confirm_request_id}}
                        return
                    # ──── CONFIRM_ACTION 结束 ────

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")

                    # 数据沉淀区
                    self._accumulate_data_result(ctx, cast(PacketType, p_type), content, step, exec_info)

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

                        await self._persist_pending(
                            ctx,
                            request_id=request_id,
                            current_step_index=idx,
                            entry_id=entry_id,
                            suspend_reason=suspend_ctx.get("reason", ""),
                            user_prompt=suspend_ctx.get("sql_to_run", ""),
                            sql_to_run=suspend_ctx.get("sql_to_run", ""),
                            expected_fields=suspend_ctx.get("expected_fields", []),
                            suspended_by_skill=suspend_ctx.get("suspended_by", "unknown"),
                        )

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

                # 步骤后处理：状态写入、output_var、action_result 检查
                should_continue = self._finalize_step(ctx, exec_info, _monitor_snapshot, _metric_snapshot)
                if not should_continue:
                    break

            except Exception as e:
                self._handle_step_error(ctx, exec_info, skill_name, e)
                continue

        # --- 4. 验证阶段 (Verify) ---
        llm_model = model_params.llm_model if model_params else ""
        executed_actions = self._build_executed_actions(ctx)

        verify_packets, verify_result, rollback_info = await self._verify_and_maybe_rollback(ctx)
        for p in verify_packets:
            yield p

        # --- 4b. 生成执行报告 ---
        report_packets, report_md = await self._generate_and_persist_report(
            ctx, entry_id=entry_id, session_id=session_id, user_id=user_id,
            agent_id=agent_id, start_time=start_time, original_question=question,
            executed_actions=executed_actions, verify_result=verify_result,
            rollback_info=rollback_info, llm_model=llm_model,
        )
        for p in report_packets:
            yield p

        if report_md:
            final_answer_accumulator = report_md

        # --- 5. 闭环落库 ---
        action_result = ctx.get("variables", {}).get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("status") == "failed":
            if "变更执行失败" not in final_answer_accumulator:
                final_answer_accumulator = f"❌ 变更执行失败: {action_result.get('error', '未知错误')}"
                yield {
                    "type": PacketType.ANSWER,
                    "content": final_answer_accumulator,
                }

        self._schedule_memory_persistence(
            ctx, background_tasks, entry_id, final_answer_accumulator,
            start_time, plan_steps, model_params, question,
        )

        logger.success(f"[Orchestrator] 运维自愈强类型流水线圆满结束, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ==================================================================
    # 上下文构建
    # ==================================================================

    @staticmethod
    def _build_initial_context(
        user_id: str,
        session_id: str,
        agent_id: int,
        question: str,
        instance_id: str,
        trigger_type: str,
        client_time: str | None,
        client_tz: str | None,
        llm_model: str,
        embedding_model: str,
    ) -> OpsContextMemory:
        """构建初始 OpsContextMemory。"""
        return {
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
            "instance_name": "",
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

            "is_resuming": False,
            "hitl_history": [],
        }

    # ==================================================================
    # 安全门禁 v2
    # ==================================================================

    @staticmethod
    def _extract_first_metric_value(data: list[dict]) -> float | None:
        """从 monitor/metric_results 的 data 列表中提取第一个数值。"""
        if not data:
            return None
        item = data[0]
        if isinstance(item, dict):
            v = item.get("value")
            if isinstance(v, list) and len(v) >= 2:
                return float(v[1])
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v)
                except ValueError:
                    return None
        return None

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

        # 2. 逐命令批准 — 由 ops-heal-skill 通过 CONFIRM_ACTION 控制，不再此处全局拦截

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
    # 静态工具方法
    # ==================================================================

    @staticmethod
    def _collect_pre_snapshot(ctx: OpsContextMemory) -> dict[str, dict]:
        """从 monitor_results 和 metric_results 中提取修复前指标快照，写入 ctx。"""
        pre_snapshot: dict[str, dict] = {}
        for items in (ctx.get("monitor_results", []), ctx.get("metric_results", [])):
            for item in items:
                meta = item.get("meta", {})
                promql = meta.get("promql", "")
                metric_name = meta.get("metric_code") or item.get("task_description", "")
                data = item.get("data", [])
                if promql and metric_name and data:
                    val = OpsOrchestrator._extract_first_metric_value(data)
                    pre_snapshot[metric_name] = {"value": val, "promql": promql}
        if pre_snapshot:
            ctx["variables"]["_pre_snapshot"] = pre_snapshot
        return pre_snapshot

    @staticmethod
    def _build_executed_actions(ctx: OpsContextMemory) -> list[dict]:
        """从 ctx.variables 构建已执行动作列表。"""
        var_center = ctx.get("variables", {})
        pending_sql = var_center.get("pending_action_sql", "")
        if not pending_sql:
            return []
        return [{
            "sql": pending_sql,
            "impact": var_center.get("pending_action_impact", ""),
            "risk_level": var_center.get("pending_action_risk_level", "medium"),
            "context": var_center.get("pending_action_context", ""),
        }]

    @staticmethod
    def _accumulate_data_result(
        ctx: OpsContextMemory,
        p_type: PacketType,
        content: Any,
        step: TaskStep,
        exec_info: SkillExecutionContext,
    ) -> None:
        """将 MONITOR_RESULTS / METRIC_RESULTS 数据沉淀到 ctx。"""
        if not isinstance(content, dict) or "data" not in content:
            return
        entry = {
            "step_id": step.get("step_id"),
            "task_description": step.get("task_description") or exec_info.get("resolved_input"),
            "data": content["data"],
            "meta": content.get("meta", {}),
        }
        if p_type == PacketType.MONITOR_RESULTS:
            ctx["monitor_results"].append(entry)
        elif p_type == PacketType.METRIC_RESULTS:
            ctx["metric_results"].append(entry)

    # ==================================================================
    # Step 执行辅助方法
    # ==================================================================

    def _prepare_step(
        self, ctx: OpsContextMemory, step: TaskStep
    ) -> tuple[Any, SkillExecutionContext, Any, str] | None:
        """
        步骤执行前准备：创建执行上下文、获取 skill 实例、通过安全门禁。
        返回 (runtime, exec_info, skill_instance, skill_name)，失败时返回 None
        （此时 exec_info 已写入错误状态并追加到 execution_history）。
        """
        runtime = SkillRuntime(context=ctx)
        exec_info = runtime.create_execution_context(step_config=step)
        skill_name = exec_info["skill"]
        ctx["current_execution"] = cast(Any, exec_info)

        skill_instance = self.skill_manager.get_skill_instance(skill_name)
        if not skill_instance:
            exec_info.update({"status": "failed", "error": f"组件 {skill_name} 未激活"})
            ctx["execution_history"].append(cast(Any, exec_info))
            return None

        gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
        if not gate_result["allowed"]:
            exec_info.update({"status": "blocked", "error": gate_result["reason"]})
            ctx["execution_history"].append(cast(Any, exec_info))
            return None

        return runtime, exec_info, skill_instance, skill_name

    def _finalize_step(
        self, ctx: OpsContextMemory, exec_info: SkillExecutionContext,
        monitor_snapshot: int, metric_snapshot: int,
    ) -> bool:
        """
        步骤成功后的清理工作。返回 True 继续下一步，False 停止流水线。
        """
        exec_info.update({"status": "success"})

        # 检测执行类技能是否失败
        action_result = ctx.get("variables", {}).get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("status") in ("failed", "error"):
            exec_info["status"] = "failed"
            exec_info["error"] = action_result.get("error", "执行失败")
            ctx["execution_history"].append(cast(Any, exec_info))
            ctx["current_execution"] = None
            return False

        output_var = exec_info.get("output_var")
        if output_var:
            new_monitor = ctx.get("monitor_results", [])[monitor_snapshot:]
            new_metric = ctx.get("metric_results", [])[metric_snapshot:]
            step_data = {"monitor": new_monitor, "metric": new_metric}
            ctx["variables"][output_var] = json.dumps(step_data, ensure_ascii=False, default=str)

        ctx["execution_history"].append(cast(Any, exec_info))
        ctx["current_execution"] = None
        return True

    @staticmethod
    def _handle_step_error(ctx: OpsContextMemory, exec_info: SkillExecutionContext, skill_name: str, error: Exception) -> None:
        """统一的步骤异常处理。"""
        logger.error(f"[Orchestrator] 执行自愈组件 [{skill_name}] 发生非致命中断: {error}")
        exec_info.update({"status": "failed", "error": str(error)})
        ctx["execution_history"].append(cast(Any, exec_info))
        ctx["current_execution"] = None

    # ==================================================================
    # 统一挂起持久化 (替代 _suspend_execution / _suspend_confirm_action / _suspend_for_approval)
    # ==================================================================

    async def _persist_pending(
        self,
        ctx: OpsContextMemory,
        *,
        request_id: str,
        current_step_index: int,
        entry_id: str,
        suspend_reason: str,
        user_prompt: str,
        sql_to_run: str,
        expected_fields: Any,
        suspended_by_skill: str,
        suspend_type: str | None = None,
        timeout_minutes: int = 30,
    ) -> None:
        """统一持久化挂起请求到 kbot_ops_pending_request。"""
        timeout_at = datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)

        pending_data: dict[str, Any] = {
            "request_id": request_id,
            "session_id": ctx["session_id"],
            "user_id": ctx["user_id"],
            "agent_id": ctx["agent_id"],
            "instance_id": ctx["instance_id"],
            "entry_id": entry_id,
            "suspend_reason": suspend_reason,
            "user_prompt": user_prompt,
            "sql_to_run": sql_to_run,
            "expected_fields": json.dumps(expected_fields, ensure_ascii=False),
            "suspended_by_skill": suspended_by_skill,
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
        if suspend_type:
            pending_data["suspend_type"] = suspend_type

        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            await repo.create(pending_data)

        logger.info(
            f"[PersistPending] request_id={request_id} | "
            f"type={suspend_type or 'hitl'} | 快照已持久化"
        )

    # ==================================================================
    # 验证 + 回滚 (公共逻辑)
    # ==================================================================

    async def _verify_and_maybe_rollback(
        self, ctx: OpsContextMemory, *,
        enable_rollback: bool = True,
    ) -> tuple[list[dict], Any, dict | None]:
        """
        执行验证阶段，必要时触发自动回滚。
        返回: (packets_to_yield, verify_result, rollback_info)
        """
        packets: list[dict] = []
        var_center = ctx.get("variables", {})
        pending_sql = var_center.get("pending_action_sql", "")
        pending_rollback = var_center.get("pending_action_rollback", "")
        pre_snapshot = var_center.get("_pre_snapshot", {})

        verify_result = None
        rollback_info = None

        if not pending_sql or not pre_snapshot:
            return packets, verify_result, rollback_info

        packets.append({"type": PacketType.THOUGHT, "content": "🔍 开始验证自愈效果..."})

        try:
            verifier = OpsVerifier()
            verify_result = await verifier.verify(
                instance_id=ctx["instance_id"],
                db_type=ctx.get("db_type", ""),
                monitor_type=var_center.get("monitor_type", "prometheus"),
                pre_snapshot=pre_snapshot,
                executed_sql=pending_sql,
                rollback_sql=pending_rollback,
            )

            packets.append({
                "type": PacketType.VERIFICATION_RESULTS,
                "content": {
                    "status": verify_result.status.value,
                    "pre_snapshot": verify_result.pre_snapshot,
                    "post_snapshot": verify_result.post_snapshot,
                    "health_check": verify_result.health_check_result,
                    "summary": verify_result.summary,
                },
            })

            if verify_result.status.value == "failed" and enable_rollback:
                if pending_rollback:
                    packets.append({"type": PacketType.WARNING,
                                    "content": "❌ 验证失败，执行自动回滚..."})
                    try:
                        rollback_result = await self.ops_db_executor.execute_rollback_ops_sql(
                            instance_id=ctx["instance_id"],
                            db_type=ctx.get("db_type", ""),
                            rollback_sql=pending_rollback,
                            reason=f"自愈验证失败: {verify_result.summary}",
                        )
                        rollback_info = {
                            "rollback_sql": pending_rollback,
                            "executed": True,
                            "result": str(rollback_result)[:500],
                        }
                    except Exception as rollback_err:
                        rollback_info = {
                            "rollback_sql": pending_rollback,
                            "executed": True,
                            "result": f"回滚执行异常: {rollback_err}",
                        }
            elif verify_result.status.value == "degraded":
                packets.append({"type": PacketType.WARNING,
                                "content": "⚠️ 部分指标已恢复但未达预期，建议人工检查"})
        except Exception as verify_err:
            logger.error(f"[Orchestrator] 验证阶段异常: {verify_err}")
            packets.append({"type": PacketType.WARNING,
                            "content": f"⚠️ 验证过程异常: {verify_err}"})

        return packets, verify_result, rollback_info

    # ==================================================================
    # 报告生成 + 持久化 (公共逻辑)
    # ==================================================================

    async def _generate_and_persist_report(
        self, ctx: OpsContextMemory, *,
        entry_id: str, session_id: str, user_id: str, agent_id: int,
        start_time: datetime, original_question: str,
        executed_actions: list[dict], verify_result: Any, rollback_info: dict | None,
        llm_model: str,
    ) -> tuple[list[dict], str]:
        """
        生成执行报告并持久化到数据库。
        返回: (packets_to_yield, report_md)
        """
        packets: list[dict] = []

        try:
            reporter = OpsReporter()
            report_md = await reporter.generate_report(
                instance_name=ctx.get("instance_name", ""),
                db_type=ctx.get("db_type", ""),
                environment=ctx.get("environment", "prod"),
                trigger_type=ctx.get("trigger_type", "manual"),
                original_question=original_question,
                diagnosis_summary=ctx.get("diagnosis_summary", ""),
                executed_actions=executed_actions,
                verify_result=verify_result,
                rollback_info=rollback_info,
                total_duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
                llm_model=llm_model,
            )

            # 持久化报告
            async with db_instance().get_session() as report_session:
                report_repo = OpsExecutionReportRepository(report_session)
                await report_repo.create({
                    "entry_id": entry_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "instance_id": ctx["instance_id"],
                    "instance_name": ctx.get("instance_name", ""),
                    "db_type": ctx.get("db_type", ""),
                    "environment": ctx.get("environment", "prod"),
                    "trigger_type": ctx.get("trigger_type", "manual"),
                    "original_question": original_question,
                    "diagnosis_summary": ctx.get("diagnosis_summary", ""),
                    "actions_executed": executed_actions,
                    "pre_snapshot": verify_result.pre_snapshot if verify_result else None,
                    "post_snapshot": verify_result.post_snapshot if verify_result else None,
                    "verification_status": verify_result.status.value if verify_result else "skipped",
                    "health_check_result": verify_result.health_check_result if verify_result else None,
                    "rollback_info": rollback_info,
                    "report_content": report_md,
                    "recommendations": "",
                    "total_duration_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
                })
                logger.info(f"[Orchestrator] 执行报告已持久化: entry={entry_id}")

            # SSE 推送报告
            packets.append({"type": PacketType.ANSWER, "content": report_md})
            return packets, report_md

        except Exception as report_err:
            logger.error(f"[Orchestrator] 报告生成/持久化失败: {report_err}")
            return packets, ""

    # ==================================================================
    # Memory 持久化调度 (公共逻辑)
    # ==================================================================

    def _schedule_memory_persistence(
        self, ctx: OpsContextMemory, background_tasks: BackgroundTasks,
        entry_id: str, final_answer: str, start_time: datetime,
        plan_steps: list[TaskStep], model_params: Any, raw_question: str,
    ) -> None:
        """构建 prepared_data 并调度后台 memory 持久化任务。"""
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
            "standalone_query": raw_question,
            "search_keywords": raw_question,
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
            session_id=ctx["session_id"],
            user_id=ctx["user_id"],
            entry_id=entry_id,
            raw_question=raw_question,
            answer=final_answer.strip() or "自动化自愈 SOP 链路安全执行完毕。",
            model_params=model_params,
            prepared_data=prepared_data_payload,
            context_memory=cast(Any, ctx),
            request_time=start_time,
            response_time=response_time,
        )

    # ==================================================================
    # HITL: 恢复执行
    # ==================================================================

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

                    await self._persist_pending(
                        ctx,
                        request_id=approval_request_id,
                        current_step_index=idx,
                        entry_id=entry_id,
                        suspend_reason=f"等待审批: {skill_name}",
                        suspend_type="approval",
                        user_prompt=(
                            f"## 待审批的变更操作\n\n"
                            f"**SQL**:\n```sql\n{action_sql}\n```\n\n"
                            f"**影响**: {action_impact}\n\n"
                            f"**回滚**: {action_rollback}\n"
                        ),
                        sql_to_run=action_sql,
                        expected_fields={
                            "type": "approval",
                            "action_sql": action_sql,
                            "impact": action_impact,
                            "rollback_sql": action_rollback,
                        },
                        suspended_by_skill=skill_name,
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
                        await self._persist_pending(
                            ctx,
                            request_id=new_request_id,
                            current_step_index=idx,
                            entry_id=entry_id,
                            suspend_reason=suspend_ctx.get("reason", ""),
                            user_prompt=suspend_ctx.get("sql_to_run", ""),
                            sql_to_run=suspend_ctx.get("sql_to_run", ""),
                            expected_fields=suspend_ctx.get("expected_fields", []),
                            suspended_by_skill=suspend_ctx.get("suspended_by", "unknown"),
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

                    self._accumulate_data_result(ctx, cast(PacketType, p_type), content, plan_steps[idx], exec_info)

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                # 步骤后处理
                should_continue = self._finalize_step(ctx, exec_info, _monitor_snapshot, _metric_snapshot)
                if not should_continue:
                    break

            except Exception as e:
                self._handle_step_error(ctx, exec_info, skill_name, e)
                continue

        # 闭环落库
        model_params = await self.agent_service.get_agent_model_params(ctx["agent_id"])

        # 汇总执行结果（失败时向用户展示明确信息）
        action_result = ctx.get("variables", {}).get("action_result", {})
        if isinstance(action_result, dict) and action_result.get("status") == "failed":
            final_answer_accumulator = f"❌ 变更执行失败: {action_result.get('error', '未知错误')}"
            yield {"type": PacketType.ANSWER, "content": final_answer_accumulator}

        self._schedule_memory_persistence(
            ctx, background_tasks, entry_id, final_answer_accumulator,
            start_time, plan_steps, model_params, ctx["command_or_query"],
        )

        logger.success(f"[Orchestrator Resume] 运维自愈流水线圆满结束, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ==================================================================
    # 逐命令确认恢复
    # ==================================================================

    async def resume_confirm_action(
        self,
        background_tasks: BackgroundTasks,
        request_id: str,
        confirmed: bool,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """逐命令确认恢复 — 用户确认/取消后继续执行"""
        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            pending = await repo.get_by_request_id(request_id)

        if not pending or pending.get("status") != "pending":
            yield {"type": PacketType.ERROR, "content": "❌ 确认请求不存在或已处理"}
            yield {"type": PacketType.DONE, "content": {"entry_id": "N/A", "status": "error"}}
            return

        async with db_instance().get_session() as session:
            repo = PendingRequestRepository(session)
            await repo.mark_answered(request_id)

        logger.info(f"[ConfirmResume] request_id={request_id} | confirmed={confirmed}")

        # 重建上下文
        ctx = self._rebuild_context_from_pending(pending)
        ctx["variables"]["_prometheus_client"] = self.prometheus_client
        ctx["variables"]["_zabbix_client"] = self.zabbix_client
        ctx["variables"]["_metric_registry"] = self.metric_registry
        ctx["variables"]["_ops_db_executor"] = self.ops_db_executor

        ctx["variables"]["_action_confirmed"] = confirmed
        ctx["is_resuming"] = True

        # 采集修复前快照
        self._collect_pre_snapshot(ctx)

        plan_steps = ctx["runtime_plan"]["steps"] if ctx["runtime_plan"] else []
        current_step_index = pending.get("current_step_index", 0)
        entry_id = pending.get("entry_id", "")
        start_time = pending.get("requested_at") or datetime.now(timezone.utc)
        final_answer_accumulator = ""

        for idx in range(current_step_index, len(plan_steps)):
            ctx["current_step_index"] = idx

            result = self._prepare_step(ctx, plan_steps[idx])
            if result is None:
                exec_info = ctx.get("current_execution") or {}
                if exec_info.get("status") == "failed":
                    yield {"type": PacketType.ERROR,
                           "content": f"⚠️ 关键自愈组件 [{exec_info.get('skill', 'unknown')}] 离线, 本步骤跳过。"}
                else:
                    yield {"type": PacketType.ERROR,
                           "content": f"🚫 安全熔断: {exec_info.get('error', '未知原因')}"}
                ctx["current_execution"] = None
                continue

            runtime, exec_info, skill_instance, skill_name = result

            yield {"type": PacketType.CALL,
                   "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}}

            try:
                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")
                    if content is None:
                        continue

                    if p_type == PacketType.CONFIRM_ACTION:
                        new_request_id = str(uuid.uuid4())
                        yield {"type": PacketType.CONFIRM_ACTION,
                               "content": {"request_id": new_request_id, **(content if isinstance(content, dict) else {})}}
                        await self._persist_pending(
                            ctx,
                            request_id=new_request_id,
                            current_step_index=idx,
                            entry_id=entry_id,
                            suspend_reason=f"等待确认: {skill_name} (第{content.get('round', 1)}轮)",
                            suspend_type="confirm_action",
                            user_prompt=(
                                f"## 待确认的变更操作\n\n"
                                f"**SQL**:\n```sql\n{content.get('sql', '')}\n```\n\n"
                                f"**影响**: {content.get('impact', '')}\n\n**回滚**: {content.get('rollback_sql', '')}\n"
                            ),
                            sql_to_run=content.get("sql", ""),
                            expected_fields={
                                "type": "confirm_action",
                                "action_sql": content.get("sql", ""),
                                "impact": content.get("impact", ""),
                                "rollback_sql": content.get("rollback_sql", ""),
                            },
                            suspended_by_skill=skill_name,
                        )
                        yield {"type": PacketType.DONE,
                               "content": {"entry_id": entry_id, "status": "awaiting_confirm", "request_id": new_request_id}}
                        return

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                exec_info.update({"status": "success"})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None

            except Exception as e:
                self._handle_step_error(ctx, exec_info, skill_name, e)
                continue

        # 清理确认状态
        ctx["variables"].pop("_action_confirmed", None)

        # 验证 (不含回滚) + 报告 + 落库
        executed_actions = self._build_executed_actions(ctx)
        verify_packets, verify_result, _ = await self._verify_and_maybe_rollback(
            ctx, enable_rollback=False,
        )
        for p in verify_packets:
            yield p

        model_params = await self.agent_service.get_agent_model_params(ctx["agent_id"])
        llm_model = model_params.llm_model if model_params else ""

        report_packets, report_md = await self._generate_and_persist_report(
            ctx, entry_id=entry_id, session_id=ctx["session_id"],
            user_id=ctx["user_id"], agent_id=ctx["agent_id"],
            start_time=start_time, original_question=ctx["command_or_query"],
            executed_actions=executed_actions, verify_result=verify_result,
            rollback_info=None, llm_model=llm_model,
        )
        for p in report_packets:
            yield p

        if report_md:
            final_answer_accumulator = report_md

        self._schedule_memory_persistence(
            ctx, background_tasks, entry_id, final_answer_accumulator,
            start_time, plan_steps, model_params, ctx["command_or_query"],
        )

        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ==================================================================
    # 审批恢复执行
    # ==================================================================

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

        # 采集修复前的指标快照
        self._collect_pre_snapshot(ctx)

        final_answer_accumulator = ""
        for idx in range(current_step_index, len(plan_steps)):
            ctx["current_step_index"] = idx

            result = self._prepare_step(ctx, plan_steps[idx])
            if result is None:
                exec_info = ctx.get("current_execution") or {}
                if exec_info.get("status") == "failed":
                    yield {"type": PacketType.ERROR,
                           "content": f"⚠️ 关键自愈组件 [{exec_info.get('skill', 'unknown')}] 离线, 本步骤跳过。"}
                else:
                    yield {"type": PacketType.ERROR,
                           "content": f"🚫 安全熔断: {exec_info.get('error', '未知原因')}"}
                ctx["current_execution"] = None
                continue

            runtime, exec_info, skill_instance, skill_name = result

            yield {"type": PacketType.CALL,
                   "content": {"skill": skill_name, "description": (exec_info["resolved_input"] or "")[:120]}}

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
                self._handle_step_error(ctx, exec_info, skill_name, e)
                continue

        # 验证 + 报告 + 落库
        executed_actions = self._build_executed_actions(ctx)

        verify_packets, verify_result, rollback_info = await self._verify_and_maybe_rollback(ctx)
        for p in verify_packets:
            yield p

        model_params = await self.agent_service.get_agent_model_params(ctx["agent_id"])
        llm_model = model_params.llm_model if model_params else ""

        report_packets, report_md = await self._generate_and_persist_report(
            ctx, entry_id=entry_id, session_id=ctx["session_id"],
            user_id=ctx["user_id"], agent_id=ctx["agent_id"],
            start_time=start_time, original_question=ctx["command_or_query"],
            executed_actions=executed_actions, verify_result=verify_result,
            rollback_info=rollback_info, llm_model=llm_model,
        )
        for p in report_packets:
            yield p

        if report_md:
            final_answer_accumulator = report_md

        self._schedule_memory_persistence(
            ctx, background_tasks, entry_id, final_answer_accumulator,
            start_time, plan_steps, model_params, ctx["command_or_query"],
        )

        logger.success(f"[HITL Approval] 变更执行完成, Entry ID: {entry_id}")
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ==================================================================
    # 上下文重建
    # ==================================================================

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
            "instance_name": pending.get("instance_name", ""),
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

    # ==================================================================
    # HITL 超时检测
    # ==================================================================

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
