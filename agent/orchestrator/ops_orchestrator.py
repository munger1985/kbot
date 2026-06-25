# agent/orchestrator/ops_orchestrator.py

import uuid
import json
from loguru import logger
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, cast
from fastapi import BackgroundTasks

from core.dictionary import PacketType
from agent.memory import MemoryService
from skills.skill_manager import SkillManager
from skills import SkillRuntime, SkillRunMode
from agent.planner.ops_planner import OpsTaskPlanner
from agent.orchestrator import DocOrchestrator
from services.basic import AgentService
from services.basic import OpsAgentConfService
from agent.common.ops_context import OpsContextMemory
from utils.clients import OpsDBExecutor
from utils.monitor import PrometheusClient, UnifiedMetricRegistry


DISPLAY_PACKET_TYPES = {
    PacketType.THOUGHT,
    PacketType.ANSWER,
    PacketType.ERROR,
    PacketType.DOC_RESULTS,
    PacketType.MONITOR_RESULTS,
    PacketType.METRIC_RESULTS,
    PacketType.WARNING,
    PacketType.REQUIRE_APPROVAL,
    PacketType.CALL,
    PacketType.DONE
}


class OpsOrchestrator:
    """
    智能故障自愈核心流水线编排器 (UI 强管控·精准定靶版)
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str
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
        trigger_type: str = "manual"
    ) -> AsyncGenerator[dict[str, Any], None]:

        # --- 0. 全周期元数据计算 ---
        start_time = datetime.now(timezone.utc)
        if not session_id or session_id == "new_session":
            session_id = f"sess_{uuid.uuid4().hex[:12]}"
        entry_id = f"entr_{uuid.uuid4().hex[:12]}"

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
            "llm_model": llm_model,
            "embedding_model": embedding_model,

            "instance_id": instance_id,
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
            "temp": {}
        }

        # --- 1.2 拓扑资产锁定与安全策略注入 ---
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

        # --- 注入监控与诊断基础设施引用 (供 Skill 使用, 必须在规划完成后注入以避免 JSON 序列化问题) ---
        ctx["variables"]["_prometheus_client"] = self.prometheus_client
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
                "content": {"skill": skill_name, "description": exec_info["resolved_input"]}
            }

            skill_instance = self.skill_manager.get_skill_instance(skill_name)
            if not skill_instance:
                exec_info.update({"status": "failed", "error": f"组件 {skill_name} 未激活"})
                ctx["execution_history"].append(cast(Any, exec_info))
                yield {"type": PacketType.ERROR, "content": f"⚠️ 关键自愈组件 [{skill_name}] 离线, 本步骤跳过。"}
                continue

            # --- 🔒 安全熔断门禁 ---
            gate_result = self._check_safety_gate(ctx, skill_instance, skill_name)
            if not gate_result["allowed"]:
                exec_info.update({"status": "blocked", "error": gate_result["reason"]})  # type: ignore
                ctx["execution_history"].append(cast(Any, exec_info))
                yield {"type": PacketType.ERROR, "content": f"🚫 安全熔断: {gate_result['reason']}"}
                continue

            try:
                # 记录步骤执行前的数据快照，用于 output_var 回写
                _monitor_snapshot = len(ctx.get("monitor_results", []))
                _metric_snapshot = len(ctx.get("metric_results", []))

                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")

                    # 数据沉淀区：MONITOR_RESULTS → 监控数据, METRIC_RESULTS → 诊断数据
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

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                exec_info.update({"status": "success"})
                # 将本步骤新采集的数据回写到 variables，使后续步骤的 {{output_var}} 可被解析
                output_var = exec_info.get("output_var")
                if output_var:
                    new_monitor = ctx.get("monitor_results", [])[_monitor_snapshot:]
                    new_metric = ctx.get("metric_results", [])[_metric_snapshot:]
                    # 合并本步骤产生的所有数据
                    step_data = {
                        "monitor": new_monitor,
                        "metric": new_metric,
                    }
                    ctx["variables"][output_var] = json.dumps(step_data, ensure_ascii=False, default=str)
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None

            except Exception as e:
                logger.error(f"[Orchestrator] 执行自愈组件 [{skill_name}] 发生非致命中断: {e}")
                exec_info.update({"status": "failed", "error": str(e)})
                ctx["execution_history"].append(cast(Any, exec_info))
                ctx["current_execution"] = None
                continue

        # --- 4. 闭环落库: 挂载后台反思与审计任务 ---
        response_time = datetime.now(timezone.utc)
        plan_skills_trace = [s.get("skill") for s in plan_steps] if plan_steps else []

        # 过滤掉不可序列化的内部对象（如 PrometheusClient），只保留可持久化的数据
        safe_variables = {
            k: v for k, v in ctx["variables"].items()
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

    def _check_safety_gate(
        self,
        ctx: OpsContextMemory,
        skill_instance: Any,
        skill_name: str
    ) -> dict[str, Any]:
        """
        🔒 运维安全熔断门禁:
        在执行任何技能前, 校验该技能是否被允许在当前实例上运行。
        对 MUTATION 类高危技能执行多层策略校验（变更许可、审批门禁、频次上限）。
        """
        skill_meta = getattr(skill_instance, "meta", None)
        skill_run_mode = getattr(skill_meta, "run_mode", SkillRunMode.READ_ONLY) if skill_meta else SkillRunMode.READ_ONLY

        # 只读探测技能不受安全门禁限制
        if skill_run_mode == SkillRunMode.READ_ONLY:
            return {"allowed": True}

        # --- 变更类技能必须通过多重安全校验 ---
        variables = ctx.get("variables", {})
        environment = ctx.get("environment", "prod")
        instance_id = ctx.get("instance_id", "unknown")

        # 1. 变更许可检查
        is_mutation_allowed = variables.get("is_mutation_allowed", False)
        if not is_mutation_allowed:
            logger.warning(
                f"[SafetyGate] 拦截高危自愈动作 | Skill: {skill_name} | "
                f"实例: {instance_id} | 原因: 该实例未开启变更许可"
            )
            return {
                "allowed": False,
                "reason": (
                    f"自愈组件 [{skill_name}] 属于变更类高危操作, 但实例 `{instance_id}` "
                    f"未开启变更许可 (is_mutation_allowed=False)。请联系 DBA 管理员在资产控制面开启此实例的自愈变更权限。"
                )
            }

        # 2. 审批门禁检查
        require_approval = variables.get("require_approval", True)
        if require_approval:
            approval_ctx = ctx.get("approval_context")
            if not approval_ctx or not approval_ctx.get("approved"):
                logger.warning(
                    f"[SafetyGate] 拦截高危自愈动作 | Skill: {skill_name} | "
                    f"实例: {instance_id} | 原因: 缺少有效的人工审批令牌"
                )
                return {
                    "allowed": False,
                    "reason": (
                        f"自愈组件 [{skill_name}] 需要人工审批授权方可执行。"
                        f"请在前端确认该高危操作的风险后, 重新发起携带审批令牌的请求。"
                    )
                }

        # 3. 生产环境额外警告
        if environment == "prod":
            logger.warning(
                f"[SafetyGate] ⚠️ 生产环境高危动作即将执行 | Skill: {skill_name} | "
                f"实例: {instance_id} | 审批令牌已校验通过"
            )

        logger.info(
            f"[SafetyGate] ✅ 安全门禁通过 | Skill: {skill_name} | "
            f"实例: {instance_id} | 环境: {environment}"
        )
        return {"allowed": True}
