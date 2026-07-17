import json
import uuid
import asyncio
from loguru import logger
from datetime import datetime, timezone
from typing import Any, AsyncGenerator
from fastapi import BackgroundTasks

# 核心类型与组件
from agent.memory import MemoryService
from skills.skill_manager import SkillManager
from services.basic.agent_service import AgentService
from .intent_router import IntentRouter
from agent.planner.decision_engine import PlanningEngine
from agent.planner.execution_scheduler import compute_execution_waves
from skills import SkillRuntime
from agent.common import ContextMemory
from agent.common.skill_context import TaskStep
from core.dictionary import IntentType, PacketType
from utils.lang_detect import detect_user_language


# 定义需要实时分发给前端的流包类型
DISPLAY_PACKET_TYPES = {
    PacketType.THOUGHT, 
    PacketType.ANSWER, 
    PacketType.ECHARTS,
    PacketType.CALL,
    PacketType.ERROR,
    PacketType.DOC_RESULTS,
    PacketType.SQL_RESULTS,
    PacketType.DONE
}

class RootOrchestrator:
    def __init__(self):
        self.skill_manager = SkillManager()
        self.memory_service = MemoryService()
        self.agent_service = AgentService()
        self.intent_router = IntentRouter()
        self.planning_engine = PlanningEngine()

    async def chat_stream_pipeline(
        self,
        background_tasks: BackgroundTasks,
        user_id: str,
        session_id: str,
        agent_id: int,
        question: str,
        security_level: int,
        tags: list[str] | None = None
    ) -> AsyncGenerator[dict[str, Any], None]:
        # --- 0. 元数据初始化 ---
        start_time = datetime.now(timezone.utc)
        entry_id = str(uuid.uuid4())
        model_params = await self.agent_service.get_agent_model_params(agent_id)
        llm_model = model_params.llm_model
        embedding_model = model_params.txt_embedding_model
        
        # --- 1. 上下文预处理与画像同步 ---
        user_profile = await self.memory_service.get_user_profile(user_id)
        init_thought = {"type": PacketType.THOUGHT, "content": "正在同步用户画像与上下文...\n"}
        yield init_thought

        prepared = await self.memory_service.prepare_context_and_rewrite(
            user_id=user_id,
            session_id=session_id,
            raw_question=question,
            llm_model=llm_model,
            user_profile=user_profile
        )
        
        # --- 2. 精细化意图路由（含 SOP 工作流匹配） ---
        analysis = await self.intent_router.route(llm_model, prepared['standalone_query'], agent_id=agent_id)
        
        # 检测用户语言
        user_language = detect_user_language(question)

        # --- 3. 初始化 ContextMemory 实例 (严格对齐规范) ---
        ctx: ContextMemory = {
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "question": question,
            "standalone_query": prepared['standalone_query'],
            "search_keywords": prepared.get('search_keywords') or prepared.get('keywords') or "",
            "llm_model": llm_model,
            "embedding_model": embedding_model,
            "security_level": int(security_level),
            "tags": tags or [],
            "intent_context": analysis.model_dump() if hasattr(analysis, 'model_dump') else {},
            "runtime_plan": None,
            "current_step_index": 0,
            "current_execution": None,
            "execution_history": [],
            "variables": {
                "extracted_entities": getattr(analysis, "detected_entities", [])
            },
            "doc_results": [],
            "sql_results": [],
            "graph_results": [],
            "session_state": prepared.get('new_state', {}),
            "blocks": [init_thought],  # 初始 thought 也需保存以支持历史重放
            "user_language": user_language,
            "temp": {}
        }

        logger.info(f"[LangTrace] RootOrchestrator user_language={user_language!r} question={question[:80]!r} intent={analysis.intent}")

        # --- 4. 分支任务规划阶段 ---
        if analysis.intent == IntentType.CHITCHAT:
            ctx["runtime_plan"] = {
                "thought": "闲聊模式，直接生成回复。",
                "final_goal": "回答用户的日常寒暄",
                "plan_type": "workflow",
                "workflow_id": "sys_chitchat",
                "inputs": {"query": ctx["standalone_query"]},
                "steps": [{
                    "step_id": 1,
                    "skill": "ChitChatSkill", 
                    "task_description": ctx["standalone_query"],
                    "output_var": "chitchat_output",
                    "condition": None
                }]
            }
        elif analysis.intent == IntentType.OFF_TOPIC:
            ctx["runtime_plan"] = {
                "thought": "触发安全策略或超出边界，转入拒答。",
                "final_goal": "安全拦截并礼貌拒答",
                "plan_type": "workflow",
                "workflow_id": "sys_safety_gate",
                "inputs": {"query": ctx["standalone_query"]},
                "steps": [{
                    "step_id": 1,
                    "skill": "OffTopicSkill", 
                    "task_description": ctx["standalone_query"],
                    "output_var": "safety_output",
                    "condition": None
                }]
            }
        elif analysis.intent == IntentType.SYSTEM_CMD:
            ctx["runtime_plan"] = {
                "thought": "识别到系统管理指令。",
                "final_goal": "执行系统级元命令运维动作",
                "plan_type": "workflow",
                "workflow_id": "sys_command",
                "inputs": {"query": ctx["standalone_query"]},
                "steps": [{
                    "step_id": 1,
                    "skill": "SystemCommandSkill", 
                    "task_description": ctx["standalone_query"],
                    "output_var": "cmd_output",
                    "condition": None
                }]
            }
        else:
            try:
                async for packet in self.planning_engine.decide_stream(context=ctx):
                    if packet.get("type") == PacketType.THOUGHT:
                        ctx["blocks"].append(packet)
                        yield packet
            except Exception as e:
                logger.critical(f"决策引擎崩溃: {str(e)}")
                err_packet = {"type": PacketType.ERROR, "content": "决策引擎崩溃，无法继续执行任务。"}
                ctx["blocks"].append(err_packet)
                yield err_packet
                return

        # --- 5. 波次执行阶段 (支持并行) ---
        final_answer_accumulator = ""
        reasoning_triggered = False
        plan_steps = ctx["runtime_plan"].get("steps", []) if ctx["runtime_plan"] else []

        # 获取依赖图并计算执行波次
        dep_graph = ctx.get("temp", {}).get("_dep_graph", {})
        if dep_graph:
            waves = compute_execution_waves(plan_steps, dep_graph)
            logger.info(f"波次调度: {len(plan_steps)} 步骤 → {len(waves)} 个波次")
        else:
            # 无依赖图时回退为串行（每个步骤一个波次）
            waves = [[i] for i in range(len(plan_steps))]

        for wave_idx, wave_indices in enumerate(waves):
            wave_steps = [plan_steps[i] for i in wave_indices]

            if len(wave_steps) == 1:
                # 单步骤波次：串行执行
                async for packet in self._execute_single_step(
                    ctx, wave_steps[0], wave_indices[0]
                ):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")
                    if p_type == PacketType.SQL_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["sql_results"].append(content["data"])
                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

                    # 追踪 reasoning
                    skill_name = wave_steps[0].get("skill", "")
                    if skill_name in ("ReasoningSkill", "reasoning-skill"):
                        reasoning_triggered = True
            else:
                # 多步骤波次：并行执行
                logger.info(f"Wave {wave_idx}: 并行执行 {len(wave_steps)} 个步骤")
                async for packet in self._execute_wave_parallel(
                    ctx, wave_steps, wave_indices
                ):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")
                    if p_type == PacketType.SQL_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["sql_results"].append(content["data"])
                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

        # --- 6. 兜底强制总结判定 ---
        if (ctx["doc_results"] or ctx["sql_results"]) and not reasoning_triggered:
            reasoning_skill = self.skill_manager.get_skill_instance("ReasoningSkill")
            if reasoning_skill:
                # 在兜底生命周期内，动态派生全新的沙箱环境
                fallback_runtime = SkillRuntime(context=ctx)
                exec_info = fallback_runtime.create_execution_context({
                    "step_id": 99,
                    "skill": "ReasoningSkill", 
                    "task_description": "结合检索素材进行深度推理总结。",
                    "output_var": "final_summary_output",
                    "condition": None
                })
                async for packet in fallback_runtime.execute_skill(reasoning_skill, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")
                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")
                    # 保存到 blocks 以支持历史重放
                    if p_type != PacketType.DONE:
                        ctx["blocks"].append(packet)
                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet

        # --- 6.5 合并相邻同类型 block，避免流式传输碎片化影响历史重放 ---
        merged_blocks = []
        for blk in ctx["blocks"]:
            if blk["type"] in (PacketType.THOUGHT, PacketType.ANSWER):
                if merged_blocks and merged_blocks[-1]["type"] == blk["type"]:
                    merged_blocks[-1]["content"] += (blk.get("content") or "")
                else:
                    merged_blocks.append(blk)
            else:
                merged_blocks.append(blk)
        ctx["blocks"] = merged_blocks

        # --- 7. 异步非阻塞记忆持久化与画像反思 ---
        plan_skills_trace = [s.get("skill") for s in plan_steps] if plan_steps else []
        current_plan_payload = {
            "skill_sequence": plan_skills_trace,
            "total_steps": len(plan_skills_trace)
        }
        
        logger.debug(f"[Pipeline] 提交记忆持久化任务: entry={entry_id}, session={session_id}")

        # 使用 asyncio.create_task 而非 background_tasks.add_task，
        # 避免在非流式接口中 BackgroundTasks 不执行导致记忆丢失
        task = asyncio.create_task(
            self.memory_service.persist_and_reflect_memory(
                session_id=session_id,
                user_id=user_id,
                entry_id=entry_id,
                raw_question=question,
                answer=final_answer_accumulator.strip() or "任务处理完成。",
                model_params=model_params,
                prepared_data={
                    **prepared,
                    "thought": ctx["runtime_plan"].get("thought", "") if ctx["runtime_plan"] else "",
                    "current_plan": current_plan_payload,
                    "turn_type": "chitchat" if analysis.intent in [IntentType.CHITCHAT, IntentType.OFF_TOPIC] else "task_oriented"
                },
                context_memory=ctx,
                request_time=start_time,
                response_time=datetime.now(timezone.utc)
            )
        )
        task.add_done_callback(
            lambda t: logger.error(f"[MemoryPersist] 后台记忆持久化任务异常: {t.exception()}", exc_info=t.exception())
            if t.exception() else None
        )

        logger.debug(f"[Pipeline] 记忆持久化任务已提交: entry={entry_id}")
        
        yield {"type": PacketType.DONE, "content": {"entry_id": entry_id}}

    # ═══════════════════════════════════════════════════════════════
    # 步骤执行辅助方法
    # ═══════════════════════════════════════════════════════════════

    async def _execute_single_step(
        self,
        ctx: ContextMemory,
        step: TaskStep,
        step_index: int,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """执行单个步骤（串行路径）"""
        ctx["current_step_index"] = step_index

        runtime = SkillRuntime(context=ctx)
        exec_info = runtime.create_execution_context(step_config=step)
        skill_name = exec_info["skill"]

        # 下发 CALL 信号
        call_packet = {
            "type": PacketType.CALL,
            "content": {
                "skill": skill_name,
                "description": step["task_description"]
            }
        }
        ctx["blocks"].append(call_packet)
        yield call_packet

        skill_instance = self.skill_manager.get_skill_instance(skill_name)
        if not skill_instance:
            exec_info.update({"status": "failed", "error": f"组件 {skill_name} 损坏或未注册"})
            ctx["execution_history"].append(exec_info)
            err_packet = {"type": PacketType.ERROR, "content": f"⚠️ 关键组件 [{skill_name}] 离线，本步骤跳过"}
            ctx["blocks"].append(err_packet)
            yield err_packet
            return

        try:
            async for packet in runtime.execute_skill(skill_instance, exec_info):
                self._collect_packet_to_blocks(ctx, packet)
                yield packet
        except Exception as e:
            logger.error(f"运行时在驱动组件 {skill_name} 时遭遇未知错误: {e}")

    async def _execute_wave_parallel(
        self,
        ctx: ContextMemory,
        wave_steps: list[TaskStep],
        step_indices: list[int],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        并行执行一个波次内的多个步骤。

        策略:
        - THOUGHT 实时 yield（前端可看到各步骤进度）
        - ANSWER/DATA 类 packet 也实时 yield
        - 使用 asyncio.Queue 收集各任务输出，避免竞态
        """
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        step_count = len(wave_steps)

        async def run_one(step: TaskStep, idx: int):
            """并行任务：执行单个步骤，将输出推入队列"""
            try:
                async for packet in self._execute_single_step(ctx, step, idx):
                    await queue.put(packet)
            except Exception as e:
                logger.error(f"并行步骤执行异常: {e}")
                await queue.put({
                    "type": PacketType.ERROR,
                    "content": f"并行步骤 {step.get('skill', '?')} 执行异常: {e}"
                })
            finally:
                await queue.put(None)  # 哨兵：标记本任务完成

        # 启动所有并行任务
        tasks = [
            asyncio.create_task(run_one(step, idx))
            for step, idx in zip(wave_steps, step_indices)
        ]

        # 收集输出直到所有任务完成
        finished = 0
        while finished < step_count:
            packet = await queue.get()
            if packet is None:
                finished += 1
                continue

            p_type = packet.get("type")
            if p_type != PacketType.DONE:
                self._collect_packet_to_blocks(ctx, packet)

            yield packet

        # 确保所有任务清理完毕
        await asyncio.gather(*tasks, return_exceptions=True)

    def _collect_packet_to_blocks(self, ctx: ContextMemory, packet: dict[str, Any]) -> None:
        """将 packet 收集到 ctx['blocks']，合并相邻同类型文本块"""
        p_type = packet.get("type")
        content = packet.get("content", "")

        if p_type == PacketType.DONE:
            return

        if p_type in (PacketType.THOUGHT, PacketType.ANSWER) and ctx["blocks"]:
            if ctx["blocks"][-1]["type"] == p_type:
                ctx["blocks"][-1]["content"] += (content or "")
            else:
                ctx["blocks"].append({"type": p_type, "content": content or ""})
        else:
            ctx["blocks"].append({"type": p_type, "content": content})
