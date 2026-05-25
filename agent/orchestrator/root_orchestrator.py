import uuid
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
from skills import SkillRuntime
from agent.common import ContextMemory, ExecutionPlan
from core.dictionary import IntentType, PacketType
from utils.simulate_stream import simulate_stream


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
        
        # --- 1. 上下文预处理与画像同步 ---
        user_profile = await self.memory_service.get_user_profile(user_id)
        content = "Synchronizing user profile and context...\n"
        async for char in simulate_stream(content):
            yield {"type": PacketType.THOUGHT, "content": char}

        prepared = await self.memory_service.prepare_context_and_rewrite(
            user_id=user_id,
            session_id=session_id,
            raw_question=question,
            llm_model=llm_model,
            user_profile=user_profile
        )
        
        # --- 2. 精细化意图路由 ---
        analysis = await self.intent_router.route(llm_model, prepared['standalone_query'])
        
        # --- 3. 初始化 ContextMemory 实例 (严格对齐规范) ---
        ctx: ContextMemory = {
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "question": question,
            "standalone_query": prepared['standalone_query'],
            "llm_model": llm_model,
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
            "blocks": [],
            "temp": {}
        }

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
                        yield packet
            except Exception as e:
                logger.critical(f"决策引擎崩溃: {str(e)}")
                content = "Decision engine crashed, unable to proceed with the task."
                async for char in simulate_stream(content):
                    yield {"type": PacketType.ERROR, "content": char}
                return

        # --- 5. 统一流式循环执行阶段 ---
        final_answer_accumulator = ""
        reasoning_triggered = False
        plan_steps = ctx["runtime_plan"].get("steps", []) if ctx["runtime_plan"] else []

        for idx, step in enumerate(plan_steps):
            ctx["current_step_index"] = idx
            
            # 每个独立步骤，精准拉起一个专属的 SkillRuntime 隔离沙箱
            runtime = SkillRuntime(context=ctx)
            
            # 建立执行快照（Runtime 内部自动完成输入变量占位符替换）
            exec_info = runtime.create_execution_context(step_config=step)
            skill_name = exec_info["skill"]
            
            if skill_name in ("ReasoningSkill", "reasoning"):
                reasoning_triggered = True

            # B. 向前端下发明确的组件唤醒信号
            yield {
                "type": PacketType.CALL, 
                "skill": skill_name, 
                "description": exec_info["resolved_input"]
            }
            
            skill_instance = self.skill_manager.get_skill_instance(skill_name)
            if not skill_instance:
                exec_info.update({"status": "failed", "error": f"组件 {skill_name} 损坏或未注册"})
                ctx["execution_history"].append(exec_info)
                content = f"⚠️ Critical component [{skill_name}] offline, skip current step"
                async for char in simulate_stream(content):
                    yield {"type": PacketType.ERROR, "content": char}
                continue

            try:
                # C. 托管给沙箱驱动具体的技能执行周期
                async for packet in runtime.execute_skill(skill_instance, exec_info):
                    p_type = packet.get("type")
                    content = packet.get("content")

                    if p_type == PacketType.ANSWER:
                        final_answer_accumulator += (content or "")

                    if p_type == PacketType.SQL_RESULTS:
                        if isinstance(content, dict) and "data" in content:
                            ctx["sql_results"].append(content["data"])

                    # UI Block 聚合（在 Orchestrator 层做防碎化去重）
                    if p_type not in (PacketType.DONE, PacketType.CALL):
                        if p_type in (PacketType.THOUGHT, PacketType.ANSWER) and ctx["blocks"]:
                            if ctx["blocks"][-1]["type"] == p_type:
                                ctx["blocks"][-1]["content"] += (content or "")
                            else:
                                ctx["blocks"].append({"type": p_type, "content": content or ""})
                        else:
                            ctx["blocks"].append({"type": p_type, "content": content})

                    if p_type in DISPLAY_PACKET_TYPES:
                        yield packet
                        
            except Exception as e:
                logger.error(f"运行时在驱动组件 {skill_name} 时遭遇未知错误: {e}")

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
                    if packet.get("type") == PacketType.ANSWER:
                        final_answer_accumulator += (packet.get("content") or "")
                    if packet.get("type") in DISPLAY_PACKET_TYPES:
                        yield packet

        # --- 7. 异步非阻塞记忆持久化与画像反思 ---
        plan_skills_trace = [s.get("skill") for s in plan_steps] if plan_steps else []
        current_plan_payload = {
            "skill_sequence": plan_skills_trace,
            "total_steps": len(plan_skills_trace)
        }
        
        background_tasks.add_task(
            self.memory_service.persist_and_reflect_memory,
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
            context_memory=ctx,  # 完美沉淀清洗后的 ContextMemory
            request_time=start_time,
            response_time=datetime.now(timezone.utc)
        )
        
        yield {"type": PacketType.DONE, "entry_id": entry_id}