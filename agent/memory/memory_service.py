from datetime import datetime
from typing import Any
from loguru import logger
import json
import re
from core.database import db_instance
from core.config import get_prompt_config
from core.exceptions import InternalServerError
from dao.entities import MemoryEntryEntity
from dao.repositories import MemoryRepository
from .state_manager import SessionStateManager
from .context_manager import ContextManager
from agent.prompt import default_prompt
from agent.common import ContextMemory
from services.kb import ModelParams
from utils.clients.model import AIModelClient
from utils.thread import safe_read_content


class MemoryService:
    def __init__(self):
        self.manager = ContextManager()
        self.model_client = AIModelClient()
        self.user_profile_prompt = get_prompt_config().user_profile
    
    @property
    def db_session(self):
        return db_instance().get_session()
    
    # ========================== 画像读取与准备 ==========================

    async def get_user_profile(self, user_id: str) -> dict:
        """获取用户画像并转换为用于 Context 的字典格式"""
        async with self.db_session as session:
            repo = MemoryRepository(session)
            profile = await repo.get_user_profile(user_id)

        if not profile:
            return {}

        # 仅偏好和实体参与状态合并，profile_summary 由 reflection 单独管理
        return {
            **(profile.global_preferences or {}),
            **(profile.frequent_entities or {}),
        }

    async def prepare_context_and_rewrite(
        self, 
        user_id: str,
        session_id: str, 
        raw_question: str,
        llm_model: str,
        user_profile: dict | None = None  # 1. 接收从 Orchestrator 传来的画像
    ) -> dict:
        """
        功能: 加载画像与上下文 -> 注入重写 -> 状态合并
        修正点：
        1. 增加对 current_plan 的加载（如果存在未完成的任务）。
        2. 增加对 active_topic 的感知。
        """
        async with self.db_session as session:
            context_repo = MemoryRepository(session)

            # 1. 加载 Session 上下文
            context = await context_repo.get_context_by_id(session_id)

            # 2. 获取短期对话历史 (格式化逻辑保持不变)
            recent_entries = await context_repo.get_recent_entries(session_id, limit=5)
        chat_history_list = []
        for i, entry in enumerate(recent_entries):
            # 如果是最后一次对话（索引通常是 0 或 len-1，取决于你的排序），保留更长内容
            is_last_turn = (i == 0) # 假设最新的一条在 index 0
            limit = 500 if is_last_turn else 150
            
            ans_content = entry.answer[:limit] if entry.answer else '[无回答]'
            chat_history_list.append(f"User: {entry.raw_question}\nAssistant: {ans_content}")

        # 倒序排列，确保时间线正确（从旧到新）
        chat_history = "\n\n".join(reversed(chat_history_list))

        old_state = (context.state_machine if context else {}) or {}
        history_summary = (context.context_summary if context else "") or ""
        
        # --- 新增：获取当前的执行计划和主题 ---
        current_plan = (context.execution_plan if context else None)
        active_topic = (context.active_topic if context else None)

        # 3. 优先级合并：Session State 覆盖 User Profile
        rewrite_context_state = {**(user_profile or {}), **old_state}

        # 4. 构造上下文线索
        context_hint = ""
        if active_topic:
            context_hint = f"当前对话正在讨论的话题是: {active_topic}。"
        if current_plan:
            context_hint += f"系统正在执行的任务计划: {current_plan.get('thought', '')}。"

        # 5. 调用 LLM 改写模块
        rewrite_data = await self.manager.process_query_with_memory(
            query=raw_question,
            chat_history=chat_history,
            context_summary=f"{history_summary}\n{context_hint}", # 将 hint 注入 context_summary
            session_state=rewrite_context_state, 
            model_name=llm_model,
            active_topic=active_topic 
        )

        # 5. 合并状态 (逻辑保持不变)
        new_state = SessionStateManager.merge_state(
            rewrite_context_state, 
            rewrite_data.get('turn_entities')
        )

        return {
            "standalone_query": rewrite_data["standalone_query"],
            "search_keywords": rewrite_data["search_keywords"],
            "turn_type": rewrite_data["turn_type"],
            "turn_entities": rewrite_data["turn_entities"],
            "new_state": new_state,
            "old_context": context,
            "current_plan": current_plan,
            "active_topic": active_topic,
            "context_summary": rewrite_data.get("context_summary", ""),
            "user_profile_updates": rewrite_data.get("user_profile_updates", {}),
        }

    # ========================== 长期画像刷新 (Reflection) ==========================

    async def _do_llm_reflection(
        self,
        user_id: str,
        old_summary: str,
        question: str,
        answer: str,
        llm_model: str
    ) -> dict[str, Any]:
        """
        调用 LLM 深度加工记忆。
        返回: {profile_summary, memory_snapshot, global_preferences, frequent_entities, corrections}
        """
        reflection_prompt = await default_prompt.generate(
            self.user_profile_prompt,
            old_summary=old_summary,
            question=question,
            answer=answer
        )
        logger.debug(f"触发用户 {user_id} 画像刷新。")

        try:
            res_data = await self.model_client.get_llm_json(
                model_name=llm_model,
                prompt=reflection_prompt,
                temperature=0.0
            )

            prefs = res_data.get("global_preferences") or {}
            ents = res_data.get("frequent_entities") or {}
            rels = res_data.get("entity_relations") or []
            corrs = res_data.get("correction_history") or []
            logger.info(
                f"[Reflection] user={user_id} "
                f"prefs_keys={list(prefs.keys())} ent_count={len(ents)} "
                f"rel_count={len(rels)} corr_count={len(corrs)}"
            )
            return {
                "profile_summary": res_data.get("profile_summary", old_summary),
                "memory_snapshot": res_data.get("memory_snapshot", f"Q: {question}\nA: {answer}"),
                "global_preferences": prefs,
                "frequent_entities": ents,
                "entity_relations": rels,
                "corrections": corrs,
            }

        except Exception as e:
            logger.error(f"用户 {user_id} 画像反思处理失败: {e}")
            return {
                "profile_summary": old_summary,
                "memory_snapshot": f"Q: {question}\nA: {answer}",
                "global_preferences": {},
                "frequent_entities": {},
                "entity_relations": [],
                "corrections": [],
            }
            
    # ========================== 持久化与同步 ==========================
    async def persist_and_reflect_memory(
        self,
        session_id: str,
        user_id: str,
        entry_id: str,
        raw_question: str,
        answer: str,
        model_params: ModelParams,
        prepared_data: dict, 
        context_memory: ContextMemory, # 这里现在是包含 execution_history, sql_results, doc_results 等的新结构
        request_time: datetime,
        response_time: datetime
    ):
        """
        统一的记忆持久化与反思任务。
        已适配新的 context_memory 结构。
        """
        llm_model = model_params.llm_model
        embedding_model = model_params.txt_embedding_model
        standalone_query = prepared_data.get('standalone_query', raw_question)

        async with self.db_session as session:
            repo = MemoryRepository(session)
            try:
                # ========== 1. 从 context_memory 提取执行轨迹 ==========
                reasoning_path_structured = []
                execution_history = context_memory.get("execution_history") or []

                for step in execution_history:
                    if isinstance(step, dict):
                        t_start = step.get("start_time")
                        t_end = step.get("end_time")
                        skill_name = step.get("skill") or step.get("skill_name", "unknown")
                        status = step.get("status", "unknown")
                        task_desc = step.get("task_description", "")
                        error = step.get("error") if status == "failed" else None
                        resolved_input = step.get("resolved_input")
                    else:
                        t_start = getattr(step, 'start_time', None)
                        t_end = getattr(step, 'end_time', None)
                        skill_name = getattr(step, 'skill', None) or getattr(step, 'skill_name', 'unknown')
                        status = getattr(step, 'status', 'unknown')
                        task_desc = getattr(step, 'task_description', "")
                        error = getattr(step, 'error', None) if status == "failed" else None
                        resolved_input = getattr(step, 'resolved_input', None)

                    duration = None
                    if t_start and t_end:
                        try:
                            if isinstance(t_start, str):
                                t_start = datetime.fromisoformat(t_start)
                            if isinstance(t_end, str):
                                t_end = datetime.fromisoformat(t_end)
                            duration = int((t_end - t_start).total_seconds() * 1000)
                        except Exception:
                            pass

                    step_info = {
                        "skill": str(skill_name) if skill_name else "unknown",
                        "status": str(status) if status else "unknown",
                        "duration_ms": duration,
                        "task_description": str(task_desc)[:200] if task_desc else "",
                    }
                    if error:
                        step_info["error"] = str(error)[:500]
                    if resolved_input:
                        step_info["resolved_input"] = str(resolved_input)[:1200]
                    reasoning_path_structured.append(step_info)

                # ========== 2. 从 context_memory 提取统计信息 ==========
                sql_results = context_memory.get("sql_results") or []
                doc_results = context_memory.get("doc_results") or []

                execution_stats = {
                    "total_steps": len(reasoning_path_structured),
                    "successful_steps": sum(1 for s in reasoning_path_structured if s["status"] == "success"),
                    "failed_steps": sum(1 for s in reasoning_path_structured if s["status"] == "failed"),
                    "total_duration_ms": int((response_time - request_time).total_seconds() * 1000),
                    "sql_queries_executed": len(sql_results) if isinstance(sql_results, list) else 1 if sql_results else 0,
                    "documents_retrieved": len(doc_results),
                }

                # ========== 3. 实体识别 ==========
                # 优先使用 LLM 提取的本轮实体，fallback 为空
                turn_entities = prepared_data.get('turn_entities') or {}

                # ========== 4. 持久化存储 ==========
                new_state = prepared_data.get('new_state') or {}
                current_plan = prepared_data.get('current_plan')

                await repo.update_context_state(
                    session_id=session_id,
                    new_state=new_state,
                    active_topic=prepared_data.get('active_topic'),
                    current_plan=current_plan,
                    increment_count=True,
                )
                # 持久化上下文摘要：优先 LLM 改写输出，兜底用记忆快照
                ctx_summary = prepared_data.get("context_summary") or ""
                if not ctx_summary:
                    ctx_summary = standalone_query  # 至少用改写后的问题作为摘要
                await repo.update_context_summary(session_id, ctx_summary)

                new_entry = MemoryEntryEntity(
                    entry_id=entry_id,
                    session_id=session_id,
                    user_id=user_id,
                    raw_question=raw_question,
                    answer=answer,
                    standalone_query=standalone_query,
                    current_plan=current_plan,
                    search_keywords=prepared_data.get('search_keywords', ""),
                    thought=prepared_data.get('thought', ''),
                    reasoning_path=reasoning_path_structured,
                    turn_entities=turn_entities,
                    blocks=context_memory.get("blocks") or [],
                    request_time=request_time,
                    response_time=response_time,
                    turn_type=prepared_data.get('turn_type', "task_oriented"),
                    memory_summary="",
                )
                await repo.add_memory_entry(new_entry)

                # ========== 5. 反思逻辑 ==========
                profile_entity = await repo.get_user_profile(user_id=user_id)
                old_summary = (profile_entity.profile_summary or "") if profile_entity else ""
                if not old_summary:
                    old_summary = "新用户，暂无历史画像"

                reflection_context = self._build_reflection_context(
                    standalone_query=standalone_query,
                    answer=answer,
                    execution_stats=execution_stats,
                    reasoning_path=reasoning_path_structured,
                    evidence_count=execution_stats["documents_retrieved"],
                )

                reflection = await self._do_llm_reflection(
                    user_id=user_id, old_summary=old_summary,
                    question=standalone_query, answer=reflection_context,
                    llm_model=llm_model,
                )

                # 向量化
                memory_snapshot = reflection["memory_snapshot"]
                vector = None
                if memory_snapshot:
                    res = await self.model_client.call_embedding_model(embedding_model, [memory_snapshot])
                    if res:
                        vector = res[0].embedding

                # 持久化画像摘要
                await repo.update_user_profile_summary(user_id=user_id, profile_summary=reflection["profile_summary"])
                # 合并所有画像更新到一次调用
                profile_updates: dict[str, Any] = dict(prepared_data.get("user_profile_updates") or {})
                if reflection.get("entity_relations"):
                    profile_updates["entity_relations"] = reflection["entity_relations"]
                if reflection.get("corrections"):
                    profile_updates["correction_history"] = reflection["corrections"]
                await repo.upsert_user_profile(
                    user_id=user_id,
                    profile_updates=profile_updates
                )

                # 回写条目向量
                await repo.update_entry_vector(entry_id=entry_id, vector=vector, summary=memory_snapshot)

            except Exception as e:
                logger.error(f"Persistence error ({entry_id}): {e}", exc_info=True)


    def _build_reflection_context(
        self,
        standalone_query: str,
        answer: str,
        execution_stats: dict,
        reasoning_path: list,
        evidence_count: int
    ) -> str:
        """
        构建用于反思的增强上下文
        """
        # 增加对"决策质量"的描述
        quality_label = "顺利" if execution_stats.get('failed_steps', 0) == 0 else "存在阻碍"
        
        # 获取详细的失败分析
        failure_analysis = self._extract_failed_reasons(reasoning_path)
        
        # 组装最终给 LLM 反思的文本
        # 包含：问题、答案、耗时统计、SQL执行情况、失败细节
        sql_info = f"，执行了 {execution_stats['sql_queries_executed']} 个 SQL 查询" if execution_stats.get('sql_queries_executed', 0) > 0 else ""

        return (
            f"### 任务回放 ###\n"
            f"用户问题: {standalone_query}\n"
            f"系统回答: {answer[:500]}\n\n"
            f"### 执行元数据 ###\n"
            f"整体状态: {quality_label}\n"
            f"步骤总数: {execution_stats.get('total_steps', 0)} 步\n"
            f"总耗时: {execution_stats.get('total_duration_ms', 0)}ms{sql_info}\n"
            f"证据支持: 引用了 {evidence_count} 份外部文档\n\n"
            f"### 异常与诊断 ###\n"
            f"{failure_analysis}"
        )

    def _extract_failed_reasons(self, reasoning_path: list) -> str:
        """
        从执行路径中提取失败原因，用于反思上下文。
        """
        if not reasoning_path:
            return "无执行记录。"

        failed_steps = [step for step in reasoning_path if step.get('status') == 'failed']
        
        if not failed_steps:
            return "执行过程顺利，未遇到异常。"

        reasons = []
        for step in failed_steps:
            skill = step.get('skill', '未知技能')
            # 优先获取结构化错误，如果没有则展示任务描述
            error_msg = step.get('error') or step.get('task_description') or "未捕获到具体异常详情"
            # 限制长度，防止单个错误占用过大 Prompt 空间
            reasons.append(f"技能 [{skill}] 执行失败，原因: {str(error_msg)[:150]}")

        return "发现以下执行异常：\n" + "\n".join([f"  - {r}" for r in reasons])

    async def _send_execution_metrics(self, execution_stats: dict, user_id: str, session_id: str):
        """
        发送执行统计到监控系统（可选）
        """
        try:
            # 这里可以集成 Prometheus、StatsD 等监控系统
            # 示例：记录关键指标
            metrics = {
                "execution_duration_ms": execution_stats['total_duration_ms'],
                "execution_steps_total": execution_stats['total_steps'],
                "execution_steps_failed": execution_stats['failed_steps'],
                "documents_retrieved": execution_stats.get('documents_retrieved', 0),
                "sql_queries": execution_stats.get('sql_queries_executed', 0)
            }
            
            # 示例：记录到日志或发送到监控
            logger.debug(f"执行指标 [{session_id}]: {metrics}")
            
            # 如果已经集成了监控客户端
            # await metrics_client.gauge("agent.execution.duration", execution_stats['total_duration_ms'], tags={"user_id": user_id})
            # await metrics_client.increment("agent.execution.steps", execution_stats['total_steps'])
            
        except Exception as e:
            logger.debug(f"发送执行指标失败: {e}")

    async def get_relevant_memories(self, user_id: str, query_vector: list) -> str:
        """
        召回长期记忆并格式化为带权重的字符串 (适配结构化推理路径)
        """
        async with self.db_session as session:
            repo = MemoryRepository(session)

            # 1. 向量检索
            entries = await repo.search_vector_memory(
                user_id=user_id,
                query_vector=query_vector,
                limit=5
            )
        
        if not entries:
            return ""

        memory_blocks = []
        for e in entries:
            # 1. 严格过滤负反馈 (用户显式标记为错误的经验不应误导 LLM)
            if getattr(e, 'feedback', 0) == -1:
                continue
                
            # 2. 优先级标记
            prefix = "⭐ [重点参考-高价值经验]" if getattr(e, 'feedback', 0) == 1 else "[历史参考]"
            
            # 3. 修复 Reasoning Path 提取逻辑 (处理结构化字典)
            formatted_path = ""
            raw_path = getattr(e, 'reasoning_path', [])
            if raw_path and isinstance(raw_path, list):
                path_names = []
                for step in raw_path:
                    if isinstance(step, dict):
                        # 仅提取技能名称，可选包含状态
                        name = step.get("skill", "unknown")
                        # 如果该步骤失败了，打个标记提示 LLM 避坑
                        if step.get("status") == "failed":
                            name += "(failed)"
                        path_names.append(name)
                    else:
                        path_names.append(str(step))
                
                if path_names:
                    formatted_path = f"\n执行链路: {' -> '.join(path_names)}"

            # 4. 内容选择策略：反思快照优先
            # memory_summary 是 LLM 深度思考后的精髓，比 raw_question 更利于模型理解
            summary = getattr(e, 'memory_summary', None)
            if summary and summary.strip():
                content = f"经验总结: {summary}"
            else:
                content = f"历史问题: {getattr(e, 'raw_question', '')}\n对应方案: {getattr(e, 'answer', '')[:500]}"
            
            # 5. 格式化时间
            req_time = getattr(e, 'request_time', None)
            time_str = req_time.strftime('%Y-%m-%d') if req_time else "未知时间"
            
            # 组装 Block
            block = f"{prefix} (日期: {time_str}){formatted_path}\n{content}"
            memory_blocks.append(block)

        if not memory_blocks:
            return ""
            
        return "\n\n--- 相关历史记忆 (可参考以下过往经验) ---\n" + \
            "\n\n".join(memory_blocks) + \
            "\n------------------------------------------"
    
    async def record_user_feedback(self, user_id: str, entry_id: str, feedback: int):
        """
        记录用户对某一轮回答的满意度
        """
        if feedback not in [-1, 0, 1]:
            raise ValueError("反馈值必须为 -1、0 或 1")

        async with self.db_session as session:
            context_repo = MemoryRepository(session)
            try:
                await context_repo.update_feedback(entry_id, feedback)

                if feedback == -1:
                    # 策略：如果是差评，可以在此处记录日志或推送到飞书/钉钉告警，
                    # 方便开发人员后续针对性调优 RAG
                    logger.warning(f"收到用户对记录 {entry_id} 的负面反馈")

                return True
            except Exception as e:
                logger.error(f"记录 {entry_id} 的反馈信息失败: {e}")
                return False

    async def ensure_session_exists(self, session_id: str, user_id: str, agent_id: int, question: str | None = None):
        """确保数据库中存在会话 Context 主记录，防止写入 Entry 时找不到父文档"""
        async with self.db_session as session:
            repo = MemoryRepository(session)
            try:
                await repo.ensure_session(
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    question=question
                )
            except Exception as e:
                logger.error(f"检查或创建会话上下文失败 {session_id}: {e}")
                raise InternalServerError(f"会话初始化失败: {e}")

    async def sync_user_profile(self, user_id: str, profile_updates: dict) -> None:
        """
        后台任务专用：仅同步更新用户长期画像 (如 Preferences, Entity Stats)
        """
        if not profile_updates:
            return

        async with self.db_session as session:
            repo = MemoryRepository(session)
            try:
                logger.debug(f"后台任务：正在增量同步用户 {user_id} 的结构化画像字段")
                await repo.upsert_user_profile(user_id=user_id, profile_updates=profile_updates)
                logger.info(f"用户 {user_id} 画像同步完成")

            except Exception as e:
                # 后台任务的错误通常不抛出给前端，但必须记录详细日志
                logger.error(f"后台任务：用户 {user_id} 画像同步失败，数据: {profile_updates}, Error: {e}")

    async def update_context_state(self, user_id: str, session_id: str, new_state: dict) -> None:
        """
        更新会话 State（实体、路径等临时变量）
        """
        async with self.db_session as session:
            repo = MemoryRepository(session)
            try:
                await repo.update_context_state(
                    session_id=session_id,
                    new_state=new_state,
                    increment_count=True
                )
            except Exception as e:
                logger.error(f"更新会话状态失败 {session_id}: {e}")
                raise InternalServerError(f"会话状态更新失败: {e}")

    async def get_context_data(self, user_id: str, session_id: str) -> dict | None:
        """
        纯粹的上下文读取接口：供上层 Agent 或专职 Skill 获取历史持久化的 State 字典
        """
        async with self.db_session as session:
            repo = MemoryRepository(session)
            try:
                context = await repo.get_context_by_id(session_id)
                if context and context.state_machine:
                    # 确保返回的是纯字典，方便上层做类型转换或二次加工
                    return context.state_machine
                return None
            except Exception as e:
                logger.error(f"从仓库加载会话上下文失败 {session_id}: {e}")
                return None