from datetime import datetime
from loguru import logger
import json
import re
from core.database.oracle import get_session
from core.exceptions import InternalServerError
from dao.entities import MemoryEntryEntity
from dao.repositories import MemoryEntryRepository
from .state_manager import SessionStateManager
from .context_manager import ContextManager
from services.agent.agent_params import ModelParams
from utils.clients.model_client import AIModelClient
from utils.common import safe_read_content


class MemoryService:
    def __init__(self):
        self.manager = ContextManager()
        self.model_client = AIModelClient()
    
    @property
    def oracle_session(self):
        return get_session()
    
    # ========================== 画像读取与准备 ==========================

    async def get_user_profile(self, user_id: str) -> dict:
        """获取用户画像并转换为用于 Context 的字典格式"""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            profile = await repo.get_user_profile(user_id)
            
            if not profile:
                return {}

            # 将实体中的多个字段合并为一个扁平化的画像字典
            combined_profile = {
                "profile_summary": safe_read_content(profile.profile_summary) or "",
                **(profile.global_preferences or {}),
                **(profile.frequent_entities or {})
            }
            return combined_profile

    async def prepare_context_and_rewrite(
        self, 
        session_id: str, 
        raw_question: str,
        llm_model: str,
        user_profile: dict | None = None  # 1. 接收从 Orchestrator 传来的画像
    ) -> dict:
        """
        功能: 加载画像与上下文 -> 注入重写 -> 状态合并
        """
        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            
            # 1.短期记忆补全
            recent_entries = await context_repo.get_recent_entries(session_id, limit=5)
            # 将对话对象转为 LLM 可理解的文本
            # 建议格式：User: xxx \n Assistant: yyy
            chat_history = ""
            for entry in recent_entries:
                ans_preview = entry.answer[:100] if entry.answer else '[无回答]'
                chat_history += f"User: {entry.raw_question}\nAssistant: {ans_preview}...\n"

            # 2. 提取 Session 状态和历史摘要
            context = await context_repo.get_context_by_id(session_id)
            old_state = (context.session_state if context else {}) or {}
            history_summary = context.context_summary if context else ""

            # 3. 优先级合并：Session State 覆盖 User Profile
            rewrite_context_state = {**(user_profile or {}), **old_state}

            # 4. 调用 LLM 改写模块 (传入合并后的认知信息)
            rewrite_data = await self.manager.process_query_with_memory(
                query=raw_question,
                chat_history=chat_history,
                context_summary=history_summary,
                session_state=rewrite_context_state, 
                model_name=llm_model
            )

            # 5. 内存合并新状态
            new_state = SessionStateManager.merge_state(
                rewrite_context_state, 
                rewrite_data.get('turn_entities')
            )

            # 6. 处理重写器提取的即时画像更新
            profile_updates = rewrite_data.get('user_profile_updates', {})
            if profile_updates:
                new_state = {**new_state, **profile_updates}

            return {
                "standalone_query": rewrite_data['standalone_query'],
                "search_keywords": rewrite_data['search_keywords'],
                "intent_category": rewrite_data['intent_category'],
                "turn_entities": rewrite_data['turn_entities'],
                "user_profile_updates": profile_updates, # 显式返回，方便后续写入
                "new_state": new_state,
                "old_context": context
            }

    # ========================== 长期画像刷新 (Reflection) ==========================

    async def _do_llm_reflection(
        self, 
        user_id: str, 
        old_summary: str,
        question: str, 
        answer: str,
        llm_model: str
    ) -> tuple[str, str]:
        """
        调用 LLM 深度加工记忆。
        1. 更新全局画像 (Profile Summary)
        2. 提炼本轮记忆快照 (Memory Snapshot)
        """
        reflection_prompt = f"""
你是一位资深的系统架构师与用户画像专家。请分析对话并输出 JSON 格式的更新记录。

### 原有画像摘要:
{old_summary}

### 最新对话片段:
Q: {question}
A: {answer}

### 任务指令:
1. 分析最新对话，提取用户的专业身份(如DevOps)、使用的技术栈(如Oracle Linux 8)、当前关注的具体项目或痛点。
2. 将新提取的信息与原有摘要进行逻辑合并。
3. 如果信息重复，则保留；如果信息冲突（如用户从 Ubuntu 换到了 RHEL），以最新对话为准。
4. 保持摘要简洁、专业，总字数不超过 300 字。
5. 直接输出更新后的摘要文本，不要包含“根据对话...”等废话。
### 额外任务：
6. 请为本次对话生成一个【记忆快照】（Memory Snapshot），用于语义搜索。
### 任务要求:
1. 更新【profile_summary】：合并新老信息，字数<300，专业简洁。如果原有摘要为默认初始化信息，请直接以本次对话内容开启新摘要。
2. 生成【memory_snapshot】：本次对话的核心事实快照，需消解指代（如“它”->“Oracle 26ai”），去除废话。

### 输出格式 (必须为纯 JSON):
{{
    "profile_summary": "更新后的完整画像摘要...",
    "memory_snapshot": "本次对话的高纯度事实快照..."
}}
"""

        logger.debug(f"Triggering profile reflection for user: {user_id}")
        
        # 调用 LLM 提炼记忆
        new_summary = ""
        async for chunk in self.model_client.call_llm_model(llm_model, reflection_prompt, stream=False):
            # 处理非流式返回的完整字典 (参考测试3的日志结构)
            if isinstance(chunk, dict):
                # 尝试从非流式结构提取: choices[0].message.content
                choices = chunk.get("choices", [{}])
                message = choices[0].get("message", {})
                content = message.get("content")
                
                # 如果是流式结构，content 会在 delta 里
                if content is None:
                    content = choices[0].get("delta", {}).get("content", "")
                
                new_summary += (content or "")
            
            # 处理可能的字符串返回
            elif isinstance(chunk, str):
                # 如果 model_client 内部没做 json.loads，这里需要解析
                if chunk.startswith("{"):
                    try:
                        data = json.loads(chunk)
                        new_summary += data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    except:
                        pass
                else:
                    new_summary += chunk

        if not new_summary:
            logger.warning("Reflection failed: LLM returned empty summary.")
            return old_summary, f"Q: {question}\nA: {answer}"

        # 解析 LLM 返回的 JSON 内容
        try:
            json_str = new_summary.strip()
            # 在解析前增加更强的正则提取
            json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            
            # 针对 LLM 可能返回 Markdown JSON 块的情况进行清洗
            if json_str.startswith("```json"):
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif json_str.startswith("```"):
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            res_data = json.loads(json_str)
            profile_summary = res_data.get("profile_summary", old_summary)
            memory_snapshot = res_data.get("memory_snapshot", f"Q: {question}\nA: {answer}")
            
        except Exception as e:
            logger.error(f"Failed to parse Reflection JSON: {e}. Raw: {new_summary}")
            # 兜底逻辑：解析失败则只更新画像，或按原始文本处理
            profile_summary = new_summary[:300]
            memory_snapshot = question

        return profile_summary, memory_snapshot
            

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
        request_time: datetime,
        response_time: datetime,
        retrieved_chunks: list | None = None
    ):
        """统一的记忆持久化与反思任务, 仅用于后台异步调用"""
        # 获取 LLM 和 Embedding 模型
        llm_model = model_params.llm_model
        embedding_model = model_params.embedding_model

        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            try:
                # --------- STAGE 1: 基础持久化 ---------

                # 1. 更新 Context 状态 (Session 级别的结构化上下文)
                await repo.update_context_state(
                    session_id=session_id,
                    new_state=prepared_data.get('new_state', {})
                )

                # 2. 创建并保存 Memory Entry (对话流水账)
                standalone_query=prepared_data.get('standalone_query', raw_question)
                new_entry = MemoryEntryEntity(
                    entry_id=entry_id,
                    session_id=session_id,
                    raw_question=raw_question,
                    answer=answer,
                    standalone_query=standalone_query,
                    search_keywords=prepared_data.get('search_keywords', ""),
                    turn_entities=prepared_data.get('turn_entities', {}), 
                    intent_category=prepared_data.get('intent_category', "general"),
                    retrieved_chunks=retrieved_chunks,
                    request_time=request_time,
                    response_time=response_time
                )
                await repo.add_memory_entry(new_entry)
                logger.info(f"Interaction persisted for session: {session_id}")

                # 3. 增量更新 Profile 表中的结构化 JSON 字段 (global_preferences 等)
                # 从 prepared_data 中提取由 ContextManager 识别的画像更新
                profile_updates = prepared_data.get("user_profile_updates", {})
                if profile_updates:
                    await repo.upsert_user_profile(
                        user_id=user_id,
                        profile_updates=profile_updates
                    )
                logger.info(f"Full persistence cycle completed for session: {session_id}")
                # 手动触发提交，对话记录比画像更新更重要，防止 LLM 反思（Stage 2）因为网络或 API 限制超时，导致 Stage 1 已经写入的基础对话数据被一起回滚
                await session.commit()

                # --------- STAGE 2: 语义反思与向量化 ---------

                # 1. 获取当前最新画像
                user_profile = await self.get_user_profile(user_id)
                old_summary = user_profile.get("profile_summary", "Automatically initialized user profile")

                # 2. 构建反思 Prompt，要求模型提炼并合并新信息
                new_summary, memory_snapshot = await self._do_llm_reflection(user_id, old_summary, standalone_query, answer, llm_model)

                # 5. 获取语义向量 (异步，不阻塞画像更新)
                logger.debug(f"Vectorizing snapshot for entry {entry_id}...")
                try:
                    memory_vector = await self.model_client.call_embedding_model(embedding_model, [memory_snapshot])
                    if memory_vector:
                        vector = memory_vector[0].embedding
                    else:
                        vector = None
                except Exception as e:
                    logger.error(f"Failed to vectorize snapshot: {e}")
                    vector = None

                # 一次性回写用户画像和记忆向量
                # 1. 更新长期画像
                await repo.update_user_profile_summary(user_id=user_id, profile_summary=new_summary)
                # 2. 回填本轮 Entry 的向量与增强文本
                # 这样这条记录从此刻起，就能在未来的 search_vector_memory 中被检索到了
                await repo.update_entry_vector(
                    entry_id=entry_id,
                    vector=vector,
                    summary=memory_snapshot
                )
                logger.debug(f"LLM Reflection Result: {new_summary}")
                logger.info(f"Successfully updated profile summary for user: {user_id}")
                logger.info(f"Successfully refined memory for entry_id: {entry_id}")

            except Exception as e:
                logger.error(f"Error during memory capsule refresh: {e}")
                raise InternalServerError(f"Failed to persist memory: {e}")
                

    async def get_relevant_memories(self, user_id: str, query_vector: list) -> str:
        """
        召回长期记忆并格式化为带权重的字符串
        """
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            # 1. 向量检索 (假设你的 repo 已经支持向量搜索并过滤了 feedback >= 0)
            entries = await repo.search_vector_memory(
                user_id=user_id, 
                query_vector=query_vector, 
                limit=5
            )
            
            if not entries:
                return ""

            # 2. 格式化逻辑：将对象转为带权重的字符串
            memory_blocks = []
            for e in entries:
                # 这里的 e 是 MemoryEntryEntity 对象
                # 再次确保过滤掉用户点踩的内容（如果 repo 层没过滤干净）
                if e.feedback == -1:
                    continue
                    
                # 根据反馈强度添加视觉引导标签
                prefix = "⭐ [高价值历史方案]" if e.feedback == 1 else "[历史参考]"
                
                block = f"{prefix}\n问题: {e.raw_question}\n方案: {e.answer}"
                memory_blocks.append(block)

            return "\n\n".join(memory_blocks)
    
    async def record_user_feedback(self, entry_id: int, score: int):
        """
        记录用户对某一轮回答的满意度
        """
        if score not in [-1, 0, 1]:
            raise ValueError("Feedback score must be -1, 0, or 1")

        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            try:
                await context_repo.update_feedback(entry_id, score)
                await context_repo.session.commit()
                
                if score == -1:
                    # 策略：如果是差评，可以在此处记录日志或推送到飞书/钉钉告警，
                    # 方便开发人员后续针对性调优 RAG
                    logger.warning(f"User negative feedback received for entry: {entry_id}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to record feedback for {entry_id}: {e}")
                await context_repo.session.rollback()
                return False
            
    async def ensure_session_exists(self, session_id: str, user_id: str, agent_id: int, question: str | None = None):
        """确保 Oracle 中存在会话主表记录，防止外键约束报错"""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            try:
                await repo.ensure_session(session_id=session_id, agent_id=agent_id, user_id=user_id, question=question)
            except Exception as e:
                logger.error(f"Check session {session_id} failed")
                raise InternalServerError(f"Check session {session_id} failed: {e}")
            
    async def sync_user_profile(self, user_id: str, profile_updates: dict) -> None:
        """
        后台任务专用：仅同步更新用户长期画像
        """
        if not profile_updates:
            return

        # 后台任务需要独立获取 session，确保事务完整
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            try:
                logger.info(f"Background Task: Updating profile for user {user_id}")
                
                # 调用 repo 层进行 JSON_MERGEPATCH 或 覆盖逻辑
                await repo.upsert_user_profile(
                    user_id=user_id,
                    profile_updates=profile_updates
                )
                
                # 必须在后台任务内部提交
                await session.commit()
                logger.info(f"Background Task: Profile sync completed for {user_id}")
                
            except Exception as e:
                logger.error(f"Background Task: Profile sync failed for {user_id}: {e}")
                # 后台任务失败通常不抛出异常给前端，但需要记录日志
                await session.rollback()