import time
import asyncio
from loguru import logger
from typing import Any
from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.exceptions import *
from dao.entities import AgentConfEntity, AgentEntity
from dao.repositories import (AgentRepository, AgentConfRepository, MemoryEntryRepository,
                             PromptRepository, FileRepository)
from services.search.result import TxtBaseSearchResult
from services.search.rerank import TxtBaseRerank
from utils.clients import AIModelClient
from services.memory import MemoryService, ContextManager
from services.ai_model import AIModelService
from services.search.kb_search import TxtBaseSearch
from services.search.result import TxtBaseSearchResult
from .agent_params import ModelParams

class ChatOrchestrator:
    def __init__(self):
        self.tb_search = TxtBaseSearch()
        self.rerank_client = TxtBaseRerank()
        self.memory_service = MemoryService()
        self.context_manager = ContextManager()
        self.model_client = AIModelClient()
        self.model_service = AIModelService()

    @property
    def oracle_session(self):
        return get_session()

    async def run_pipeline(
        self,
        user_id: str, 
        session_id: str, 
        agent_id: int, 
        question: str, 
        security_level: int, 
        tags: list | None = None
    ) -> dict[str, Any]:
        """
        聊天流程编排
        """
        # 1. 加载画像 (Long-term Profile) 和 会话上下文 (Short-term Context)
        # 这一步是"认识用户"的开始
        user_profile = await self.memory_service.get_user_profile(user_id) # 获取用户画像

        # 获取 Agent 和 Model 参数
        async with self.oracle_session as session:
            agent, model_params = await self._get_agent_and_params(session, agent_id)

        # 1. 记忆预处理：获取 State, Summary 并生成 standalone_query
        prepared = await self.memory_service.prepare_context_and_rewrite(
            session_id=session_id,
            raw_question=question,
            llm_model=model_params.llm_model,
            user_profile=user_profile # 传入画像
        )
        
        # 2. 知识库检索：使用改写后的 standalone_query 和 search_keywords
        kb_results, model_params, query_vec = await self._execute_knowledge_search_pipeline(
            agent_id=agent_id,
            security_level=security_level,
            question=prepared['standalone_query'], 
            keywords=prepared['search_keywords'],
            model_params=model_params,
            tags=tags
        )

        # 3. 跨会话长期记忆召回
        long_term_memory = await self.memory_service.get_relevant_memories(
            user_id=user_id,
            query_vector=query_vec
        )

        # 4. 组装最终 Prompt
        final_prompt = self.context_manager.build_final_prompt(
            system_prompt=prepared.get('system_prompt', "You are a helpful assistant."),
            user_question=prepared['standalone_query'],
            kb_results=kb_results,
            session_state=prepared['new_state'],
            context_summary=prepared['old_context'].context_summary if prepared['old_context'] else "",
            long_term_memory=long_term_memory
        )

        return {
            "final_prompt": final_prompt,
            "kb_results": kb_results,
            "model_params": model_params,
            "prepared_data": prepared  # 包含 new_state 和改写后的信息，用于后续持久化
        }
    
    async def _execute_knowledge_search_pipeline(
        self, 
        agent_id: int, 
        security_level: int, 
        question: str,
        keywords: str,
        model_params: ModelParams,
        tags: list | None = None
    ) -> tuple[list[TxtBaseSearchResult], ModelParams, list[float]]:
        """
        Core pipeline: handles configuration loading, question vectorization, 
        knowledge base retrieval, and reranking. Shared by search and stream_chat.
        """
        # 1. Vectorize the question
        logger.info(f"Step 1: Vectorize the question, calling embedding model: {model_params.embedding_model} for question: {question}")

        # Validate question is not empty
        question = question.strip() if question else ""
        if not question:
            logger.error(f"Question cannot be empty for agent {agent_id}")
            raise ParamValueError(f"Question cannot be empty")

        embed_resp = await self.model_client.call_embedding_model(model_params.embedding_model, [question])

        # Validate embed_resp is a list and has at least one item
        if not embed_resp:
            logger.error(f"Embedding service returned empty list for question: {question}")
            raise InternalServerError("Embedding service returned empty result")
        
        if not isinstance(embed_resp, list):
            logger.error(f"Embedding response is not a list: {type(embed_resp)}, content: {embed_resp}")
            raise InternalServerError(f"Embedding service returned unexpected type: {type(embed_resp).__name__}")

        query_vec = embed_resp[0].embedding
        if not query_vec:
            logger.error(f"Embedding vector is empty for question: {question}")
            raise InternalServerError("Embedding vector is empty")
        
        # 2. Execute retrieval and reranking
        logger.info(f"Step 2: Starting KB search for agent {agent_id} with security level {security_level}")
        start_time = time.time()
        kb_results = await self._retrieve_result(
            agent_id=agent_id,
            security_level=security_level,
            question=question,
            query_vec=query_vec,
            keywords=keywords,
            rerank_model=model_params.rerank_model,
            rerank_top_k=model_params.rerank_top_k,
            tags=tags
        )
        logger.info(f"KB search completed in {time.time() - start_time:.2f}s, found {len(kb_results)} results.")
        
        return kb_results, model_params, query_vec
    
    async def _get_agent_and_params(self, session, agent_id: int) -> tuple[AgentEntity, ModelParams]:
        """Fetches Agent entity and parses its model configurations."""
        logger.debug(f"Starting _get_agent_and_params for agent_id={agent_id}")
        agent_repo = AgentRepository(session)
        agent = await agent_repo.get_by_id(agent_id)
        if not agent:
            logger.error(f"Agent {agent_id} not found in database.")
            raise NotFoundError(f"Agent {agent_id} does not exist.")
        
        logger.debug(f"Successfully retrieved agent {agent_id}, parsing model params...")
        model_params = await self._get_model_params(agent)
        logger.debug(f"Model params retrieved for agent {agent_id}: embedding={model_params.embedding_model}, llm={model_params.llm_model}")
        return agent, model_params
    
    async def _get_model_params(self, agent: AgentEntity) -> ModelParams:
        """Resolves model names from IDs and organizes parameters."""
        llm_model = await self.model_service.get_display_name_by_id(agent.llm_id) if agent.llm_id else None
        emb_model = await self.model_service.get_display_name_by_id(agent.embedding_model_id) if agent.embedding_model_id else None
        rerank_model = await self.model_service.get_display_name_by_id(agent.reranker_model_id) if agent.reranker_model_id else None
        if not llm_model or not emb_model:
            logger.error(f"No valid LLM/Embedding model configured for agent {agent.id}")
            raise NotFoundError("Agent has no valid LLM/Embedding model configured.")

        params = ModelParams(
            llm_model=llm_model,
            llm_params=agent.llm_params,
            embedding_model=emb_model,
            rerank_model=rerank_model,
            rerank_top_k=agent.reranker_topk or 10
        )
        return params
    
    async def _retrieve_result(
        self, 
        agent_id: int, 
        security_level: int, 
        question: str, 
        query_vec: list[float],
        keywords: str, 
        rerank_top_k: int, 
        rerank_model: str | None = None, 
        tags: list | None = None
    ) -> list[TxtBaseSearchResult]:
        """Retrieves and optionally reranks results from multiple KB configurations."""
        async with self.oracle_session as session:
            conf_repo = AgentConfRepository(session)
            agent_confs = await conf_repo.get_by_agent_id(agent_id)

        rerank_results, norerank_results = await self._search_text_base(
            agent_confs, security_level, question, keywords, query_vec, tags
        )

        if not rerank_model or len(rerank_results) <= 1:
            logger.debug("Skipping rerank as model is missing or results are insufficient.")
            return rerank_results + norerank_results

        logger.info(f"Applying rerank model: {rerank_model} for {len(rerank_results)} items.")
        final = await self.rerank_client.rerank(
            model_name=rerank_model, top_k=rerank_top_k, question=question, kb_results=rerank_results
        )
        final.extend(norerank_results)
        final.sort(key=lambda x: x.weight, reverse=True)
        return final
    
    async def _search_text_base(
        self, 
        confs: list[AgentConfEntity], 
        security: int, 
        question: str, 
        keywords: str, 
        query_vec: list[float],
        tags: list | None = None
    ) -> tuple[list, list]:
        """Prepares and dispatches KB search tasks."""
        tb_tasks = []
        for conf in confs:
            tb_tasks.append((
                int(conf.tool_id), 
                str(question), 
                str(keywords),
                int(conf.search_topk or 10), 
                float(conf.search_score_threshold or 0.0),
                bool(conf.reranker_flag == 1), 
                float(conf.tool_weight or 1.0),
                int(security),
                query_vec, 
                tags
            ))

        rerank_all, norerank_all = [], []
        if tb_tasks:
            results = await self._run_tb_search_parallel(tb_tasks)
            for res in results:
                rerank_all.extend(res.get("rerank_result", []))
                norerank_all.extend(res.get("norerank_result", []))
        return rerank_all, norerank_all
    
    async def _run_tb_search_parallel(self, tb_tasks: list[tuple]) -> list[dict]:
        """Executes KB search tasks in parallel using asyncio.gather."""
        async_tasks = [self.tb_search.search(*task) for task in tb_tasks]
        raw_results = await asyncio.gather(*async_tasks, return_exceptions=True)
        processed_results = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                logger.error(f"KB Task {i} failed with error: {res}")
                processed_results.append({"rerank_result": [], "norerank_result": []})
            else:
                processed_results.append(res)
        return processed_results