import time, json
from datetime import datetime, timezone
from loguru import logger
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks
from typing import AsyncGenerator

from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.exceptions import *
from dao.repositories import FileRepository
from services.search.result import TxtBaseSearchResult
from services.search.rerank import TxtBaseRerank
from services.search.kb_search import TxtBaseSearch
from utils.clients import AIModelClient
from services.ai_model import AIModelService
from services.memory import MemoryService
from .orchestrator import ChatOrchestrator
from .agent_params import ModelParams

class ChatService:
    def __init__(self):
        self.rerank_client = TxtBaseRerank()
        self.tb_search = TxtBaseSearch()
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
        self.memory_service = MemoryService()
        self.orchestrator = ChatOrchestrator()

    @property
    def oracle_session(self):
        return get_session()

    # ========================== 核心业务接口 ==========================

    async def stream_chat(
        self, 
        background_tasks: BackgroundTasks, 
        session_id: str, 
        agent_id: int, 
        question: str, 
        security_level: int, 
        user_id: str, 
        tags: list[str] = []
    ) -> StreamingResponse:
        """流式对话入口：将流程委托给 Orchestrator"""
        request_time = datetime.now(tz=timezone.utc)
        logger.info(f"Stream chat started: session={session_id}, agent={agent_id}")

        # 1. 运行核心流水线获取最终提示词和知识库检索结果
        pipe_out = await self.orchestrator.run_pipeline(
            user_id, session_id, agent_id, question, security_level, tags
        )

        # 2. 返回 StreamingResponse
        return StreamingResponse(
            self._generate_chat_stream(
                background_tasks, 
                session_id, 
                user_id, 
                question, 
                pipe_out['kb_results'], 
                pipe_out['prepared_data'],
                pipe_out['model_params'], 
                pipe_out['final_prompt'], 
                request_time
            ),
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    async def non_stream_chat(
        self, session_id: str, agent_id: int, question: str, 
        security_level: int, user_id: str, tags: list[str] = []
    ) -> dict:
        """非流式对话入口"""
        request_time = datetime.now(tz=timezone.utc)
        
        pipe_out = await self.orchestrator.run_pipeline(
            user_id, session_id, agent_id, question, security_level, tags
        )
        
        # 调用 LLM 并收集完整结果
        full_answer = ""
        async for chunk in self.model_client.call_llm_model(
            pipe_out['model_params']['llm_model_name'], 
            pipe_out['final_prompt']
        ):
            temp_chunks = []
            await self._collect_chunks(chunk, temp_chunks)
            if temp_chunks: full_answer += "".join(temp_chunks)

        enriched_refs = await self._enrich_results_with_metadata(pipe_out['kb_results'])

        # 持久化：使用 MemoryService 的新闭环方法
        await self.memory_service.finalize_and_persist(
            session_id=session_id,
            raw_question=question,
            answer=full_answer,
            prepared_data=pipe_out['prepared_data'],
            retrieved_chunks=enriched_refs,
            request_time=request_time
        )

        return {
            "question": question,
            "answer": full_answer,
            "qa_embedding": pipe_out['model_params'].get("query_vec"),
            "references": enriched_refs,
            "request_time": request_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "response_time": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        }

    
    
    # ========================== Generators & Persistence Helpers ==========================

    async def _generate_chat_stream(
        self, 
        background_tasks: BackgroundTasks, 
        session_id: str, 
        user_id: str,
        question: str,
        kb_results: list[TxtBaseSearchResult], 
        prepared_data: dict,
        model_params: ModelParams, 
        final_prompt: str, 
        request_time: datetime
    ) -> AsyncGenerator:
        """Internal generator: yields LLM chunks and pushes post-processing tasks."""
        answer_chunks = []
        llm_params = model_params.llm_params
        if llm_params:
            llm_args = {k: v for k, v in {
                "max_tokens": llm_params.get("llm_max_tokens"),
                "temperature": llm_params.get("llm_temperature"),
                "top_p": llm_params.get("llm_top_p"),
                "top_k": llm_params.get("llm_top_k"),
            }.items() if v is not None}

        try:
            logger.info(f"Calling LLM: {model_params.llm_model} for session {session_id}")
            async for chunk in self.model_client.call_llm_model(model_params.llm_model, final_prompt, **llm_args):
                yield chunk
                await self._collect_chunks(chunk, answer_chunks)
        except Exception as e:
            logger.error(f"Exception during LLM stream generation: {e}")
            yield json.dumps({'type': 'error', 'message': str(e)}) + '\n'
        finally:
            # Send reference metadata to frontend
            logger.debug(f"Stream generation finished. Sending references for session {session_id}")
            references = await self._enrich_results_with_metadata(kb_results)
            yield json.dumps({'type': 'reference', 'references': references, 'is_complete': True}) + '\n'

            # Add background task for ES persistence
            background_tasks.add_task(
                self._persist_memory_cycle,
                session_id=session_id,
                raw_question=question,
                chunks=answer_chunks,
                prepared_data=prepared_data,
                references=references,
                request_time=request_time
            )

    # ========================== Utils & Helpers ==========================

    def _get_stream_headers(self) -> dict:
        """Returns standard SSE response headers."""
        return {"Cache-Control": "no-cache", "Connection": "keep-alive", "Access-Control-Allow-Origin": "*"}

    async def _collect_chunks(self, chunk, chunks: list):
        """Extracts content from LLM stream chunks and accumulates them."""
        if isinstance(chunk, str) and chunk.startswith('data: '):
            data = chunk[6:].strip()
            if data == '[DONE]': return
            try:
                js = json.loads(data)
                # Handle error responses
                if 'error' in js:
                    logger.error(f"LLM error response: {js.get('error')}")
                    return
                content = js.get("choices", [{}])[0].get("delta", {}).get("content")
                if content: chunks.append(content)
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Failed to parse chunk: {e}, chunk: {chunk[:100] if chunk else ''}")
        elif isinstance(chunk, dict):
            try:
                # Handle error responses
                if 'error' in chunk:
                    logger.error(f"LLM error response: {chunk.get('error')}")
                    return
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content: chunks.append(content)
            except (KeyError, IndexError, AttributeError) as e:
                logger.debug(f"Failed to parse dict chunk: {e}, chunk keys: {chunk.keys() if chunk else ''}")
        else:
            logger.debug(f"Unknown chunk type: {type(chunk)}, content: {str(chunk)[:100] if chunk else ''}")

    async def _process_answer(self, chunks) -> str:
        """Joins accumulated chunks into a single string."""
        str_chunks = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in chunks]
        return "".join(str_chunks).strip()

    async def _persist_memory_cycle(
        self, 
        session_id: str, 
        raw_question: str, 
        chunks: list, 
        references: list, 
        prepared_data: dict,
        request_time: datetime
    ):
        """
        闭环持久化：
        1. 组合 LLM 碎片为完整回答
        2. 调用 MemoryService.finalize_and_persist 
        3. 同步更新 Oracle 中的 State 和 Summary
        """
        try:
            if not chunks:
                logger.warning(f"No content chunks to persist for session {session_id}")
                return

            # 1. 还原回答内容
            full_answer = await self._process_answer(chunks)
            
            logger.info(f"[Memory-Cycle] Starting persistence for session {session_id}")

            # 2. 调用 MemoryService 的高级持久化接口
            # prepared_data 包含了：new_state, standalone_query, search_keywords 等
            await self.memory_service.finalize_and_persist(
                session_id=session_id,
                raw_question=raw_question,
                answer=full_answer,
                prepared_data=prepared_data,
                retrieved_chunks=references,  # 传入已经 enrich 过的 metadata
                request_time=request_time
            )

            logger.info(f"[Memory-Cycle] State, Summary, and Chat record synchronized for {session_id}")

        except Exception as e:
            import traceback
            logger.error(f"Failed to finalize memory cycle: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")

    async def _enrich_results_with_metadata(self, kb_results: list[TxtBaseSearchResult]) -> list[dict]:
        """Converts raw search results into dictionaries with file metadata for frontend display."""
        if not kb_results:
            return []

        # Safely collect file_ids (ensure they are strings)
        file_ids = []
        for res in kb_results:
            file_id = res.file_id
            if not isinstance(file_id, str):
                file_id = str(file_id)
            file_ids.append(file_id)

        unique_file_ids = list(set(file_ids))
        logger.debug(f"Collected {len(unique_file_ids)} unique file IDs from {len(kb_results)} results")

        file_name_map = {}
        try:
            async with self.oracle_session as session:
                file_repo = FileRepository(session)
                file_name_map = await file_repo.get_names_by_ids(unique_file_ids)
                logger.debug(f"Mapped {len(file_name_map)} file IDs to names.")
        except Exception as e:
            logger.error(f"Failed to fetch file names for references: {e}")

        config = get_app_config()
        base_url = f"http://{config.host_ip}:{config.service_port}"

        references = []
        for idx, res in enumerate(kb_results):
            try:
                ref = res.to_dict()

                # Ensure file_id is a string for URL construction
                file_id = res.file_id
                if not isinstance(file_id, str):
                    file_id = str(file_id)

                ref["file_name"] = file_name_map.get(file_id, "Unknown File")
                ref["download_link"] = f"{base_url}/api/kb/download?file_id={file_id}"
                ref["preview_link"] = f"{base_url}/api/kb/preview?file_id={file_id}"
                references.append(ref)
            except Exception as e:
                logger.error(f"Error processing search result at index {idx}: {e}, type: {type(e).__name__}, res: {res}")
                raise
        return references
