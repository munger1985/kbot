import asyncio
import time
import json
from datetime import datetime, timezone
from typing import Sequence, Any

from loguru import logger
from fastapi.responses import StreamingResponse
from fastapi import BackgroundTasks

from core.database.oracle import get_session
from core.config.settings import get_app_config
from core.exceptions import *
from dao.entities import AgentConfEntity, AgentEntity
from dao.repositories import (AgentRepository, AgentConfRepository, ChatMemoryRepository,
                             PromptRepository, ChatSessionRepository, FileRepository)
from services.search.result import TxtBaseSearchResult
from services.search.rerank import TxtBaseRerank
from services.search.kb_search import TxtBaseSearch
from utils.clients import AIModelClient
from services.ai_model import AIModelService
from services.agent.memory import MemoryService
from services.agent.context_builder import ContextBuilder


class AgentService:
    """Agent service class for managing knowledge base search and AI chat interactions."""

    def __init__(self):
        """Initializes service clients and user context."""
        self.rerank_client = TxtBaseRerank()
        self.tb_search = TxtBaseSearch()
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
        self.context_builder = ContextBuilder()
        self.memory_service = MemoryService()


    @property
    def oracle_session(self):
        """Provides a database session instance."""
        return get_session()

    # ========================== Core Private Pipeline ==========================

    async def _get_agent_and_params(self, session, agent_id: int) -> tuple[AgentEntity, dict[str, Any]]:
        """Fetches Agent entity and parses its model configurations."""
        logger.debug(f"[AgentService] Starting _get_agent_and_params for agent_id={agent_id}")
        agent_repo = AgentRepository(session)
        agent = await agent_repo.get_by_id(agent_id)
        if not agent:
            logger.error(f"[AgentService] Agent {agent_id} not found in database.")
            raise NotFoundError(f"Agent {agent_id} does not exist.")
        
        logger.debug(f"[AgentService] Successfully retrieved agent {agent_id}, parsing model params...")
        model_params = await self._get_model_params(agent)
        logger.debug(f"[AgentService] Model params retrieved for agent {agent_id}: embedding={model_params.get('embedding_model_name')}, llm={model_params.get('llm_model_name')}")
        return agent, model_params

    async def _execute_knowledge_search_pipeline(
        self, agent_id: int, security_level: int, question: str, tags: list = []
    ) -> tuple[list[TxtBaseSearchResult], dict[str, Any]]:
        """
        Core pipeline: handles configuration loading, question vectorization, 
        knowledge base retrieval, and reranking. Shared by search and stream_chat.
        """
        async with self.oracle_session as session:
            _, model_params = await self._get_agent_and_params(session, agent_id)

        # 1. Vectorize the question
        embedding_model = model_params.get("embedding_model_name")
        if not embedding_model:
            logger.warning(f"Embedding model not configured for agent {agent_id}.")
            raise NotFoundError(f"Agent {agent_id} lacks embedding model configuration.")
        
        logger.debug(f"Calling embedding model: {embedding_model} for question: {question}")
        embed_resp = await self.model_client.call_embedding_model(embedding_model, [question])
        query_vec = embed_resp[0].embedding
        model_params["query_vec"] = query_vec 

        llm_model = model_params.get("llm_model_name")
        if not llm_model:
            raise NotFoundError(f"Agent {agent_id} lacks LLM model configuration.")
        
        rerank_model = model_params.get("rerank_model")
        if not rerank_model:
            raise NotFoundError(f"Agent {agent_id} lacks rerank model configuration.")
        
        # 2. Execute retrieval and reranking
        logger.info(f"Starting KB search for agent {agent_id} with security level {security_level}")
        start_time = time.time()
        kb_results = await self._retrieve_result(
            agent_id=agent_id,
            security_level=security_level,
            question=question,
            query_vec=query_vec,
            llm_model=llm_model,
            rerank_model=rerank_model,
            rerank_top_k=model_params.get("rerank_top_k", 10),
            tags=tags
        )
        logger.info(f"KB search completed in {time.time() - start_time:.2f}s, found {len(kb_results)} results.")
        
        return kb_results, model_params

    async def _enrich_results_with_metadata(self, kb_results: list[TxtBaseSearchResult]) -> list[dict]:
        """Converts raw search results into dictionaries with file metadata for frontend display."""
        if not kb_results:
            return []

        file_ids = list(set([res.file_id for res in kb_results]))
        file_name_map = {}
        try:
            async with self.oracle_session as session:
                file_repo = FileRepository(session)
                file_name_map = await file_repo.get_names_by_ids(file_ids)
                logger.debug(f"Mapped {len(file_name_map)} file IDs to names.")
        except Exception as e:
            logger.error(f"Failed to fetch file names for references: {e}")

        config = get_app_config()
        base_url = f"http://{config.host_ip}:{config.service_port}"

        references = []
        for res in kb_results:
            ref = res.to_dict()
            ref["file_name"] = file_name_map.get(res.file_id, "Unknown File")
            ref["download_link"] = f"{base_url}/api/kb/download?file_id={res.file_id}"
            ref["preview_link"] = f"{base_url}/api/kb/preview?file_id={res.file_id}"
            references.append(ref)
        return references

    async def _get_prompt_template(self, session, agent: AgentEntity) -> str:
        """Retrieves prompt template for chat; falls back to default if not configured."""
        if agent.prompt_id:
            try:
                prompt_repo = PromptRepository(session)
                template = await prompt_repo.get_prompt_by_id(agent.prompt_id)
                if template: 
                    logger.debug(f"Custom prompt template loaded for agent {agent.id}")
                    return template
            except Exception as e:
                logger.warning(f"Prompt ID {agent.prompt_id} configured but not found: {e}")
        
        logger.debug("Using default prompt template.")
        return "Please answer the question based on the reference content.\n\nReferences:{context}\n\nQuestion:{question}"

    # ========================== Public Business Interfaces ==========================

    async def search(self, agent_id: int, security_level: int, question: str, tags: list = []) -> list[TxtBaseSearchResult]:
        """Public interface: Performs KB search and returns raw result objects."""
        try:
            kb_results, _ = await self._execute_knowledge_search_pipeline(
                agent_id, security_level, question, tags
            )
            return kb_results
        except Exception as e:
            logger.error(f"Error in search interface: {e}")
            handle_exception(e, f"Knowledge base search failed: {e}")

    async def stream_chat(
        self, background_tasks: BackgroundTasks, session_id: str, agent_id: int, question: str, 
        security_level: int, user_id: str, tags: list[str] = []
    ) -> StreamingResponse:
        """Public interface: Handles retrieval, context building, LLM generation, and memory persistence."""
        request_time = datetime.now(tz=timezone.utc)
        logger.info(f"Initiating stream chat for session {session_id}, agent {agent_id}")

        # 1. Run retrieval pipeline
        kb_results, model_params = await self._execute_knowledge_search_pipeline(
            agent_id, security_level, question, tags
        )

        # 2. Load Prompt Template (On-demand)
        async with self.oracle_session as session:
            agent, _ = await self._get_agent_and_params(session, agent_id)
            system_prompt = await self._get_prompt_template(session, agent)

        # 3. Retrieve memory and build context
        memories = await self.memory_service.get_context_parts(
            session_id, question, query_vec=model_params["query_vec"]
        )
        
        final_prompt = self.context_builder.build_final_prompt(
            system_prompt=system_prompt,
            user_question=question,
            kb_results=kb_results,
            short_term_memory=memories.get("short_term", ""),
            long_term_memory=memories.get("long_term", "")
        )

        # 4. Return streaming response
        return StreamingResponse(
            self._generate_chat_stream(
                background_tasks, session_id, user_id, question, kb_results, model_params, final_prompt, request_time
            ),
            media_type="text/event-stream",
            headers=self._get_stream_headers()
        )

    # ========================== Generators & Persistence Helpers ==========================

    async def _generate_chat_stream(self, background_tasks, session_id, user_id, question, kb_results, model_params, final_prompt, request_time):
        """Internal generator: yields LLM chunks and pushes post-processing tasks."""
        chunks = []
        llm_name = model_params.get("llm_model_name")
        llm_args = {k: v for k, v in {
            "max_tokens": model_params.get("llm_max_tokens"),
            "temperature": model_params.get("llm_temperature"),
            "top_p": model_params.get("llm_top_p"),
            "top_k": model_params.get("llm_top_k"),
        }.items() if v is not None}

        try:
            logger.info(f"Calling LLM: {llm_name} for session {session_id}")
            async for chunk in self.model_client.call_llm_model(llm_name, final_prompt, **llm_args):
                yield chunk
                await self._collect_chunks(chunk, chunks)
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
                self._persist_chat_data,
                session_id=session_id,
                user_id=user_id,
                question=question,
                query_vec=model_params["query_vec"],
                chunks=chunks,
                references=references,
                request_time=request_time
            )

    async def _get_model_params(self, agent: AgentEntity) -> dict[str, Any]:
        """Resolves model names from IDs and organizes parameters."""
        llm_model = await self.model_service.get_model_name_by_id(agent.llm_id) if agent.llm_id else None
        if not llm_model:
            logger.error(f"No valid LLM model configured for agent {agent.id}")
            raise NotFoundError("Agent has no valid LLM model configured.")
        
        emb_model = await self.model_service.get_model_name_by_id(agent.embedding_model_id) if agent.embedding_model_id else None
        rerank_model = await self.model_service.get_model_name_by_id(agent.reranker_model_id) if agent.reranker_model_id else None

        params = {
            "llm_model_name": llm_model,
            "embedding_model_name": emb_model,
            "rerank_model": rerank_model,
            "rerank_top_k": agent.reranker_topk or 10,
            "do_rerank": rerank_model is not None,
        }
        if agent.llm_params:
            for key in ["max_tokens", "temperature", "top_p", "top_k"]:
                val = agent.llm_params.get(key)
                if val is not None: params[f"llm_{key}"] = val
        return params

    async def _run_tb_search_parallel(self, tb_tasks: list[tuple]) -> list[dict]:
        """Executes KB search tasks in parallel using asyncio.gather."""
        async_tasks = [self.tb_search.search(*task) for task in tb_tasks]
        try:
            return await asyncio.gather(*async_tasks)
        except Exception as e:
            logger.error(f"Parallel KB search failed: {e}")
            return [{"rerank_result": [], "norerank_result": []} for _ in tb_tasks]

    async def _search_text_base(self, confs: Sequence[AgentConfEntity], security: int, question: str, 
                                llm_model: str, query_vec: list[float], tags: list) -> tuple[list, list]:
        """Prepares and dispatches KB search tasks."""
        tb_tasks = []
        for conf in confs:
            tb_tasks.append((
                conf.tool_id, question, conf.search_topk, conf.search_score_threshold,
                conf.reranker_flag == 1, float(conf.tool_weight or 1.0),
                security, llm_model, query_vec, tags
            ))

        rerank_all, norerank_all = [], []
        if tb_tasks:
            results = await self._run_tb_search_parallel(tb_tasks)
            for res in results:
                rerank_all.extend(res.get("rerank_result", []))
                norerank_all.extend(res.get("norerank_result", []))
        return rerank_all, norerank_all

    async def _retrieve_result(self, agent_id: int, security_level: int, question: str, query_vec: list[float],
                               llm_model: str, rerank_model: str, rerank_top_k: int, tags: list) -> list[TxtBaseSearchResult]:
        """Retrieves and optionally reranks results from multiple KB configurations."""
        async with self.oracle_session as session:
            conf_repo = AgentConfRepository(session)
            agent_confs = await conf_repo.get_by_agent_id(agent_id)

        rerank_results, norerank_results = await self._search_text_base(
            agent_confs, security_level, question, llm_model, query_vec, tags
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
                content = js.get("choices", [{}])[0].get("delta", {}).get("content")
                if content: chunks.append(content)
            except: pass
        elif isinstance(chunk, dict):
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
            if content: chunks.append(content)

    async def _process_answer(self, chunks) -> str:
        """Joins accumulated chunks into a single string."""
        str_chunks = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in chunks]
        return "".join(str_chunks).strip()

    async def _persist_chat_data(self, session_id, user_id, question, query_vec, chunks, references, request_time):
        """Asynchronously saves chat interaction to long-term memory."""
        try:
            if not chunks: 
                logger.warning(f"No content chunks to persist for session {session_id}")
                return
            answer = await self._process_answer(chunks)
            logger.info(f"Persisting chat memory, session {session_id}")
            await self.memory_service.save_memory(
                session_id=session_id, user_id=user_id, question=question, answer=answer,
                query_vec=query_vec, references=references, request_time=request_time
            )
            logger.info(f"Memory persistence successful for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to persist chat memory: {e}")

    async def remove_agent(self, agent_id: int, del_prompt: bool = False):
        """Removes agent, its configurations, and optionally its associated prompts."""
        try:
            async with self.oracle_session as session:
                agent_repo = AgentRepository(session)
                if del_prompt:
                    agent = await agent_repo.get_by_id(agent_id)
                    if agent and agent.prompt_id:
                        await PromptRepository(session).delete(agent.prompt_id)
                        logger.info(f"Deleted prompt {agent.prompt_id} for agent {agent_id}")

                await AgentConfRepository(session).delete_by_agent_id(agent_id)
                await agent_repo.delete(agent_id)
                
                sess_repo = ChatSessionRepository(session)
                sessions = await sess_repo.get_by_agent(agent_id)
                await ChatMemoryRepository(session).delete_by_ids([s.id for s in sessions])
                await sess_repo.delete_by_agent_id(agent_id)
                logger.info(f"Successfully removed agent {agent_id} and related records.")
        except Exception as e:
            logger.error(f"Error removing agent {agent_id}: {e}")
            handle_exception(e, "Failed to remove agent.")

    async def feedback(self, chat_record_id: int, feedback: int):
        """Updates user feedback for a specific chat record."""
        async with self.oracle_session as session:
            logger.info(f"Submitting feedback {feedback} for record {chat_record_id}")
            await ChatMemoryRepository(session).feedback(chat_record_id, feedback)

    async def get_session_history(self, session_id: str):
        """Retrieves history for a specific chat session."""
        async with self.oracle_session as session:
            return await ChatMemoryRepository(session).get_session_history(session_id)

    async def remove_session(self, session_id: str):
        """Deletes a chat session and all its associated records."""
        async with self.oracle_session as session:
            try:
                await ChatMemoryRepository(session).delete_session_records(session_id)
                await ChatSessionRepository(session).delete(session_id)
                logger.info(f"Successfully removed session {session_id}")
            except Exception as e:
                logger.error(f"Error removing session {session_id}: {e}")
                raise InternalServerError(f"Failed to delete session: {e}")
            
    # ========================== Non-Streaming Chat Interface ==========================

    async def non_stream_chat(
        self, session_id: str, agent_id: int, question: str, 
        security_level: int, user_id: str, tags: list[str] = []
    ) -> dict:
        """
        Public interface: Synchronous (non-streaming) chat.
        Returns a specific dictionary format including embeddings and timing.
        """
        # Start timing and record request time
        start_ts = time.time()
        request_time_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
        
        logger.info(f"Initiating non-stream chat for session {session_id}, agent {agent_id}")

        try:
            # 1. Execute retrieval pipeline
            kb_results, model_params = await self._execute_knowledge_search_pipeline(
                agent_id, security_level, question, tags
            )

            # 2. Load Agent and Prompt Template
            async with self.oracle_session as session:
                agent, _ = await self._get_agent_and_params(session, agent_id)
                system_prompt = await self._get_prompt_template(session, agent)

            # 3. Build Context (Memory + KB)
            memories = await self.memory_service.get_context_parts(
                session_id, question, query_vec=model_params["query_vec"]
            )
            
            final_prompt = self.context_builder.build_final_prompt(
                system_prompt=system_prompt,
                user_question=question,
                kb_results=kb_results,
                short_term_memory=memories.get("short_term", ""),
                long_term_memory=memories.get("long_term", "")
            )

            # 4. Call LLM and collect full response
            llm_name = model_params.get("llm_model_name")
            if not llm_name:
                raise ParamValueError(f"llm_model_name is missing for agent {agent_id}")
            
            llm_args = {k: v for k, v in {
                "max_tokens": model_params.get("llm_max_tokens"),
                "temperature": model_params.get("llm_temperature"),
                "top_p": model_params.get("llm_top_p"),
                "top_k": model_params.get("llm_top_k"),
            }.items() if v is not None}

            full_answer = ""
            async for chunk in self.model_client.call_llm_model(llm_name, final_prompt, **llm_args):
                temp_chunks = []
                await self._collect_chunks(chunk, temp_chunks)
                if temp_chunks:
                    full_answer += "".join(temp_chunks)

            # 5. Prepare reference metadata
            enriched_refs = await self._enrich_results_with_metadata(kb_results)

            # Calculate response time and format timestamp
            response_time_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

            # 6. Persistence (Immediate)
            # We use the raw datetime objects for the database logic
            await self.memory_service.save_memory(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=full_answer,
                query_vec=model_params["query_vec"],
                references=enriched_refs,
                request_time=datetime.strptime(request_time_str, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            )

            # 7. Return the exact dictionary format requested
            return {
                "question": question,
                "answer": full_answer,
                "qa_embedding": model_params.get("query_vec"), # This is the vector generated during search
                "references": enriched_refs, 
                "feedback": 0,  # Default feedback value (neutral/none)
                "by": "agent",  # Source identifier
                "request_time": request_time_str,
                "response_time": response_time_str
            }

        except Exception as e:
            logger.error(f"Error in non-stream chat execution: {e}")
            handle_exception(e, f"Chat processing failed: {e}")