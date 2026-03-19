import os
import json
import asyncio
from datetime import datetime, timezone
from loguru import logger
from fastapi import BackgroundTasks

from .agent_service import AgentService
from core.exceptions import InternalServerError
from services.search.result import TxtBaseSearchResult

class DifyService:
    """Retrieval service adapter for Dify interface."""
    
    def __init__(self):
        # Initialize with a fixed user_id for Dify-sourced requests
        self.agent_service = AgentService()
        self.security_level = 9  # Level 9 bypasses security checks
        self.user_id="dify_system"

    async def search(self, 
                    agent_id: int, 
                    question: str, 
                    session_id: str,
                    background_tasks: BackgroundTasks | None = None
                    ) -> dict:
        """
        Agent interaction for Dify (Search + Record Persistence).
        """
        request_time = datetime.now(tz=timezone.utc)
        logger.info(f"Processing Dify request for session {session_id}, Agent {agent_id}")

        try:
            # 1. Execute the unified search pipeline
            # This handles embedding, search, and reranking in one go
            kb_results, model_params = await self.agent_service._execute_knowledge_search_pipeline(
                agent_id=agent_id,
                security_level=self.security_level,
                question=question
            )

            # 2. Build Dify-specific records and standardized references
            # We fetch file names and build the metadata required by Dify
            logger.debug(f"Starting to enrich {len(kb_results)} search results")
            try:
                references = await self.agent_service._enrich_results_with_metadata(kb_results)
                logger.debug(f"Successfully enriched {len(references)} references")
            except Exception as e:
                logger.error(f"Error in _enrich_results_with_metadata: {e}, type: {type(e).__name__}", exc_info=True)
                raise

            try:
                records = self._build_dify_records(references)
                logger.debug(f"Successfully built {len(records)} Dify records")
            except Exception as e:
                logger.error(f"Error in _build_dify_records: {e}, type: {type(e).__name__}", exc_info=True)
                raise

            # 3. Persistence (Sync with AgentService logic)
            # Since Dify usually doesn't need the LLM answer back from us (it does its own generation),
            # we record the 'question' and 'retrieval results'. 
            # If Dify expects us to save the interaction:
            
            persist_task = self.agent_service._persist_chat_data(
                session_id=session_id,
                user_id=self.user_id,
                question=question,
                query_vec=model_params.get("query_vec"),
                chunks=["[Dify Retrieval Only]"], # Placeholder for answer as Dify handles LLM
                references=references,
                request_time=request_time
            )

            if background_tasks:
                background_tasks.add_task(lambda: persist_task)
            else:
                # Fallback: execute as background task if no manager provided
                asyncio.create_task(persist_task)

            return {"records": records}

        except Exception as e:
            msg = f"Dify interaction failed for Agent {agent_id}: {str(e)}"
            # Safely format question for logging (handle non-string types)
            safe_question = str(question)[:100] if question else ""
            logger.error(f"[DifyService] {msg}, error type: {type(e).__name__}, "
                        f"session_id: {session_id}, question: {safe_question}", exc_info=True)
            raise InternalServerError(message=msg)

    def _build_dify_records(self, enriched_references: list[dict]) -> list[dict]:
        """
        Formats enriched references into Dify-compatible record structure.
        Uses the output from _enrich_results_with_metadata.
        """
        records = []
        for idx, ref in enumerate(enriched_references):
            try:
                # Handle content that might be a list or other type
                content = ref.get("content")
                if not isinstance(content, str):
                    content = str(content) if content else ""

                record = {
                    "metadata": {
                        "path": ref.get("download_link"),
                        "preview": ref.get("preview_link"),
                        "page": ref.get("page_num"),
                        "chunk_type": ref.get("chunk_type")
                    },
                    "score": ref.get("reranker_score") or ref.get("similarity_score"),
                    "title": ref.get("file_name", "Unknown File"),
                    "content": content
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Error building Dify record at index {idx}: {e}, type: {type(e).__name__}, ref type: {type(ref).__name__}")
                raise
        return records

    async def _remove_session_dify(self, session_id: str):
        """Cleanup session data using core AgentService logic."""
        logger.info(f"Removing Dify session: {session_id}")
        await self.agent_service.remove_session(session_id)