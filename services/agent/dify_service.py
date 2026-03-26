from datetime import datetime, timezone
from loguru import logger
from fastapi import BackgroundTasks

from .orchestrator import ChatOrchestrator
from .chat_service import ChatService
from services.memory import MemoryService
from core.exceptions import InternalServerError
from services.search.result import TxtBaseSearchResult


class DifyService:
    """Retrieval service adapter for Dify interface."""
    
    def __init__(self):
        # Initialize with a fixed user_id for Dify-sourced requests
        self.orchestrator = ChatOrchestrator()
        self.security_level = 9  # Level 9 bypasses security checks
        self.user_id="dify_system"
        self.chat_service = ChatService()
        self.memory_service = MemoryService()


    async def search(
        self,
        agent_id: int,
        question: str,
        session_id: str,
        background_tasks: BackgroundTasks,
        tags: list | None = None
    ) -> dict:
        """
        Agent interaction for Dify (Search + Record Persistence).
        """
        # Validate and normalize question parameter
        if not isinstance(question, str):
            logger.warning(f"Question is not a string, got type: {type(question).__name__}, converting to string")
            question = str(question)

        request_time = datetime.now(tz=timezone.utc)
        logger.info(f"Processing Dify request for session {session_id}, Agent {agent_id}")

        request_time = datetime.now(tz=timezone.utc)
        
        pipe_out = await self.orchestrator.run_pipeline(
            self.user_id, session_id, agent_id, question, self.security_level, tags
        )
        
        enriched_refs = await self.chat_service._enrich_results_with_metadata(pipe_out['kb_results'])
        records = self._build_dify_records(enriched_refs)

        # 持久化：使用 MemoryService 的新闭环方法
        background_tasks.add_task(
                self.memory_service.finalize_and_persist,
                session_id=session_id,
                user_id=self.user_id,
                raw_question=question,
                answer="",
                prepared_data=pipe_out['prepared_data'],
                retrieved_chunks=enriched_refs,
                request_time=request_time
            )

        return {"records": records}
        

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
