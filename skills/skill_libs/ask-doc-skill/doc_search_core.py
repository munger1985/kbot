import uuid
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from agent.memory import MemoryService
from core.dictionary import PacketType
from agent.common import ContextMemory


class AskDocSkill(BaseSkill):
    """
    Document retrieval skill: Fully compliant with distributed autonomous package, lowercase hyphen naming, and data flow backfill bus specifications.
    """
    def __init__(self):
        super().__init__()
        self.security_level = 9
        # Lazy import to avoid circular dependency
        from agent.agent import DocAgent
        self.doc_agent = DocAgent()
        self.memory_service = MemoryService()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Execute document retrieval task (high robustness streaming bus version)
        """
        # 1. Extract current execution snapshot (avoid system-level Key crashes caused by hardcoded retrieval)
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-doc-skill")

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        
        # 2. Intelligent and high-confidence retrieval of clean query text after variable replacement by the base Runtime
        query_text = (
            current_execution.get("resolved_input") 
            or getattr(context, 'current_task', None) 
            or context.get("standalone_query") 
            or context.get("question")
        )
        
        search_keywords = (
            current_execution.get("search_keywords")
            or current_execution.get("keywords")
            or context.get("search_keywords")
            or context.get("keywords")
            or ""
        )
        logger.debug(f"[{runtime_skill_name}] 接收到的 search_keywords: '{search_keywords}'")
        
        tags = context.get("tags") or []
        
        if not query_text:
            content = f"{runtime_skill_name}: Variable parsing exception, failed to obtain any valid search text.\n"
            yield {"type": PacketType.ERROR, "content": content}
            return

        if not current_agent:
            content = f"{runtime_skill_name}: Missing critical parameter agent_id in global context.\n"
            yield {"type": PacketType.ERROR, "content": content}
            return

        # Push thinking status: Start retrieving documents
        content = f"Retrieving knowledge base documents, query: '{query_text}'...\n"
        yield {"type": PacketType.THOUGHT, "content": content}

        try:
            # 4. Execute underlying RAG retrieval
            enriched_refs = await self.doc_agent.rag_retrieval(
                session_id=current_session,
                agent_id=current_agent,
                question=context.get("question", query_text),
                standalone_query=query_text,
                search_keywords=search_keywords,
                security_level=self.security_level,
                user_id=current_user,
                tags=tags
            )
            
            # Push thinking status: Retrieval completed, rearranging documents
            content = f"Found {len(enriched_refs)} related document slices in the knowledge base, performing correlation reorganization...\n"
            yield {"type": PacketType.THOUGHT, "content": content}

            # 5. Format and clean the results
            records_dict = self._build_records(enriched_references=enriched_refs)
            logger.debug(f"[{runtime_skill_name}] Number of formatted high-quality document records: {len(records_dict['doc_results'])}")
            
            # 6. Push the result package required for final front-end rendering or orchestration layer tracking
            yield {"type": PacketType.DOC_RESULTS, "content": records_dict["doc_results"]}
            logger.debug(f"[{runtime_skill_name}] Document retrieval results pushed to the bus.")
            
            # 7. Store the formatted results in the context and deliver them to the Runtime bus.
            context["doc_results"] = records_dict["doc_results"]

        except Exception as e:
            logger.error(f"Autonomous component [{runtime_skill_name}] encountered a critical obstacle during runtime: {e}", exc_info=True)
            content = f"⚠️ System-level failure occurred in document retrieval: {str(e)}\n"
            yield {"type": PacketType.ERROR, "content": content}

    def _build_records(self, enriched_references: list[dict]) -> dict[str, Any]:
        """
        Format the output according to the TxtBaseSearchResult definition.
        Ensure the downstream ReasoningSkill can obtain complete metadata for traceability.
        """
        records = []
        for ref in enriched_references:
            content = ref.get("content", "")
            
            record = {
                # Basic display information
                "title": ref.get("file_name", "Unknown File"),
                "content": content,
                "chunk_type": ref.get("chunk_type", "text"),
                "chunk_num": ref.get("chunk_num", 0),

                # Scoring system (prioritize cross-dimensional rerank score, then original vector distance score)
                "score": ref.get("rerank_score") if ref.get("rerank_score", 0) > 0 else ref.get("score", 0),
                
                # Extended metadata (for front-end rendering highlighting or click-to-locate navigation)
                "metadata": {
                    "chunk_id": ref.get("chunk_id") or ref.get("id"), 
                    "file_id": ref.get("file_id"),
                    "kb_id": ref.get("kb_id"),
                    "header": ref.get("header", ""),
                    "page_num": ref.get("page_num", 0),
                    "bbox": ref.get("bbox", []),
                    "image_name": ref.get("image_name", "")
                },
                "biz_metadata": ref.get("biz_metadata", {})
            }
            records.append(record)
            
        # Sort by score from high to low to ensure the reasoning layer sees the most relevant fragments first
        records.sort(key=lambda x: x["score"], reverse=True)
        return {"doc_results": records}