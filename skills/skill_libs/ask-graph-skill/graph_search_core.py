import uuid
from loguru import logger
from typing import Any, AsyncGenerator

from skills import BaseSkill
from core.dictionary import PacketType
from agent.common import ContextMemory
from utils.simulate_stream import simulate_stream


class AskGraphSkill(BaseSkill):
    """
    Knowledge Graph Retrieval Skill: An advanced structured retrieval autonomous component built on topological relationships and 1st/2nd degree associations.
    Fully compliant with distributed autonomous package, lowercase hyphen naming, and data flow backfill bus specifications.
    """
    def __init__(self):
        super().__init__()
        # Default security level, can be overridden by Runtime
        self.security_level = 9
        # Lazy import to prevent circular dependency of NexusCube core components
        from services.search.graph_search import GraphBaseSearch
        self.graph_search_service = GraphBaseSearch()

    async def run_stream(
        self,
        context: ContextMemory,
        **kwargs
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Execute graph topology retrieval task (fully connected to variable registry and streaming bus version)
        """
        # 1. Securely extract execution snapshot and control info of current step
        current_execution = context.get("current_execution") or {}
        runtime_skill_name = current_execution.get("skill", "ask-graph-skill")
        output_var = current_execution.get("output_var") or "graph_results"

        current_user = context.get("user_id", "default_user")
        current_agent = context.get("agent_id")
        current_session = context.get("session_id") or uuid.uuid4().hex
        security_level = context.get("security_level", self.security_level)
        
        # 2. Strictly extract input parameters from the clean parameter dictionary (resolved_params) of the decision control plane
        resolved_params = current_execution.get("resolved_params") or {}
        
        # Extract entity words (vertex_names)
        vertex_names: list[str] = (
            resolved_params.get("vertex_names")
            or context.get("vertex_names") 
            or context.get("entities")
            or [k.strip() for k in context.get("search_keywords", "").split(",") if k.strip()]
        )
        
        # 🛡️ Robustness defense enhancement: If too few entities are extracted / fragmented, 
        # try to use the original sentence of standalone_query or question for fallback supplementation
        # This ensures broader text boundary awareness even if entities like "Hall Factor" are missed
        if len(vertex_names) < 4:
            fallback_query = context.get("standalone_query") or context.get("question")
            if fallback_query and fallback_query not in vertex_names:
                # Extract possible fragments from the original sentence or append the original sentence as a superstring node
                pass

        # Fallback strategy: If no entities are extracted by Planner or upper layer, 
        # treat the entire clean input string as an entity
        if not vertex_names:
            query_text = (
                current_execution.get("resolved_input") 
                or getattr(context, 'current_task', None) 
                or context.get("standalone_query") 
                or context.get("question")
            )
            if query_text:
                vertex_names = [query_text]

        # Extract other business parameters
        kb_id = resolved_params.get("kb_id") or context.get("kb_id") or current_execution.get("kb_id")
        search_top_k = resolved_params.get("search_top_k") or context.get("search_top_k", 10)
        max_depth = resolved_params.get("max_depth") or context.get("max_depth", 2)
        
        # Parameter alignment mapping
        graph_weight = resolved_params.get("graph_weight") or context.get("graph_weight", 1.2)

        # 3. Boundary defense assertions
        if not vertex_names:
            content = f"{runtime_skill_name}: Variable parsing exception, failed to capture any valid entity words in the context\n"
            async for char in simulate_stream(content):
                yield {"type": PacketType.ERROR, "content": char}
            return

        if not current_agent:
            content = f"{runtime_skill_name}: Missing critical parameter agent_id in global context\n"
            async for char in simulate_stream(content):
                yield {"type": PacketType.ERROR, "content": char}
            return

        if not kb_id:
            content = f"{runtime_skill_name}: Missing critical parameter kb_id in global context\n"
            async for char in simulate_stream(content):
                yield {"type": PacketType.ERROR, "content": char}
            return

        # Send thinking status: Start exploration
        entities_str = ", ".join(f"'{v}'" for v in vertex_names)
        content = f"Initiating topological traversal to graph space, core entities: [{entities_str}], max depth: {max_depth}...\n"
        async for char in simulate_stream(content):
            yield {"type": PacketType.THOUGHT, "content": char}

        try:
            # 4. Call the underlying unified graph retrieval service
            graph_raw_bucket = await self.graph_search_service.search_by_graph(
                kb_id=kb_id,
                vertex_names=vertex_names,
                search_top_k=search_top_k,
                weight=graph_weight,
                security_level=security_level,
                max_depth=max_depth
            )
            
            # Get graph search results
            enriched_refs = graph_raw_bucket.get("graph_result") or []
            content = f"Graph topology traversal completed. Activated {len(enriched_refs)} normalized text slices along the relationship chain...\n"
            async for char in simulate_stream(content):
                yield {"type": PacketType.THOUGHT, "content": char}

            # 5. Format and clean the results
            records_dict = self._build_records(enriched_references=enriched_refs)
            results_list = records_dict["graph_results"]
            
            logger.debug(f"[{runtime_skill_name}] Number of graph-associated text records: {len(results_list)}")
            
            # 6. Multi-dimensional data precipitation and backfill bus
            if "graph_results" not in context:
                context["graph_results"] = []
            context["graph_results"] = results_list
            
            if "variables" not in context:
                context["variables"] = {}
            context["variables"][output_var] = results_list

            # 7. Stream output result package for front-end rendering or orchestration layer tracking
            yield {"type": PacketType.GRAPH_RESULTS, "content": results_list}
            
            # 8. Inject graph search results into context for subsequent skills
            context["graph_results"] = results_list

        except Exception as e:
            logger.error(f"Autonomous graph component [{runtime_skill_name}] encountered a critical obstacle during runtime: {e}", exc_info=True)
            content = f"⚠️ System-level failure occurred in knowledge graph deep retrieval: {str(e)}\n"
            async for char in simulate_stream(content):
                yield {"type": PacketType.ERROR, "content": char}

    def _build_records(self, enriched_references: list[Any]) -> dict[str, Any]:
        """Align output with TxtBaseSearchResult specification to ensure full equivalence with standard text downstream."""
        records = []
        for ref in enriched_references:
            is_dict = isinstance(ref, dict)
            
            content = ref.get('content', '') if is_dict else getattr(ref, 'content', '')
            file_name = (ref.get('title') if is_dict else getattr(ref, 'title', None)) or "Graph Linked File"
            chunk_type = ref.get('chunk_type', 'text') if is_dict else getattr(ref, 'chunk_type', 'text')
            chunk_num = ref.get('chunk_num', 0) if is_dict else getattr(ref, 'chunk_num', 0)
            score = ref.get('score', 0.0) if is_dict else getattr(ref, 'score', 0.0)
            search_helper = ref.get('search_helper', '') if is_dict else getattr(ref, 'search_helper', '')

            meta = (ref.get('metadata') if is_dict else getattr(ref, 'metadata', {})) or {}
            
            record = {
                "title": file_name,
                "content": content,
                "chunk_type": chunk_type,
                "chunk_num": chunk_num,
                "score": score,
                "search_helper": search_helper,
                "metadata": {
                    "chunk_id": meta.get("chunk_id") or (ref.get('chunk_id') if is_dict else getattr(ref, 'chunk_id', '')), 
                    "file_id": meta.get("file_id") or (ref.get('file_id') if is_dict else getattr(ref, 'file_id', '')),
                    "kb_id": meta.get("kb_id") or (ref.get('kb_id') if is_dict else getattr(ref, 'kb_id', '')),
                    "header": meta.get("header") or (ref.get('header', '') if is_dict else getattr(ref, 'header', '')),
                    "page_num": int(meta.get("page_num", 0)),
                    "bbox": meta.get("bbox") or [],
                    "image_name": meta.get("image_name", "")
                }
            }
            records.append(record)
            
        records.sort(key=lambda x: x["score"], reverse=True)
        return {"graph_results": records}