import time
from loguru import logger

from core.exceptions import handle_exception, DataNotFoundException
from core.database.oracle import get_session
from dao.repositories import GraphRepository, TxtChunkRepository
from .result import TxtBaseSearchResult
from .kb_search import TxtBaseSearch


class GraphBaseSearch:
    """Graph-based knowledge base search service.
    
    Retrieves subgraphs linked to extracted entities and normalizes the 
    underlying source chunks into standard TxtBaseSearchResult objects.
    """

    @property
    def oracle_session(self):
        """Returns a database session context manager."""
        return get_session()

    async def search_by_graph(
        self,
        kb_id: int,
        vertex_names: list[str],
        search_top_k: int,
        weight: float,
        security: int,
        max_depth: int = 2,
        do_rerank: bool = True
    ) -> dict[str, list[TxtBaseSearchResult]]:
        """Executes knowledge graph traversal and maps results to standard text chunks.
        
        Args:
            kb_id: 知识库ID
            vertex_names: 从用户Query中提取出的实体名称列表 (e.g., ['RTX 5080', 'Oracle 26ai'])
            search_top_k: 目标返回数量
            weight: 图检索的分数权重系数
            security: 安全级别过滤
            max_depth: 拓扑图下游走的最大深度 (默认2度)
            do_rerank: 是否将其分类进后续的重排池中

        Returns:
            符合统一格式的字典: {"rerank_result": [...]} 或 {"norerank_result": [...]}
        """
        start_time = time.time()
        logger.debug(f"Starting Graph-RAG retrieval for entities: {vertex_names} ...")
        
        if not vertex_names:
            return {"rerank_result": [], "norerank_result": []}

        async with self.oracle_session as session:
            graph_repo = GraphRepository(session)
            chunk_repo = TxtChunkRepository(session)
            
            try:
                # 1. 探索局部图拓扑结构，拿回 1~2 度关联所有的物理 chunk_id 以及边的描述
                # 这里内部执行的就是我们上一轮写的 text() 原生 SQL
                graph_data = await graph_repo.search_graph_context(
                    vertex_names=vertex_names,
                    max_depth=max_depth,
                    limit=search_top_k * 3  # 过量召回，为后续融合和去重留出空间
                )
                
                target_chunk_ids = graph_data.get("chunk_ids", [])
                if not target_chunk_ids:
                    logger.debug("[GraphSearch] No underlying chunks found for these entities.")
                    return {"rerank_result": [], "norerank_result": []}

                # 2. 核心转换：利用捞出来的 chunk_ids 批量回表反查非结构化文本块
                # 提示：如果你 TxtChunkRepository 里没有 get_chunks_by_ids，可以手写一个 IN 查询
                raw_chunks = await chunk_repo.get_chunks_by_ids(chunk_ids=target_chunk_ids)
                
                if not raw_chunks:
                    return {"rerank_result": [], "norerank_result": []}

                # 3. 将原始的 dict 列表转化为标准的 TxtBaseSearchResult
                # 并在转换过程中，把图检索带来的“知识边关系”注入到 search_helper 字段中，提供给 Prompt 增益
                results = []
                # 拼装一个给大模型看的图谱上下文背景串
                graph_context_str = " | ".join([
                    f"({edge['source']})-[{edge['relation']}]->({edge['target']})" 
                    for edge in graph_data.get("edges", [])[:10]  # 最多塞10条高频关系
                ])

                for item in raw_chunks:
                    if not isinstance(item, dict): 
                        continue
                    
                    meta = item.get("metadata") or {}
                    
                    # 基础映射，字段与你的 _construct_search_result 严格对齐
                    result = TxtBaseSearchResult(
                        chunk_id=item.get("chunk_id", ""),
                        chunk_num=item.get("chunk_num", 0),
                        chunk_type=item.get("chunk_type", "text"),
                        file_id=item.get("file_id", ""),
                        kb_id=item.get("kb_id", ""),
                        content=item.get("content", ""),
                        header=item.get("header", ""),
                        doc_summary=item.get("doc_summary", ""),
                        search_helper=f"[Graph Matrix: {graph_context_str}] " + (item.get("search_helper", "") or ""),
                        page_num=int(meta.get("page_num") or 0),
                        image_name=meta.get("image_name") or "",
                        bbox=meta.get("bbox") or [],
                        # 赋予基础初始图谱分值（可根据业务通过 weight 调整）
                        score=float(item.get("score") or 1.0) * weight,
                        biz_metadata=item.get("biz_metadata") or {},
                        weight=weight,
                        rerank_score=0.0
                    )
                    results.append(result)

                tb_search = TxtBaseSearch()
                search_result = await tb_search._enhance_context_with_window(results)
                
                duration = time.time() - start_time
                logger.info(f"Graph-RAG retrieval finished in {duration:.2f}s. Formatted {len(search_result)} chunks.")
                
                return {"graph_result": search_result}

            except DataNotFoundException:
                return {"graph_result": []}
            except Exception as e:
                handle_exception(e, f"Graph-RAG search failed for KB {kb_id}: {str(e)}")