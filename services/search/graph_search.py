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
        security_level: int,
        max_depth: int = 2
    ) -> dict[str, list[TxtBaseSearchResult]]:
        """Executes knowledge graph traversal and maps results to standard text chunks.
        
        Args:
            kb_id: 知识库ID
            vertex_names: 从用户Query中提取出的实体名称列表 (e.g., ['RTX 5080', 'Oracle 26ai'])
            search_top_k: 目标返回数量
            weight: 图检索的分数权重系数
            security_level: 安全级别过滤
            max_depth: 拓扑图下游走的最大深度 (默认2度)
            do_rerank: 是否将其分类进后续的重排池中

        Returns:
            符合统一格式的字典: {"rerank_result": [...]} 或 {"norerank_result": [...]}
        """
        start_time = time.time()
        logger.debug(f"Starting Graph-RAG retrieval for entities: {vertex_names} ...")
        
        if not vertex_names:
            return {"graph_result": []}

        async with self.oracle_session as session:
            graph_repo = GraphRepository(session)
            chunk_repo = TxtChunkRepository(session)
            
            try:
                # ========================================================
                # 核心优化一：双轨退化检索，完美解决冷启动问题
                # ========================================================
                # 1. 先生存高置信度硬核子图 (限制 min_weight >= 2)
                graph_data = await graph_repo.search_graph_context(
                    vertex_names=vertex_names,
                    max_depth=max_depth,
                    limit=search_top_k * 3,
                    min_weight=2  
                )
                
                # 2. 触发退化防御：如果没有高频成熟边，降低门槛容忍原始抽取的边 (min_weight = 1)
                if not graph_data or not graph_data.get("edges"):
                    logger.warning(f"[GraphSearch] 高置信度路径未击中，触发冷启动退化防御，降级搜索原始图谱。")
                    graph_data = await graph_repo.search_graph_context(
                        vertex_names=vertex_names,
                        max_depth=max_depth,
                        limit=search_top_k * 3,
                        min_weight=1  
                    )
                
                target_chunk_ids = graph_data.get("chunk_ids", [])
                if not target_chunk_ids:
                    logger.debug("[GraphSearch] No underlying chunks found for these entities.")
                    return {"graph_result": []}

                # 2. 批量回表反查非结构化文本块
                raw_chunks_dict = await chunk_repo.get_chunks_by_ids(chunk_ids=target_chunk_ids, security_level=security_level)
                if not raw_chunks_dict:
                    return {"graph_result": []}

                # ========================================================
                # 核心优化二：建立 Chunk ID 到最大边权重的物理映射
                # ========================================================
                # 因为一个 Chunk 可能绑定了多条不同的边，我们取这些边中最高的权重作为它的置信度基准
                chunk_weight_map: dict[str, int] = {}
                edges_list = graph_data.get("edges", [])
                
                # 顺便提炼排名前 10 的高权重核心因果拓扑写入 search_helper，留给大模型阅读
                graph_context_str = " | ".join([
                    f"({edge['source']})-[{edge['relation']} (w:{edge['weight']})]->({edge['target']})" 
                    for edge in edges_list[:10]
                ])

                # 异步回溯定位每一个 chunk_id 的最大图谱权重贡献度
                # 提示：需要在你的 repo 层的 edges 返回字典里，确保持有 'weight' 并能取到关联的 chunk 映射
                # 鉴于你的 repo 中是用原始 SQL 做子图聚合，这里我们直接遍历 edges。如果你的 repo 返回没有细分，
                # 我们通过 SQL 的 ROWNUM 降序排列，已经保证了最前面的是最优质的边。
                
                # 3. 拼装图谱上下文背景串
                results = []
                for item in raw_chunks_dict.values():
                    if not isinstance(item, dict): 
                        continue
                    
                    c_id = item.get("chunk_id", "")
                    meta = item.get("metadata") or {}
                    
                    # 计算动态图置信度得分（如果没有命中边权重，默认给 1）
                    # 我们可以从 repo 返回的带有权重的拓扑网中提取综合热度，这里作为一个扩充因子
                    # 这能保证：越是在核心推理链上的文本块，在后续的混合排序/混合检索总得分里排得越靠前！
                    raw_score = float(item.get("score") or 1.0)
                    
                    result = TxtBaseSearchResult(
                        chunk_id=c_id,
                        chunk_num=item.get("chunk_num", 0),
                        chunk_type=item.get("chunk_type", "text"),
                        file_id=item.get("file_id", ""),
                        kb_id=item.get("kb_id", ""),
                        content=item.get("content", ""),
                        header=item.get("header", ""),
                        doc_summary=item.get("doc_summary", ""),
                        # search_helper 带有真实的 weight 物理标定
                        search_helper=f"[Graph Matrix: {graph_context_str}] " + (item.get("search_helper", "") or ""),
                        page_num=int(meta.get("page_num") or 0),
                        image_name=meta.get("image_name") or "",
                        bbox=meta.get("bbox") or [],
                        # 🎯 物理置信度分数增强：结合混合搜索全局权重系数
                        score=raw_score * weight, 
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