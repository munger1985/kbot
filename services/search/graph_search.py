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
        """执行知识图谱深度空间网络下游走，并将拓扑关系无缝映射回标准的非结构化文本块。"""
        start_time = time.time()
        logger.debug(f"开始为实体驱动图谱检索 (Graph-RAG): {vertex_names} ...")
        
        if not vertex_names:
            return {"graph_result": []}

        async with self.oracle_session as session:
            graph_repo = GraphRepository(session)
            chunk_repo = TxtChunkRepository(session)
            
            try:
                # ========================================================
                # 1. 双轨退化检索，完美解决冷启动问题
                # ========================================================
                # 轨道 A：优先追溯高置信度高频核心子图 (min_weight >= 2)
                graph_data = await graph_repo.search_graph_context(
                    kb_id=kb_id,
                    vertex_names=vertex_names,
                    max_depth=max_depth,
                    limit=search_top_k * 3,
                    min_weight=2  
                )
                
                # 轨道 B：触发退化防御：如果没有高频成熟边，降低门槛容忍原始抽取的轻量边 (min_weight = 1)
                if not graph_data or not graph_data.get("edges"):
                    logger.warning(f"[GraphSearch] 高置信度路径未击中，触发冷启动退化防御，降级搜索原始图谱。")
                    graph_data = await graph_repo.search_graph_context(
                        kb_id=kb_id,
                        vertex_names=vertex_names,
                        max_depth=max_depth,
                        limit=search_top_k * 3,
                        min_weight=1  
                    )
                
                if not graph_data:
                    return {"graph_result": []}

                target_chunk_ids = graph_data.get("chunk_ids", [])
                if not target_chunk_ids:
                    logger.debug("[GraphSearch] 未能捕获到该实体网络链条下的任何底层关联文本块(chunk_ids).")
                    return {"graph_result": []}

                logger.debug(
                    f"[GraphSearch] 准备批量回表反查 (KB_ID {kb_id}), "
                    f"target_chunk_ids 数量: {len(target_chunk_ids)}, "
                    f"前5个: {target_chunk_ids[:5]!r}"
                )

                # 2. 批量回表反查非结构化文本块明细
                try:
                    raw_chunks_dict = await chunk_repo.get_chunks_by_ids(chunk_ids=target_chunk_ids, security_level=security_level)
                    logger.debug(
                        f"[GraphSearch] get_chunks_by_ids 返回 (KB_ID {kb_id}): "
                        f"返回 {len(raw_chunks_dict)} 条, "
                        f"raw_chunks_dict 类型: {type(raw_chunks_dict).__name__}"
                    )
                except Exception as chunk_fetch_err:
                    logger.error(
                        f"[GraphSearch] get_chunks_by_ids 调用失败 (KB_ID {kb_id}): "
                        f"错误类型: {type(chunk_fetch_err).__name__}, 错误: {chunk_fetch_err}",
                        exc_info=True
                    )
                    raise
                if not raw_chunks_dict:
                    return {"graph_result": []}

                # ========================================================
                # 3. 提炼核心拓扑关系链条（增强防御，防止字典键名带引号或缺失）
                # ========================================================
                edges_list = graph_data.get("edges", []) or []
                cleaned_edges = []
                for edge in edges_list[:10]:
                    # 鲁棒兼容：无论底层返回的是类对象还是字典，安全提取
                    src = edge.get("source") if isinstance(edge, dict) else getattr(edge, "source", None)
                    rel = edge.get("relation") if isinstance(edge, dict) else getattr(edge, "relation", None)
                    tgt = edge.get("target") if isinstance(edge, dict) else getattr(edge, "target", None)
                    wgt = edge.get("weight") if isinstance(edge, dict) else getattr(edge, "weight", 1)
                    if src and tgt:
                        cleaned_edges.append(f"({src})-[{rel} (w:{wgt})]->({tgt})")

                graph_context_str = " | ".join(cleaned_edges) if cleaned_edges else "无显式拓扑连通路径"

                # ========================================================
                # 4. 拼装结构化图谱上下文背景列表
                # ========================================================
                results = []
                logger.debug(
                    f"[GraphSearch] 开始映射 {len(raw_chunks_dict)} 条 chunk 至标准结果集 (KB_ID {kb_id})，"
                    f"graph_context_str={graph_context_str[:200] if graph_context_str else 'N/A'}"
                )
                for idx, (chunk_key, item) in enumerate(raw_chunks_dict.items()):
                    if not item: 
                        logger.debug(f"[GraphSearch] 跳过空 item (KB_ID {kb_id}, idx={idx}, chunk_key={chunk_key!r})")
                        continue

                    try:
                        # 🔍 诊断日志：记录 item 的类型和所有键名，用于定位 KeyError 根因
                        if isinstance(item, dict):
                            item_keys = list(item.keys())
                            logger.debug(
                                f"[GraphSearch] 处理第 {idx} 条 item (KB_ID {kb_id}), "
                                f"type=dict, keys={item_keys!r}, chunk_key={chunk_key!r}"
                            )
                        else:
                            logger.debug(
                                f"[GraphSearch] 处理第 {idx} 条 item (KB_ID {kb_id}), "
                                f"type={type(item).__name__}, chunk_key={chunk_key!r}"
                            )

                        # 🛡️ 核心修复：对 item 的类型做彻底的解耦防御，严防 ORM 对象与字典导致的 Key 碰撞
                        if isinstance(item, dict):
                            c_id = item.get("chunk_id", "")
                            c_num = item.get("chunk_num", 0)
                            c_type = item.get("chunk_type", "text")
                            f_id = item.get("file_id", "")
                            # 核心防御：防止数据库返回的键名被单引号 'kb_id' 污染，没有则直接用传入的干净 kb_id
                            # 同时防御 item 的 kb_id 值为非整数字符串（如字面量 "'kb_id'"）导致 int() 崩溃
                            raw_kb_id = item.get("kb_id") or item.get("'kb_id'")
                            try:
                                item_kb_id = int(raw_kb_id) if raw_kb_id is not None else kb_id
                            except (ValueError, TypeError):
                                logger.warning(
                                    f"[GraphSearch] kb_id 值异常，无法转为整型，"
                                    f"原始值: {raw_kb_id!r}，回退使用参数 kb_id: {kb_id}"
                                )
                                item_kb_id = kb_id
                            c_content = item.get("content", "")
                            c_header = item.get("header", "")
                            c_summary = item.get("doc_summary", "")
                            c_helper = item.get("search_helper", "")
                            raw_score = float(item.get("score") or 1.0)
                            meta = item.get("metadata") or {}
                            biz_meta = item.get("biz_metadata") or {}
                        else:
                            # 兼容处理 SQLAlchemy ORM Entity 实体属性模式
                            c_id = getattr(item, "chunk_id", "")
                            c_num = getattr(item, "chunk_num", 0)
                            c_type = getattr(item, "chunk_type", "text")
                            f_id = getattr(item, "file_id", "")
                            item_kb_id = getattr(item, "kb_id", None) or kb_id
                            c_content = getattr(item, "content", "")
                            c_header = getattr(item, "header", "")
                            c_summary = getattr(item, "doc_summary", "")
                            c_helper = getattr(item, "search_helper", "")
                            raw_score = float(getattr(item, "score", 1.0) or 1.0)
                            meta = getattr(item, "metadata", {}) or {}
                            biz_meta = getattr(item, "biz_metadata", {}) or {}

                        # 🔍 诊断日志：记录提取后的关键字段值
                        logger.debug(
                            f"[GraphSearch] item 字段提取完成 (KB_ID {kb_id}, idx={idx}): "
                            f"c_id={c_id!r}, item_kb_id={item_kb_id!r}, f_id={f_id!r}, "
                            f"c_type={c_type!r}, biz_meta_type={type(biz_meta).__name__}"
                        )

                        # 格式化清洗 metadata 内部字段（防御非 dict 类型的 meta）
                        if not isinstance(meta, dict):
                            meta = {}
                        p_num = int(meta.get("page_num") or 0)
                        img_n = meta.get("image_name") or ""
                        bbox_list = meta.get("bbox") or []

                        # biz_metadata 防御：确保一定是 dict，拒绝一切非 dict 值（包括 str/None/ORM 代理对象）
                        if not isinstance(biz_meta, dict):
                            biz_meta = {}

                        # 实例化干净、高容错的标准结果集对象
                        logger.debug(
                            f"[GraphSearch] 即将构造 TxtBaseSearchResult (KB_ID {kb_id}, idx={idx}): "
                            f"kb_id={int(item_kb_id)}, file_id={str(f_id)!r}, chunk_id={str(c_id)!r}"
                        )
                        result = TxtBaseSearchResult(
                            chunk_id=str(c_id),
                            chunk_num=int(c_num),
                            chunk_type=str(c_type),
                            file_id=str(f_id),
                            kb_id=int(item_kb_id),  # 🎯 确保转换为纯净的 int，绝不携带任何字符串引号
                            content=str(c_content),
                            header=str(c_header),
                            doc_summary=str(c_summary),
                            search_helper=f"[Graph Matrix: {graph_context_str}] " + (c_helper or ""),
                            page_num=p_num,
                            image_name=img_n,
                            bbox=bbox_list if isinstance(bbox_list, list) else [],
                            score=raw_score * weight, # 乘上混合检索分配给图谱的图置信度权重系数
                            biz_metadata=biz_meta,
                            weight=float(weight),
                            rerank_score=0.0
                        )
                        results.append(result)
                    except Exception as item_err:
                        # 逐条防御：单条 item 构造失败不中断整体流程，打印 item keys 辅助排查
                        item_keys = list(item.keys()) if isinstance(item, dict) else type(item).__name__
                        logger.error(
                            f"[GraphSearch] 单条 chunk 结果构造失败 (KB_ID {kb_id})，"
                            f"item keys/type: {item_keys!r}，错误类型: {type(item_err).__name__}，错误: {item_err}",
                            exc_info=True
                        )
                        continue

                # 5. 调用上下文滑动窗口增强（复用原有标准 RAG 基础设施）
                logger.debug(
                    f"[GraphSearch] 即将调用 _enhance_context_with_window (KB_ID {kb_id}), "
                    f"results 数量: {len(results)}, 首条 chunk_id: {results[0].chunk_id if results else 'N/A'}"
                )
                tb_search = TxtBaseSearch()
                try:
                    search_result = await tb_search._enhance_context_with_window(results)
                    logger.debug(
                        f"[GraphSearch] _enhance_context_with_window 完成 (KB_ID {kb_id}), "
                        f"返回 {len(search_result)} 条结果"
                    )
                except Exception as enhance_err:
                    logger.error(
                        f"[GraphSearch] _enhance_context_with_window 崩溃 (KB_ID {kb_id}): "
                        f"错误类型: {type(enhance_err).__name__}, 错误: {enhance_err}",
                        exc_info=True
                    )
                    raise
                
                duration = time.time() - start_time
                logger.info(f"图谱空间分析检索(Graph-RAG)优雅结束，耗时: {duration:.2f}s. 成功向总线交付 {len(search_result)} 条标准文本块.")
                
                return {"graph_result": search_result}

            except DataNotFoundException:
                logger.warning(f"[GraphSearch] 关联知识库 {kb_id} 触发业务层未找到空数据信号。")
                return {"graph_result": []}
            except Exception as e:
                # 捕获崩溃并抛给统一控制平面异常处理器
                handle_exception(e, f"图谱混合检索在映射结果集生命周期内崩溃，KB_ID {kb_id}: {str(e)}")