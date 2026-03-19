import asyncio
import time
from loguru import logger
from core.exceptions import *
from core.dictionary import KbCategory, KBSearchType
from core.database.oracle import get_session
from dao.repositories import TxtChunkRepository
from .fulltext_preprocessor import preprocess_for_fulltext
from .result import TxtBaseSearchResult


class TxtBaseSearch:
    """文本知识库搜索类"""
    @property
    def oracle_session(self):
        return get_session()

    def rrf_merge(self, search_top_k: int, weight: float, results_list: list[list[TxtBaseSearchResult]]) -> list[TxtBaseSearchResult]:
        """定义内部融合逻辑 (Reciprocal Rank Fusion)"""
        k = 60
        score_map = {}
        chunk_map = {}
        
        for results in results_list:
            for rank, r in enumerate(results, 1):
                score_map[r.chunk_id] = score_map.get(r.chunk_id, 0) + (1.0 / (k + rank))
                chunk_map[r.chunk_id] = r
        
        merged = []
        for cid, rrf_score in score_map.items():
            res = chunk_map[cid]
            # 层级增强：标题类权重加成
            level_boost = 1.2 if (0 < res.structure_level < 3) else 1.0
            res.score = rrf_score * level_boost * weight
            merged.append(res)
        
        merged.sort(key=lambda x: x.score, reverse=True)
        safe_top_k = int(search_top_k) if search_top_k is not None else 10
        return merged[:safe_top_k]
    
    async def search(self,
                     kb_id: int,
                     question: str,
                     search_top_k: int,
                     threshold: float,
                     do_rerank: bool,
                     weight: float,
                     security: int, 
                     llm_model: str | None = None,
                     query_vec: list[float] | None = None,
                     tags: list[str] = []
                    ) -> dict[str, list[TxtBaseSearchResult]]:
        """
        混合分层搜索 (支持 Rerank 分组融合)
        """
        start_time = time.time()
        logger.debug(f"开始混合分层搜索，问题: {question}")

        logger.debug(f"search_top_k: {search_top_k}, threshold: {threshold}, do_rerank: {do_rerank}, weight: {weight}, security: {security}, llm_model: {llm_model}, tags: {tags}")

        # 执行检索任务
        if not query_vec:
            logger.warning("向量为空，只进行全文检索")
            fulltext_raw = await self.serch_by_full_text(kb_id, security, question, search_top_k, do_rerank, weight, llm_model, tags)
            vector_raw = {"rerank_result": [], "norerank_result": []}
        else:
            vector_raw, fulltext_raw = await asyncio.gather(
                self.search_by_vector(kb_id, security, query_vec, threshold, search_top_k, do_rerank, weight, tags),
                self.serch_by_full_text(kb_id, security, question, search_top_k, do_rerank, weight, llm_model, tags)
            )
        

        # 分别对 rerank 和 norerank 组进行融合
        logger.debug(f"Vector IDs: {[r.chunk_id for r in vector_raw.get('rerank_result', [])]}")
        logger.debug(f"Fulltext IDs: {[r.chunk_id for r in fulltext_raw.get('rerank_result', [])]}")

        final_rerank = self.rrf_merge(search_top_k=search_top_k, weight=weight, results_list=[
            vector_raw.get("rerank_result", []),
            fulltext_raw.get("rerank_result", [])
        ])
        
        final_norerank = self.rrf_merge(search_top_k=search_top_k, weight=weight, results_list=[
            vector_raw.get("norerank_result", []),
            fulltext_raw.get("norerank_result", [])
        ])

        end_time = time.time()
        logger.debug(f"混合搜索完成，Rerank组: {len(final_rerank)}, Non-Rerank组: {len(final_norerank)}，耗时 {end_time - start_time:.2f}s")

        return {
            "rerank_result": final_rerank,
            "norerank_result": final_norerank
        }
    
    async def search_by_vector(self, kb_id: int,
                               security: int, 
                               query_vec: list[float],
                               threshold: float,
                               search_top_k: int,
                               do_rerank: bool,
                               weight: float,
                               tags: list[str] = []) -> dict[str, list[TxtBaseSearchResult]]:
        """增强版：支持层级感知搜索"""
        logger.debug(f"启用向量检索")
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
            try:
                dataset = await repo.vector_search(
                    kb_id=kb_id,
                    query_vec=query_vec,
                    security=security,
                    similarity_threshold=threshold,
                    search_top_k=search_top_k,
                    tags=tags
                )
                # 构造向量搜索结果
                results = self._construct_search_result(dataset, weight=weight, search_type="semantic")
                logger.debug(f"向量搜索完成，找到 {len(results)} 条结果")
                search_result = await self._enhance_context_by_hierarchy(results)
                if do_rerank:
                    return {"rerank_result": search_result}
                else:
                    return {"norerank_result": search_result}
            
            except DataNotFoundException as e:
                logger.warning(e.message)
                return {"rerank_result": [], "norerank_result": []}
            except Exception as e:
                msg = f"知识库 {kb_id} 向量搜索失败: {e}"
                handle_exception(e, msg)
        
    async def serch_by_full_text(self, kb_id: int,
                                security: int, 
                                question: str,
                                search_top_k: int,
                                do_rerank: bool,
                                weight: float,
                                llm_model: str | None = None,
                                tags: list[str] = []) -> dict[str, list[TxtBaseSearchResult]]:
        """
        全文搜索方法
        
        Args:
            kb_id (int): 知识库ID
            security (int): 安全级别
            question (str): 搜索问题
            search_top_k (int): 搜索TopK
            do_rerank (bool): 是否进行rerank
            weight (float): 权重
            llm_model (str, optional): LLM模型名称，默认为None
            tags (list[str]): 标签列表. 默认为空列表
            
        Returns:
            dict[str, list[TxtBaseSearchResult]]: 搜索结果
        """
        # 获取向量库和搜索参数
        async with self.oracle_session as session:
            repo = TxtChunkRepository(session)
        
            try:
                # 1. 预处理问题，获取问题改写关键词和同义词
                key = await preprocess_for_fulltext(query=question, model_name=llm_model)
                
                # 2. 执行全文搜索
                logger.debug(f"启用全文检索，搜索关键字: {key}")
                dataset = await repo.full_text_search(
                                                    kb_id=kb_id,
                                                    keyword=key, 
                                                    security=security,
                                                    search_top_k=search_top_k,
                                                    tags=tags)
                # 3. 处理搜索结果
                results = self._construct_search_result(dataset, weight=weight, search_type="fulltext")
                logger.debug(f"全文搜索找到 {len(results)} 条结果")

                # 4. 分层检索
                search_result = await self._enhance_context_by_hierarchy(results)
                if do_rerank:
                    return {"rerank_result": search_result}
                else:
                    return {"norerank_result": search_result}
            
            except DataNotFoundException as e:
                logger.warning(e.message)
                return {"rerank_result": [], "norerank_result": []}
            except Exception as e:
                msg = f"知识库 {kb_id} 全文搜索失败: {e}"
                handle_exception(e, msg)
    
    async def _enhance_context_by_hierarchy(self, results: list[TxtBaseSearchResult]) -> list[TxtBaseSearchResult]:
        """
        分层检索核心：根据路径基因(path_names)增强上下文内容
        如果命中了 1.1.2 节的内容，自动将其父级标题关联进 content 字段，方便 LLM 理解背景
        """
        for r in results:
            if r.path_names:
                # 注入结构化路径基因
                path_str = " > ".join(r.path_names)
                prefix = f"[{path_str}]\n"
                if prefix not in r.content:
                    r.content = prefix + r.content
        return results
    
    def _construct_search_result(self, dataset: list, weight: float, search_type: str) -> list[TxtBaseSearchResult]:
        """
        构造搜索结果列表
        根据 Repository 返回格式精准对齐：
        { "chunk_id", "file_id", "content", "path", "structure_level", "metadata", "score" }
        """
        results = []
        for item in dataset:
            try:
                if not isinstance(item, dict):
                    logger.warning(f"跳过非字典格式的搜索结果: {type(item)}")
                    continue
                    
                # 1. 提取元数据字典 (对应 chunk.chunk_metadata)
                # 注意：Oracle 的数值在 metadata 里可能是 Decimal 类型
                chunk_meta = item.get("metadata") or {}
                if not isinstance(chunk_meta, dict):
                    chunk_meta = {}

                # 2. 处理路径逻辑 (对应 "path": " > ".join(path_list))
                # 优先从 path 字段解析，如果 path_names 已存在则保留
                path_names = item.get("path_names", [])
                if not path_names and "path" in item:
                    path_str = item.get("path", "")
                    path_names = [p.strip() for p in path_str.split(">") if p.strip()]

                # 3. 构造结果对象，增加对 Oracle Decimal 类型的强制转换 (int/float)
                result = TxtBaseSearchResult(
                    # 基本标识
                    chunk_id=str(item.get("chunk_id", "")),
                    file_id=item.get("file_id", ""),
                    content=item.get("content", ""),
                    
                    # 层级与路径
                    structure_level=int(item.get("structure_level", 0)),
                    node_path=chunk_meta.get("node_path", "") or "", # node_path 通常藏在 metadata 里
                    path_names=path_names or [],
                    
                    # 从 metadata 中提取业务字段 (处理 Decimal)
                    page_num=int(chunk_meta.get("page_num", 0)) if chunk_meta.get("page_num") else 0,
                    chunk_num=int(chunk_meta.get("chunk_num", 0)) if chunk_meta.get("chunk_num") else 0,
                    sub_index=int(chunk_meta.get("sub_index", 0)) if chunk_meta.get("sub_index") else 0,
                    chunk_type=chunk_meta.get("chunk_type", "text"),
                    
                    # 评分与向量 (处理 similarity_score 可能为 None 的情况)
                    score=float(item.get("score") or 0.0),
                    embedding=item.get("embedding", []) or [],
                    
                    # 初始扩展字段
                    rerank_score=0.0,
                    weight=weight,
                    search_type=search_type
                )
                
                results.append(result)
                
            except Exception as e:
                # 增加 item 的 ID 打印，方便定位具体哪条数据解析崩溃
                logger.warning(f"构造搜索结果失败 (ID: {item.get('chunk_id')}): {e}")
                continue
                
        return results