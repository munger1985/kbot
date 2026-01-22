import json
from loguru import logger
from mcp_tools import KBSearchResult, KBSearchToolParams
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from core.dictionary import KbCategory, KBSearchType
from utils.oracle_vec_handler import OracleVecHandler
from utils.encoder import DecimalEncoder
from utils.model_client import CallModel
from utils.common import safe_read_content


class KBSearch:
    """知识库搜索类"""
    def __init__(self, tool_params: KBSearchToolParams):
        self.tool_params = tool_params

    async def search(self, 
                     vector_search_question: str, 
                     full_text_question: str, 
                     security: int, 
                     tags: list[str] = []
                    ) -> list[KBSearchResult] | None:
        """
        执行知识库搜索
        
        Args:
            vector_search_question (str): 语义搜索问题
            full_text_question (list[str]): 全文搜索问题
            security (int): 安全级别
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBSearchResult] | None: 搜索结果列表，搜索失败时返回None
        """
        

        # 1. 获取模型ID
        repo = KbotMdKbRepository()
        models = await repo.get_model_by_kbid(self.tool_params.tool_id)
    
        if models:
            self.tool_params.kb_catogory = models[0]
            logger.debug(f"知识库类别: {models[0]}")
            self.tool_params.img2txt_model = models[1]
            logger.debug(f"图像转文本模型: {models[1]}")
            self.tool_params.img_embed_model = models[2]
            logger.debug(f"图像嵌入模型: {models[2]}")
            self.tool_params.txt_embed_model = models[3]
            logger.debug(f"文本嵌入模型: {models[3]}")
        else:
            logger.warning(f"未找到知识库 {self.tool_params.tool_id} 的嵌入模型")
            return None
        
        # 2. 根据搜索类型决定搜索方法
        if self.tool_params.kb_catogory == KbCategory.KBOT.value:
            if self.tool_params.search_type == KBSearchType.VECTOR.value:
                logger.debug("搜索方法: 向量搜索")
                return await self.search_by_vector(vector_search_question, security, is_summary=False, tags=tags)
            elif self.tool_params.search_type == KBSearchType.FULLTEXT.value:
                logger.debug("搜索方法: 全文搜索")
                return await self.serch_by_full_text(full_text_question, security, tags=tags)
            elif self.tool_params.search_type == KBSearchType.SUMMARY.value:
                logger.debug("搜索方法: 摘要搜索")
                return await self.search_by_vector(vector_search_question, security, is_summary=True, tags=tags)
            elif self.tool_params.search_type == KBSearchType.GRAPH.value:
                logger.debug("搜索方法: 图谱搜索")
                return await self.search_by_graph("question", security)
            else:
                logger.warning(f"知识库 {self.tool_params.tool_id} 的搜索方法未实现")
                return None
        else:
            logger.warning(f"知识库 {self.tool_params.tool_id} 的搜索方法未实现")
            return None

    
    async def search_by_vector(self, 
                               question: str, 
                               security: int, 
                               is_summary: bool = False, 
                               tags: list[str] = []
                            ) -> list[KBSearchResult] | None:
        """
        向量搜索方法
        
        Args:
            question (str): 搜索问题
            security (int): 安全级别
            is_summary (bool, optional): 是否使用摘要搜索. 默认为False
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBSearchResult] | None: 搜索结果列表，搜索失败时返回None
        """
        # 调用嵌入服务
        model_id = self.tool_params.txt_embed_model
        logger.debug(f"启用语义检索，问题: {question}")

        if not model_id:
            logger.warning(f"未找到知识库 {self.tool_params.tool_id} 的嵌入模型")
            return None

        try:
            results = await CallModel().call_embedding_model(model_id, [question])
            if results is None:
                logger.error("嵌入服务未返回结果")
                return None
            kb_results = []
            for result in results:
                embedding = result.embedding
                kb_result = await self.get_similar_records(embedding, security, is_summary=is_summary, tags=tags)
                if kb_result is None or kb_result == []:
                    continue
                else:
                    kb_results.extend(kb_result)
            return kb_results

        except Exception as e:
            logger.error(f"嵌入服务错误: {str(e)}")
            return None

    async def get_similar_records(self, 
                                  query_vec: list[float], 
                                  security: int, 
                                  is_summary: bool = False, 
                                  tags: list[str] = []
                                ) -> list[KBSearchResult] | None:
        """
        从向量数据库中获取相似记录
        
        Args:
            query_vec (list[float]): 查询向量
            security (int): 安全级别
            is_summary (bool, optional): 是否使用摘要搜索. 默认为False
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBSearchResult] | None: 相似记录列表，查询失败时返回None
        """
        # 执行相似度搜索
        repo = await EmbeddingRepositoryFactory.create_repository(kb_id=self.tool_params.tool_id)
        if repo is None:
            logger.error(f"向量搜索知识库ID: {self.tool_params.tool_id} 的向量数据库未找到")
            return None
        
        convertor = OracleVecHandler()
        vec = convertor.convert(query_vec, to_string=True)
        try:
            logger.debug(f"向量搜索知识库ID: {self.tool_params.tool_id}")
            logger.debug(f"向量搜索安全级别: {security}")
            logger.debug(f"向量搜索相似度阈值: {self.tool_params.threshold}")
            logger.debug(f"向量搜索返回数量: {self.tool_params.search_top_k}")

            dataset = await repo.get_similar_embeddings(
                kb_id = self.tool_params.tool_id,
                query_vec = vec,  # type: ignore
                security = security,
                similarity_threshold = self.tool_params.threshold,
                search_top_k = self.tool_params.search_top_k,
                is_summary_search=is_summary,
                tags=tags
            )
            if not dataset:
                logger.info(f"向量搜索未找到结果")
                return None
            results = []

            for data in dataset:
                chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                result = KBSearchResult()
                result.file_id = data[0]

                # 处理 chunk_type: 确保转换为整数
                chunk_type_raw = chunk_meta.get("chunk_type", 1)
                if isinstance(chunk_type_raw, str):
                    try:
                        result.chunk_type = int(chunk_type_raw)
                    except (ValueError, TypeError):
                        result.chunk_type = 1
                else:
                    result.chunk_type = int(chunk_type_raw) if chunk_type_raw is not None else 1

                # 处理 page_num: 如果是 None 或无效值则使用默认值
                page_num_raw = chunk_meta.get("page_num", 1)
                if page_num_raw is None:
                    result.page_num = 1
                else:
                    try:
                        result.page_num = int(page_num_raw)
                    except (ValueError, TypeError):
                        result.page_num = 1

                result.chunk_file_path = chunk_meta.get("chunk_file_path", "")
                result.content = safe_read_content(data[1])
                result.similarity = data[3]
                result.weight = self.tool_params.tool_weight # type: ignore
                results.append(result)

            logger.debug(f"向量搜索找到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.debug(f"向量搜索失败: {str(e)}")
            raise ValueError(f"向量搜索失败: {str(e)}")
        
    async def serch_by_full_text(self, keywords: str, security: int, tags: list[str] = []) -> list[KBSearchResult] | None:
        """
        全文搜索方法
        
        Args:
            keywords (str): 关键词
            security (int): 安全级别
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBSearchResult] | None: 搜索结果列表，搜索失败时返回None
        """
        repo = await EmbeddingRepositoryFactory.create_repository(kb_id=self.tool_params.tool_id)
        if repo is None:
            logger.error(f"全文搜索知识库ID: {self.tool_params.tool_id} 的向量数据库未找到")
            return None

        try:
            logger.debug(f"全文搜索知识库ID: {self.tool_params.tool_id}")
            logger.debug(f"全文搜索关键词: {keywords}")
            logger.debug(f"全文搜索安全级别: {security}")
            logger.debug(f"全文搜索返回数量: {self.tool_params.search_top_k}")
            logger.debug(f"全文搜索相似度阈值: {self.tool_params.threshold}")
            logger.debug(f"全文搜索标签: {tags}")



            datasets = await repo.full_text_search(kb_id=self.tool_params.tool_id, 
                                                keyword=keywords, 
                                                security=security,
                                                similarity_threshold=self.tool_params.threshold,
                                                search_top_k=self.tool_params.search_top_k,
                                                tags=tags)
            if not datasets:
                logger.info(f"全文搜索未找到结果")
                return None
            else:
                results = []
                for data in datasets:
                    chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                    result = KBSearchResult()
                    result.file_id = data[0]

                    # 处理 chunk_type: 确保转换为整数
                    chunk_type_raw = chunk_meta.get("chunk_type", 1)
                    if isinstance(chunk_type_raw, str):
                        try:
                            result.chunk_type = int(chunk_type_raw)
                        except (ValueError, TypeError):
                            result.chunk_type = 1
                    else:
                        result.chunk_type = int(chunk_type_raw) if chunk_type_raw is not None else 1

                    # 处理 page_num: 如果是 None 或无效值则使用默认值
                    page_num_raw = chunk_meta.get("page_num", 1)
                    if page_num_raw is None:
                        result.page_num = 1
                    else:
                        try:
                            result.page_num = int(page_num_raw)
                        except (ValueError, TypeError):
                            result.page_num = 1

                    result.chunk_file_path = chunk_meta.get("chunk_file_path", "")
                    result.content = safe_read_content(data[1])
                    result.similarity = data[3]
                    result.weight = self.tool_params.tool_weight # type: ignore
                    results.append(result)
                logger.debug(f"全文搜索找到 {len(results)} 条结果")
                return results
            
        except Exception as e:
            logger.exception(f"全文搜索失败: {str(e)}")
            return None

    
    async def search_by_graph(self, question: str, security: int) -> list[KBSearchResult] | None:
        """图谱搜索方法（待实现）"""
        pass