import json
from loguru import logger
from ..chat.agent_params import KBResult
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from dao.repositories.kbot_md_agent_conf_repo import KbotMdAgentConfRepository
from utils.oracle_vec_handler import OracleVecHandler
from utils.decimal_encoder import DecimalEncoder
from utils.call_models import CallModel
from utils.common import safe_read_content


class KBSearch:
    """知识库搜索类"""
    def __init__(self, agent_id: int, kb_id: int):
        self.agent_id = agent_id
        self.kb_id = kb_id
        self.kb_catogory = 0
        self.img2txt_model = 0
        self.img_embed_model = 0
        self.txt_embed_model = 0
        

    async def search(self, 
                     questions: list[str], 
                     search_type: str,
                     security: int, 
                     tags: list[str] = []
                    ) -> list[KBResult]:
        """
        执行知识库搜索
        
        Args:
            question (list[str]): 搜索问题
            search_type (str): 搜索类型
            security (int): 安全级别
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBResult]: 搜索结果列表，搜索失败时返回[]
        """
        

        # 1. 获取模型ID
        repo = KbotMdKbRepository()
        models = await repo.get_model_by_kbid(self.kb_id)
    
        if models:
            self.kb_catogory = models[0]
            logger.debug(f"知识库类别: {models[0]}")
            self.img2txt_model = models[1]
            logger.debug(f"图像转文本模型: {models[1]}")
            self.img_embed_model = models[2]
            logger.debug(f"图像嵌入模型: {models[2]}")
            self.txt_embed_model = models[3]
            logger.debug(f"文本嵌入模型: {models[3]}")
        else:
            logger.warning(f"未找到知识库 {self.kb_id} 的嵌入模型")
            return []
        
        # 2. 根据搜索类型决定搜索方法
        if search_type == "vector":
            logger.debug("搜索方法: 向量搜索")
            return await self.search_by_vector(questions, security, is_summary=False, tags=tags)
        elif search_type == "fulltext":
            logger.debug("搜索方法: 全文搜索")
            return await self.serch_by_full_text(questions, security, tags=tags)
        elif search_type == "summary":
            logger.debug("搜索方法: 摘要搜索")
            return await self.search_by_vector(questions, security, is_summary=True, tags=tags)
        elif search_type == "hybrid":
            logger.debug("搜索方法: 混合搜索")
            return await self.search_by_hybrid(questions, security, tags=tags)
        else:
            logger.warning(f"知识库 {self.kb_id} 的搜索方法未实现")
            return []


    
    async def search_by_vector(self, 
                               question: list[str], 
                               security: int, 
                               is_summary: bool = False, 
                               tags: list[str] = []
                            ) -> list[KBResult]:
        """
        向量搜索方法
        
        Args:
            question (list[str]): 搜索问题列表
            security (int): 安全级别
            is_summary (bool, optional): 是否使用摘要搜索. 默认为False
            tags (list[str], optional): 标签列表. 默认为空列表
            
        Returns:
            list[KBResult]: 搜索结果列表，搜索失败时返回None
        """
        # 调用嵌入服务
        model_id = self.txt_embed_model
        if is_summary:
            logger.debug(f"启用摘要检索，问题: {question}")
        else:
            logger.debug(f"启用语义检索，问题: {question}")

        if not model_id:
            logger.warning(f"未找到知识库 {self.kb_id} 的嵌入模型")
            return []

        try:
            results = await CallModel().call_embedding_model(model_id, question)
            if results is None:
                logger.error("嵌入服务未返回结果")
                return []
            
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
            return []

    async def get_similar_records(self, 
                                  query_vec: list[float], 
                                  security: int, 
                                  is_summary: bool = False, 
                                  tags: list[str] = []
                                ) -> list[KBResult]:
        """
        从向量数据库中获取相似记录
        
        Args:
            query_vec (list[float]): 查询向量
            security (int): 安全级别
            is_summary (bool, optional): 是否使用摘要搜索. 默认为False
            tags (list[str]): 标签列表. 默认为空列表
            
        Returns:
            list[KBResult] | None: 相似记录列表，查询失败时返回None
        """
        # 执行相似度搜索
        repo = await EmbeddingRepositoryFactory.create_repository(kb_id=self.kb_id)
        if repo is None:
            logger.error(f"向量搜索知识库ID: {self.kb_id} 的向量数据库未找到")
            return []
        
        # 从配置表中获取相似度阈值，返回数量和工具权重
        threshold = 0.7
        search_top_k = 5
        weight = 0.1

        conf = await KbotMdAgentConfRepository().get_by_agnet_and_kb(self.agent_id, self.kb_id)
        if conf:
            threshold = conf.search_score_threshold or 0.7
            search_top_k = conf.search_topk or 5
            weight = conf.tool_weight or 0.1
        
        convertor = OracleVecHandler()
        vec = convertor.convert(query_vec, to_string=True)
        try:
            logger.debug(f"向量搜索知识库ID: {self.kb_id}")
            logger.debug(f"向量搜索安全级别: {security}")
            logger.debug(f"向量搜索相似度阈值: {threshold}")
            logger.debug(f"向量搜索返回数量: {search_top_k}")

            dataset = await repo.get_similar_embeddings(
                kb_id = self.kb_id,
                query_vec = vec,  # type: ignore
                security = security,
                similarity_threshold = threshold,
                search_top_k = search_top_k,
                is_summary_search=is_summary,
                tags=tags
            )
            if not dataset:
                logger.info(f"向量搜索未找到结果")
                return []
            results = []

            for data in dataset:
                chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                result = KBResult()
                result.file_id = data[0]
                result.chunk_type = chunk_meta.get("chunk_type", 1)
                result.page_num = chunk_meta.get("page_num", 1)
                result.content = safe_read_content(data[1])
                result.similarity = data[3]
                result.weight = weight
                results.append(result)

            logger.debug(f"向量搜索找到 {len(results)} 条结果")
            return results

        except Exception as e:
            logger.debug(f"向量搜索失败: {str(e)}")
            raise ValueError(f"向量搜索失败: {str(e)}")
        
    async def serch_by_full_text(self, question: list[str], security: int, tags: list[str] = []) -> list[KBResult]:
        """
        全文搜索方法
        
        Args:
            question (list[str]): 搜索问题列表
            security (int): 安全级别
            tags (list[str]): 标签列表. 默认为空列表
            
        Returns:
            list[KBResult] | None: 搜索结果列表，搜索失败时返回None
        """
        repo = await EmbeddingRepositoryFactory.create_repository(kb_id=self.kb_id)
        if repo is None:
            logger.error(f"全文搜索知识库ID: {self.kb_id} 的向量数据库未找到")
            return []

        # 从配置表中获取相似度阈值，返回数量和工具权重
        threshold = 0.7
        search_top_k = 5
        weight = 0.1

        conf = await KbotMdAgentConfRepository().get_by_agnet_and_kb(self.agent_id, self.kb_id)
        if conf:
            weight = conf.tool_weight or 0.1
        
        # 执行全文搜索
        try:
            logger.debug(f"启用全文检索，问题: {question}")
            datasets = []
            unique_keys = set(question)  # 去除重复的关键字

            for key in unique_keys:
                logger.debug(f"全文搜索词元: {key}")

                ds = await repo.full_text_search(kb_id=self.kb_id, 
                                                 keyword=key, 
                                                 security=security,
                                                 tags=tags)
                if ds:
                    datasets.extend(ds)

            if not datasets:
                logger.info(f"全文搜索未找到结果")
                return []
            else:
                results = []
                for data in datasets:
                    chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                    result = KBResult()
                    result.file_id = data[0]
                    result.chunk_type = chunk_meta.get("chunk_type", 1)
                    result.page_num = chunk_meta.get("page_num", 1)
                    result.content = safe_read_content(data[1])
                    result.similarity = data[3]
                    result.weight = weight
                    results.append(result)
                logger.debug(f"全文搜索找到 {len(results)} 条结果")
                return results
            
        except Exception as e:
            logger.exception(f"全文搜索失败: {str(e)}")
            return []

    
    async def search_by_hybrid(self, question: list[str], security: int, tags: list[str]) -> list[KBResult]:
        """混合搜索方式"""
        vector_search_result = await self.search_by_vector(question=question, security=security, is_summary=False, tags=tags)
        fulltext_search_result = await self.serch_by_full_text(question=question, security=security, tags=tags)
        summary_search_result = await self.search_by_vector(question=question, security=security, is_summary=True, tags=tags)
        total_result = vector_search_result + fulltext_search_result + summary_search_result
        return total_result