import json
from loguru import logger
from ..chat.agent_params import ToolParams, KBResult
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from core.dictionary import KbCategory, KBSearchType
from utils.oracle_vec_handler import OracleVecHandler
from utils.decimal_encoder import DecimalEncoder
from utils.call_models import CallModel
from utils.common_methods import lob_to_string
from .chinese_preprocessor import preprocess_cn_query

class KBSearch:
    """Knowledge base search"""
    def __init__(self, tool_params: ToolParams):
        self.tool_params = tool_params

    async def search(self, question: str, security: int) -> list[KBResult] | None:
        """Search"""
        # 0. 预处理问题，用于向量检索和全文检索，语义检索需要字符串，全文检索需要词元列表
        expand_question = await preprocess_cn_query(question)
        if expand_question is None:
            logger.warning(f"Expand question failed: {question}")
            vector_search_question = question
            full_text_question = question
        else:
            vector_search_question = expand_question.get("semantic", question)
            full_text_question = expand_question.get("fulltext", question)

        # 1. Get model ID
        repo = KbotMdKbRepository()
        models = await repo.get_model_by_kbid(self.tool_params.tool_id)
    
        if models:
            self.tool_params.kb_catogory = models[0]
            logger.debug(f"KB category: {models[0]}")
            self.tool_params.img2txt_model = models[1]
            logger.debug(f"Image to text model: {models[1]}")
            self.tool_params.img_embed_model = models[2]
            logger.debug(f"Image embedding model: {models[2]}")
            self.tool_params.txt_embed_model = models[3]
            logger.debug(f"Text embedding model: {models[3]}")
        else:
            logger.warning(f"Embedding model not found for KB {self.tool_params.tool_id}")
            return None
        
        # 2. Decide search method
        if self.tool_params.kb_catogory == KbCategory.KBOT.value:
            if self.tool_params.search_type == KBSearchType.VECTOR.value:
                logger.debug("Search method: vector")
                return await self.search_by_vector(vector_search_question, security) # type: ignore
            elif self.tool_params.search_type == KBSearchType.FULLTEXT.value:
                logger.debug("Search method: full text")
                return await self.serch_by_full_text(full_text_question, security) # type: ignore
            elif self.tool_params.search_type == KBSearchType.SUMMARY.value:
                logger.debug("Search method: summary")
                return await self.search_by_summary(question, security)
            elif self.tool_params.search_type == KBSearchType.GRAPH.value:
                logger.debug("Search method: graph")
                return await self.search_by_graph(question, security)
            else:
                logger.warning(f"Search method not implemented for KB {self.tool_params.tool_id}")
                pass
        else:
            logger.warning(f"Search method not implemented for KB {self.tool_params.tool_id}")
            pass

    
    async def search_by_vector(self, question: str, security: int) -> list[KBResult] | None:
        """Search by vector"""
        # Call embedding service
        model_id = self.tool_params.txt_embed_model
        if not model_id:
            logger.warning(f"Embedding model not found for KB {self.tool_params.tool_id}")
            return None
        model_repo = KbotMdModelsRepository()
        model_unique_name = await model_repo.get_unique_name_by_id(model_id)
        if not model_unique_name:
            logger.warning(f"Embedding model not found for KB {self.tool_params.tool_id}")
            return None
        try:
            results = await CallModel().call_embedding_model(model_unique_name, [question])
            if results is None:
                logger.error("Embedding service returned no results")
                return None
            kb_results = []
            for result in results:
                embedding = result.embedding
                kb_result = await self.get_similar_records(embedding, security)
                if kb_result is None:
                    continue
                else:
                    kb_results.extend(kb_result)
            return kb_results

        except Exception as e:
            logger.error(f"Embedding service error: {str(e)}")
            return None

    async def get_similar_records(self, query_vec: list[float], security: int) -> list[KBResult] | None:
        """Get similar records from the vector database"""
        # Perform similarity search
        repo = KbotBizTxtEmbeddingRepository()
        convertor = OracleVecHandler()
        vec = convertor.convert(query_vec, to_string=True)
        try:
            logger.debug(f"Vector search KB ID: {self.tool_params.tool_id}")
            logger.debug(f"Vector search security: {security}")
            logger.debug(f"Vector search threshold: {self.tool_params.threshold}")
            logger.debug(f"Vector search top_k: {self.tool_params.top_k}")
            dataset = await repo.get_similar_embeddings(
                kb_id = self.tool_params.tool_id,
                query_vec = vec,  # type: ignore
                security = security,
                similarity_threshold = self.tool_params.threshold,
                top_k = self.tool_params.top_k
            )
            if not dataset:
                logger.info(f"Vector search returned no results")
                return None
            results = []

            for data in dataset:
                chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                result = KBResult()
                result.file_id = data[0]
                result.chunk_type = chunk_meta.get("chunk_type", 1)
                result.page_num = chunk_meta.get("page_num", 1)
                result.content = await lob_to_string(data[1])
                result.similarity = data[3]
                result.weight = self.tool_params.tool_weight # type: ignore
                results.append(result)

            logger.debug(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.debug(f"Vector search failed: {str(e)}")
            return None
        
    async def serch_by_full_text(self, keywords: list[str], security: int) -> list[KBResult] | None:
        """Search by full text"""
        repo = KbotBizTxtEmbeddingRepository()
        try:
            logger.debug(f"Full text search KB ID: {self.tool_params.tool_id}")
            logger.debug(f"Full text search keywords: {keywords}")
            datasets = []
            unique_keys = set(keywords)  # 去除重复的关键字

            for key in unique_keys:
                logger.debug(f"Full text search token: {key}")
                ds = await repo.full_text_search(kb_id=self.tool_params.tool_id, keyword=key, security=security)
                if ds:
                    datasets.extend(ds)

            if not datasets:
                logger.info(f"Full text search returned no results")
                return None
            else:
                results = []
                for data in datasets:
                    chunk_meta = json.loads(json.dumps(data[2], cls=DecimalEncoder))
                    result = KBResult()
                    result.file_id = data[0]
                    result.chunk_type = getattr(chunk_meta, "chunk_type", 1)
                    result.page_num = getattr(chunk_meta, "page_num", 1)
                    result.content = await lob_to_string(data[1])
                    result.similarity = data[3]
                    result.weight = self.tool_params.tool_weight # type: ignore
                    results.append(result)
                logger.debug(f"Full text search found {len(results)} results")
                return results
            
        except Exception as e:
            logger.exception(f"Full text search failed: {str(e)}")
            return None

    async def search_by_summary(self, question: str, security: int) -> list[KBResult] | None:
        pass
    
    async def search_by_graph(self, question: str, security: int) -> list[KBResult] | None:
        pass