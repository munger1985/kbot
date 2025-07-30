from loguru import logger
from .agent_params import ToolParams, KBResult
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from dao.data_dict import KbCategory, KBSearchType
from utils.oracle_vec_handler import OracleVecHandler
from utils.call_models import call_embedding_model
from utils.common_methods import lob_to_string


class KBSearch:
    """Knowledge base search"""
    def __init__(self, tool_params: ToolParams):
        self.tool_params = tool_params

    async def search(self, question: str, security: int) -> list[KBResult] | None:
        """Search"""
        # 1. Get model ID
        repo = KbotMdKbRepository()
        models = await repo.get_model_by_kbid(self.tool_params.tool_id)
        if models:
            self.tool_params.kb_catogory = models[0]
            self.tool_params.img2txt_model = models[1]
            self.tool_params.img_embed_model = models[2]
            self.tool_params.txt_embed_model = models[3]
        else:
            logger.warning(f"Embedding model not found for KB {self.tool_params.tool_id}")
            return None
        
        # 2. Decide search method
        if self.tool_params.kb_catogory == KbCategory.KBOT.value:
            if self.tool_params.search_type == KBSearchType.VECTOR.value:
                return await self.search_by_vector(question, security)
            elif self.tool_params.search_type == KBSearchType.FULLTEXT.value:
                pass
            elif self.tool_params.search_type == KBSearchType.SUMMARY.value:
                pass
            elif self.tool_params.search_type == KBSearchType.GRAPH.value:
                pass
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
        model_repo = KbotMdModelsRepository()
        model_unique_name = await model_repo.get_unique_name_by_id(model_id) # type: ignore
        try:
            results = await call_embedding_model(model_unique_name, [question]) # type: ignore
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
            dataset = await repo.get_similar_embeddings(
                kb_id = self.tool_params.tool_id,
                query_vec = vec,  # type: ignore
                security = security,
                similarity_threshold = self.tool_params.threshold,
                top_k = self.tool_params.top_k
            )
            if not dataset:
                logger.info(f"Vector search found no results")
                return None
            results = []

            for data in dataset:
                result = KBResult()
                result.embed_id = data[0]
                result.kb_id = data[1]
                result.file_id = data[2]
                result.chunk_doc = await lob_to_string(data[3])
                result.chunk_metadata = data[4]
                result.similarity = data[5]
                result.weight = self.tool_params.tool_weight # type: ignore
                results.append(result)

            logger.debug(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.debug(f"Vector search failed: {str(e)}")
            return None