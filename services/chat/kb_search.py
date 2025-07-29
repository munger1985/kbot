import os
import aiohttp
from loguru import logger
from typing import Optional, List
from .agent_params import AgentParams, KBResult
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.data_dict import KbCategory, KBSearchType
from utils.oracle_vec_handler import OracleVecHandler
from utils.call_models import call_embedding_model


class KBSearch:
    """Knowledge base search"""
    def __init__(self, agent_params: AgentParams):
        self.agent_params = agent_params

    async def search(self, question: str) -> Optional[List[KBResult]]:
        """Search"""
        # 1. Get model ID
        repo = KbotMdKbRepository()
        models = await repo.get_model_by_kbid(self.agent_params.tool_id)
        if models:
            self.agent_params.kb_catogory = models[0]
            self.agent_params.img2txt_model = models[1]
            self.agent_params.img_embed_model = models[2]
            self.agent_params.txt_embed_model = models[3]
        else:
            logger.warning(f"Embedding model not found for KB {self.agent_params.tool_id}")
            return None
        
        # 2. Decide search method
        if self.agent_params.kb_catogory == KbCategory.KBOT.value:
            if self.agent_params.search_type == KBSearchType.VECTOR.value:
                return await self.search_by_vector(question)
            elif self.agent_params.search_type == KBSearchType.FULLTEXT.value:
                pass
            elif self.agent_params.search_type == KBSearchType.SUMMARY.value:
                pass
            elif self.agent_params.search_type == KBSearchType.GRAPH.value:
                pass
            else:
                logger.warning(f"Search method not implemented for KB {self.agent_params.tool_id}")
                pass
        else:
            logger.warning(f"Search method not implemented for KB {self.agent_params.tool_id}")
            pass

    
    async def search_by_vector(self, question: str) -> Optional[List[KBResult]]:
        """Search by vector"""
        # Call embedding service
        model_id = self.agent_params.txt_embed_model
        try:
            results = await call_embedding_model(model_id, [question])
            if results is None:
                logger.error("Embedding service returned no results")
                return None
            convertor = OracleVecHandler()
            query_vec = convertor.convert(results[0], to_string=True)
            logger.info("Successfully got embedding vector of chat")
        except Exception as e:
            logger.error(f"Embedding service error: {str(e)}")
            return None

        # Perform similarity search
        repo = KbotBizTxtEmbeddingRepository()
        try:
            dataset = await repo.get_similar_embeddings(
                kb_id=self.agent_params.tool_id,
                query_vec=query_vec,  # type: ignore
                similarity_threshold=self.agent_params.threshold,
                top_k=self.agent_params.top_k
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
                result.chunk_doc = data[3]
                result.chunk_metadata = data[4]
                result.similarity = data[5]
                result.weight = self.agent_params.tool_weight
                results.append(result)

            logger.debug(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.debug(f"Vector search failed: {str(e)}")
            return None