import os
import aiohttp
from loguru import logger
from typing import Optional, List, Dict, Any


async def call_embedding_model(model_id: int, texts: List[str]) -> Optional[List[List[float]]]:
    """Call embedding model"""

    embed_host = os.getenv("KBOT_EMBED_HOST", "localhost")
    embed_port = os.getenv("KBOT_EMBED_PORT", "8001")
    
    embed_url = f"http://{embed_host}:{embed_port}/embed"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_id": int(model_id),
        "texts": texts,
        "batch_size": 0
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(embed_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Embedding service error: HTTP {response.status}, {text}")
                    return None
                
                response_data = await response.json()
                query_vec = response_data["embeddings"]
                logger.info("Successfully got embedding vector")
                return query_vec
    except Exception as e:
        logger.error(f"Embedding service unavailable: {str(e)}")
        return None
    

async def call_rerank_model(model_id: int, query: str, documents: List[str], top_k: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """Call rerank model
    将文本列表进行rerank
    - **model_id**: Model ID to use for reranking.
    - **query**: Query text to be reranked.
    - **documents**: List of documents to be reranked.
    - **top_k**: Number of top documents to return (None for all)
    """

    rerank_host = os.getenv("KBOT_RERANKER_HOST", "localhost")
    rerank_port = os.getenv("KBOT_RERANKER_PORT", "8003")
    
    rerank_url = f"http://{rerank_host}:{rerank_port}/rerank"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_id": int(model_id),
        "query": query,
        "documents": documents,
        "top_k": top_k
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rerank_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Embedding service error: HTTP {response.status}, {text}")
                    return None
                
                response_data = await response.json()
                rerank = response_data["rerankers"]
                logger.info("Successfully got reranking result")
                return rerank
    except Exception as e:
        logger.error(f"Embedding service unavailable: {str(e)}")
        return None