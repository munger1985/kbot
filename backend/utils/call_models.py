import os
import aiohttp
from loguru import logger
from typing import Optional, List


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