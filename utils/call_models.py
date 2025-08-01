import os
import aiohttp
import asyncio
import json
from decimal import Decimal
from loguru import logger
from typing import Any, AsyncGenerator
from models.embedding.base import EmbeddingDataItem


async def call_embedding_model(model_unique_name: str, texts: list[str], batch_dize: int | None = 0) -> list[EmbeddingDataItem] | None:
    """Call embedding model"""

    embed_host = os.getenv("KBOT_EMBED_HOST", "localhost")
    embed_port = os.getenv("KBOT_EMBED_PORT", "8001")
    
    embed_url = f"http://{embed_host}:{embed_port}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_unique_name": model_unique_name,
        "texts": texts,
        "batch_size": int(batch_dize) if batch_dize else 0
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(embed_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Embedding service error: HTTP {response.status}, {text}")
                    return None
                
                response_data = await response.json()
                # Support both OpenAI format and current format
                if "data" in response_data and isinstance(response_data["data"], list):  # OpenAI format
                    embeddings = [
                        EmbeddingDataItem(
                            embedding=item["embedding"],
                            index=item.get("index", i),
                            object=item.get("object", "embedding")
                        )
                        for i, item in enumerate(response_data["data"])
                    ]
                elif isinstance(response_data, list):  # Current format
                    embeddings = [
                        EmbeddingDataItem(
                            embedding=item["embedding"],
                            index=item.get("index", i),
                            object=item.get("object", "embedding")
                        )
                        for i, item in enumerate(response_data)
                    ]
                else:  # Unexpected format
                    logger.error("Embedding service returned unexpected response format")
                    return None
                
                logger.info("Successfully got embedding vector")
                return embeddings
                
    except Exception as e:
        logger.error(f"Embedding service got error: {str(e)}")
        return None
    

async def call_reranker_model(model_unique_name: str, query: str, documents: list[str], top_k: int | None) -> list[dict[str, Any]] | None:
    """Call reranker model to rerank documents
    调用reranker微服务将文本列表进行rerank
    - **model_unique_name**: Model unique name to use for reranking.
    - **query**: Query text to be reranked.
    - **documents**: List of documents to be reranked.
    - **top_k**: Number of top documents to return (None for all)
    """

    rerank_host = os.getenv("KBOT_RERANKER_HOST", "localhost")
    rerank_port = os.getenv("KBOT_RERANKER_PORT", "8003")
    
    rerank_url = f"http://{rerank_host}:{rerank_port}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_unique_name": model_unique_name,
        "query": query,
        "documents": documents,
        "top_k": int(top_k) if top_k else 99999 # 设置一个很大的值，防止rerank返回的文档数小于top_k
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rerank_url, headers=headers, json=payload) as response:
                if response.status != 200:
                    text = await response.text()
                    logger.error(f"Reranking service error: HTTP {response.status}, {text}")
                    return None
                
                response_data = await response.json()
                rerank = response_data["rerankers"]
                logger.info("Successfully got reranking result")
                return rerank
    except Exception as e:
        logger.error(f"Reranking service got error: {str(e)}")
        return None
    
async def call_llm_model(model_unique_name: str, prompt: str, **kwargs):
    """
    调用LLM微服务并处理SSE格式的响应
    
    参数:
        model_unique_name: 模型唯一标识
        prompt: 输入的提示信息
        **kwargs: 其他可选参数，如stream、temperature等
        
    返回:
        一个异步生成器，逐块产生LLM的响应
    """
    # 获取LLM服务地址和端口
    llm_host = os.getenv("KBOT_LLM_HOST", "localhost")
    llm_port = os.getenv("KBOT_LLM_PORT", "8002")
    llm_url = f"http://{llm_host}:{llm_port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # 构建请求负载
    payload = {
        "model_unique_name": model_unique_name,
        "messages": prompt,
        "stream": kwargs.get("stream", True)  # 默认为流式
    }
    
    # 处理额外参数（Decimal转float/int）
    if kwargs:
        processed_kwargs = {}
        for k, v in kwargs.items():
            if v is not None:
                if isinstance(v, Decimal):
                    processed_kwargs[k] = float(v) if v % 1 else int(v)
                else:
                    processed_kwargs[k] = v
        payload.update(processed_kwargs)
    
    logger.debug(f"Calling LLM service with payload: {payload}")

    async with aiohttp.ClientSession() as session:
        async with session.post(llm_url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_msg = await response.text()
                logger.error(f"LLM service error: {error_msg}")
                raise Exception(f"LLM service error: {error_msg}")
                
            async for raw_chunk in response.content:
                yield raw_chunk.decode('utf-8')