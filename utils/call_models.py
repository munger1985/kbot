import os
import aiohttp
import configparser
from PIL import Image
from decimal import Decimal
from loguru import logger
from typing import Any
from microservices.embedding.model.base import EmbeddingDataItem
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from .common_methods import encode_image
from nacos_manager import nacos_manager # type: ignore


async def call_embedding_model(model_unique_name: str, 
                               texts: list[str], 
                               batch_dize: int | None = 0
                               ) -> list[EmbeddingDataItem] | None:
    """Call embedding model"""

    try:
    # 从 nacos 获取 embedding 服务配置
        nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
        config_parser = configparser.ConfigParser()
        embed_config = nacos_manager.get_config("embedding", nacos_group)
        config_parser.read_string(f"[{nacos_group}]\n{embed_config}")
        service_host = config_parser.get(nacos_group, "service_host") or "0.0.0.0" # 微服务地址
        service_port = int(config_parser.get(nacos_group, "service_port")) or 9201 # 微服务通信端口
    except Exception as e:
        # 如果从 nacos 获取 embedding 服务配置失败，则使用默认配置
        service_host = "0.0.0.0"
        service_port = 9201

    
    url = f"http://{service_host}:{service_port}/v1/embeddings"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_unique_name": model_unique_name,
        "texts": texts,
        "batch_size": int(batch_dize) if batch_dize else 0
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
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

    host = os.getenv("KBOT_RERANKER_HOST", "localhost")
    port = os.getenv("KBOT_RERANKER_PORT", "8003")
    
    url = f"http://{host}:{port}/v1/rerank"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model_unique_name": model_unique_name,
        "query": query,
        "documents": documents,
        "top_k": int(top_k) if top_k else 99999 # 设置一个很大的值，防止rerank返回的文档数小于top_k
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
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
    try:
    # 从 nacos 获取 llm 服务配置
        nacos_group = os.getenv("NACOS_GROUP") or "DEV_GROUP" # Nacos分组
        config_parser = configparser.ConfigParser()
        embed_config = nacos_manager.get_config("llm", nacos_group)
        config_parser.read_string(f"[{nacos_group}]\n{embed_config}")
        service_host = config_parser.get(nacos_group, "service_host") or "0.0.0.0" # 微服务地址
        service_port = int(config_parser.get(nacos_group, "service_port")) or 9202 # 微服务通信端口
    except Exception as e:
        # 如果从 nacos 获取 llm 服务配置失败，则使用默认配置
        service_host = "0.0.0.0"
        service_port = 9202

    url = f"http://{service_host}:{service_port}/v1/chat/completions"
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
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_msg = await response.text()
                logger.error(f"LLM service error: {error_msg}")
                raise Exception(f"LLM service error: {error_msg}")
                
            async for raw_chunk in response.content:
                yield raw_chunk.decode('utf-8')

async def call_vlm_model_for_parsing_picture(model_unique_name: str, 
                                             prompt_unique_name: str, 
                                             image: str | Image.Image, 
                                             **kwargs) -> str | None:
    """Call vector language model.
    
    Parameters:
    - **model_unique_name**: Model unique name.
    - **prompt_unique_name**: Get the prompt by unique name from the database.
    - **image**: Input image (file path or PIL.Image object).
    - **kwargs**: Additional arguments for inference.
    
    Returns:
    - Output text, or None on failure
    """
    
    
    # Get vector language model host and port
    host = os.getenv("KBOT_VLM_HOST", "localhost")
    port = os.getenv("KBOT_VLM_PORT", "8004")
    url = f"http://{host}:{port}/v1/inference"
    headers = {"Content-Type": "application/json"}

    # Encode image to base64
    try:
        image_base64 = await encode_image(image)
    except Exception as e:
        logger.error(f"Failed to encode image: {str(e)}")
        return None
    
    # Get prompt text
    try:
        prompt_repo = KbotMdPromptRepository()
        prompt = await prompt_repo.get_prompt_by_unique_name(prompt_unique_name)
        if not prompt:
            raise Exception(f"Prompt not found: {prompt_unique_name}")
    except Exception as e:
        logger.error(f"Failed to get prompt text: {str(e)}")
        return None

    # Build messages in required format
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ]

    # Build request payload
    payload = {
        "model_unique_name": model_unique_name,
        "messages": messages,
        "stream": False,
        **kwargs
    }
    
    # Send request
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    logger.error(f"VLM service response error: HTTP {response.status}")
                    return None
                    
                response_data = await response.json()
                output = response_data["choices"][0]["message"]["content"]
                logger.info("Successfully got VLM response")
                return output
    except Exception as e:
        logger.error(f"VLM service error: {str(e)}")
        return None 
