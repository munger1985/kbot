import aiohttp
from PIL import Image
from decimal import Decimal
from loguru import logger
from typing import Any
from microservices.embedding.model.base import EmbeddingDataItem
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from .common import encode_image
from core.config.settings import get_embed_config, get_llm_config, get_reranker_config, get_vlm_config, get_prompt_config

class CallModel():
    """模型调用类"""
    def __init__(self):
        self.embedding_config = get_embed_config()  # 获取 Embedding 模型配置
        self.llm_config = get_llm_config()  # 获取 LLM 模型配置
        self.reranker_config = get_reranker_config()  # 获取 Reranker 模型配置
        self.vlm_config = get_vlm_config()  # 获取 VLMPrompt 模型配置
        self.prompt_config = get_prompt_config()  # 获取 Prompt 配置
        self.kbot_md_prompt_repo = KbotMdPromptRepository()  # 获取提示信息仓库

    async def call_embedding_model(self, 
                                model_id: int, 
                                texts: list[str], 
                                batch_size: int = 0,
                                is_query: bool = True
                                ) -> list[EmbeddingDataItem] | None:
        """
        调用嵌入模型获取文本向量
        
        Args:
            model_id: 模型唯一标识符
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            嵌入数据项列表或None（失败时）
        """

        service_host = self.embedding_config.service_host
        service_port = self.embedding_config.service_port
        total = self.embedding_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_id": int(model_id),
            "texts": texts,
            "batch_size": int(batch_size) if batch_size else 0,
            "is_query": is_query
        }
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"嵌入服务错误: HTTP {response.status}, {text}")
                        return None
                    
                    response_data = await response.json()
                    # 支持OpenAI格式和当前格式
                    if "data" in response_data and isinstance(response_data["data"], list):  # OpenAI格式
                        embeddings = [
                            EmbeddingDataItem(
                                embedding=item["embedding"],
                                index=item.get("index", i),
                                object=item.get("object", "embedding")
                            )
                            for i, item in enumerate(response_data["data"])
                        ]
                    elif isinstance(response_data, list):  # 当前格式
                        embeddings = [
                            EmbeddingDataItem(
                                embedding=item["embedding"],
                                index=item.get("index", i),
                                object=item.get("object", "embedding")
                            )
                            for i, item in enumerate(response_data)
                        ]
                    else:  # 意外格式
                        logger.error("嵌入服务返回了意外的响应格式")
                        return None
                    
                    logger.info("成功获取嵌入向量")
                    return embeddings
                    
        except Exception as e:
            logger.error(f"嵌入服务发生错误: {str(e)}")
            return None
        
    async def call_reranker_model(self, 
                                  model_id: int, 
                                  query: str, 
                                  documents: list[str], 
                                  top_k: int | None
                                ) -> list[dict[str, Any]] | None:
        """
        调用重排序模型对文档进行重新排序
        
        Args:
            model_id: 用于重排序的模型唯一标识符
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的顶部文档数量（None表示返回所有）
            
        Returns:
            重排序结果列表或None（失败时）
        """

        service_host = self.reranker_config.service_host
        service_port = self.reranker_config.service_port
        total = self.reranker_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/rerank"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_id": int(model_id),
            "query": query,
            "documents": documents,
            "top_k": int(top_k) if top_k else 99999  # 设置一个很大的值，防止rerank返回的文档数小于top_k
        }
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        logger.error(f"重排序服务错误: HTTP {response.status}, {text}")
                        return None
                    
                    response_data = await response.json()
                    rerank = response_data["rerankers"]
                    logger.info("成功获取重排序结果")
                    return rerank
        except Exception as e:
            logger.error(f"重排序服务发生错误: {str(e)}")
            return None
        
    async def call_llm_model(self, model_id: int, prompt: str, **kwargs):
        """
        调用LLM微服务并处理SSE格式的响应
        
        Args:
            model_id: 模型唯一标识
            prompt: 输入的提示信息
            **kwargs: 其他可选参数，如stream、temperature等
            
        Returns:
            异步生成器，逐块产生LLM的响应
        """

        service_host = self.llm_config.service_host
        service_port = self.llm_config.service_port
        total = self.llm_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        
        # 构建请求体
        payload = {
            "model_id": int(model_id),
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
        
        logger.debug(f"调用LLM服务，请求负载: {payload}")

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    logger.error(f"LLM服务错误: {error_msg}")
                    raise Exception(f"LLM服务错误: {error_msg}")
                
                async for raw_chunk in response.content:
                    yield raw_chunk.decode('utf-8')

    async def call_vlm_model_for_parsing_picture(self,
                                                model_id: int, 
                                                image: str | Image.Image, 
                                                prompt_name: str | None = None, 
                                                **kwargs) -> str | None:
        """
        调用视觉语言模型进行图片解析
        
        Args:
            model_id: 模型唯一标识符
            image: 输入图片（文件路径或PIL.Image对象）
            prompt_name: 从数据库中获取的提示词名称
            **kwargs: 推理的额外参数
            
        Returns:
            输出文本或None（失败时）
        """
        
        service_host = self.vlm_config.service_host
        service_port = self.vlm_config.service_port
        total = self.vlm_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/inference"
        headers = {"Content-Type": "application/json"}

        # 将图片编码为base64
        try:
            image_base64 = await encode_image(image)
        except Exception as e:
            logger.error(f"图片编码失败: {str(e)}")
            return None
        
        # 如果没有传入 prompt_name，则使用默认的提示信息
        if not prompt_name:
            prompt_name = self.prompt_config.image2text

        # 获取提示文本
        try:
            prompt_repo = KbotMdPromptRepository()
            prompt = await prompt_repo.get_prompt_by_unique_name(prompt_name)
            if not prompt:
                raise Exception(f"提示信息未找到: {prompt_name}")
        except Exception as e:
            logger.error(f"获取提示文本失败: {str(e)}")
            return None

        # 构建所需格式的消息
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

        # 构建请求体
        payload = {
            "model_id": int(model_id),
            "messages": messages,
            "stream": False,
            **kwargs
        }
        
        # 发送请求
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"VLM服务响应错误: HTTP {response.status}")
                        return None
                        
                    response_data = await response.json()
                    output = response_data["choices"][0]["message"]["content"]
                    logger.info("成功获取VLM响应")
                    return output
        except Exception as e:
            logger.error(f"VLM服务发生错误: {str(e)}")
            return None 
        
