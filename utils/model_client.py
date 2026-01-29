import aiohttp
from PIL import Image
from decimal import Decimal
from loguru import logger
from typing import Any
from microservices.embedding.model.base import EmbeddingDataItem
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from .encoder import ImageEncoder
from core.config.settings import get_embed_config, get_llm_config, get_reranker_config, get_vlm_config, get_prompt_config
from core.exceptions import *


class CallModel():
    """模型微服务客户端"""
    def __init__(self):
        self.embedding_config = get_embed_config()  # 获取 Embedding 模型配置
        self.llm_config = get_llm_config()  # 获取 LLM 模型配置
        self.reranker_config = get_reranker_config()  # 获取 Reranker 模型配置
        self.vlm_config = get_vlm_config()  # 获取 VLMPrompt 模型配置
        self.prompt_config = get_prompt_config()  # 获取 Prompt 配置
        self.prompt_repo = KbotMdPromptRepository()  # 获取提示信息仓库
        self.model_repo = KbotMdModelsRepository()  # 获取模型仓库

    async def _get_model_name(self, model_id: int) -> str:
        """根据模型ID获取模型名称"""
        model_name = await self.model_repo.get_name_by_id(model_id)
        if not model_name:
            raise ValueError(f"模型ID {model_id} 不存在 model_name")
        return model_name

    async def call_embedding_model(self,
                                model_id: int,
                                texts: list[str],
                                batch_size: int | None = None,
                                is_query: bool = True,
                                use_health_check_timeout: bool = False
                                ) -> list[EmbeddingDataItem]:
        """
        调用嵌入模型获取文本向量

        Args:
            model_id: 模型ID
            texts: 文本列表
            batch_size: 批处理大小
            is_query: 是否为查询模式
            use_health_check_timeout: 是否使用健康检查超时（较短）

        Returns:
            嵌入数据项列表
        """
        model_name = await self._get_model_name(model_id)
        service_host = self.embedding_config.service_host
        service_port = self.embedding_config.service_port
        total = self.embedding_config.health_check_timeout if use_health_check_timeout else self.embedding_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/embeddings"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_name": model_name,
            "texts": texts,
            "batch_size": batch_size,
            "is_query": is_query
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"嵌入服务错误: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

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
                        msg = f"嵌入服务返回了意外的响应格式: {response_data}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    logger.info("成功获取嵌入向量")
                    return embeddings

        except aiohttp.ClientConnectorError as e:
            msg = f"无法连接到嵌入服务 {service_host}:{service_port}，请检查服务是否启动"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"嵌入服务响应超时（{total}秒），请检查服务状态"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            msg = f"嵌入服务发生错误: {e}"
            logger.error(msg)
            raise InternalServerError(msg)
    
    async def compute_similarity(self, 
                                model_id: int, 
                                text1: str, 
                                text2: str, 
                                method: str = "cosine"
                                ) -> float:
        """
        计算两个文本之间的相似度
        
        Args:
            model_id: 用于计算相似度的模型ID
            text1: 第一个文本
            text2: 第二个文本
            method: 相似度计算方法，支持"cosine"(余弦相似度)和"dot"(点积)
            
        Returns:
            相似度分数
        """
        model_name = await self._get_model_name(model_id)
        service_host = self.embedding_config.service_host
        service_port = self.embedding_config.service_port
        total = self.embedding_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/similarity"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_name": model_name,
            "text1": text1,
            "text2": text2,
            "method": method
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"相似度计算服务错误: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)
                    
                    response_data = await response.json()
                    similarity = response_data["similarity"]
                    logger.info("成功计算相似度")
                    return similarity
                    
        except Exception as e:
            msg = f"相似度计算服务发生错误: {e}"
            logger.error(msg)
            raise InternalServerError(msg)

    async def call_reranker_model(self,
                                  model_id: int,
                                  query: str,
                                  documents: list[str],
                                  top_k: int | None,
                                  use_health_check_timeout: bool = False
                                ) -> list[dict[str, Any]]:
        """
        调用重排序模型对文档进行重新排序

        Args:
            model_id: 用于重排序的模型ID
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的顶部文档数量（None表示返回所有）
            use_health_check_timeout: 是否使用健康检查超时（较短）

        Returns:
            重排序结果列表
        """
        model_name = await self._get_model_name(model_id)
        service_host = self.reranker_config.service_host
        service_port = self.reranker_config.service_port
        total = self.reranker_config.health_check_timeout if use_health_check_timeout else self.reranker_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/rerank"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model_name": model_name,
            "query": query,
            "documents": documents,
            "top_k": int(top_k) if top_k else 99999  # 设置一个很大的值，防止rerank返回的文档数小于top_k
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"重排序服务错误: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    response_data = await response.json()
                    rerank = response_data["rerankers"]
                    logger.info("成功获取重排序结果")
                    return rerank
        except aiohttp.ClientConnectorError as e:
            msg = f"无法连接到重排序服务 {service_host}:{service_port}，请检查服务是否启动"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"重排序服务响应超时（{total}秒），请检查服务状态"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            msg = f"重排序服务发生错误: {e}"
            logger.error(msg)
            raise InternalServerError(msg)
        
    async def call_llm_model(self, model_id: int, prompt: str, **kwargs):
        """
        调用LLM微服务并处理SSE格式的响应

        Args:
            model_id: 模型ID
            prompt: 输入的提示信息
            **kwargs: 其他可选参数，如stream、temperature等

        Returns:
            异步生成器，逐块产生LLM的响应
        """
        model_name = await self._get_model_name(model_id)
        service_host = self.llm_config.service_host
        service_port = self.llm_config.service_port
        use_health_check_timeout = kwargs.pop("use_health_check_timeout", False)
        total = self.llm_config.health_check_timeout if use_health_check_timeout else self.llm_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # 构建请求体
        payload = {
            "model_name": model_name,
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

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"LLM服务错误: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    async for raw_chunk in response.content:
                        yield raw_chunk.decode('utf-8')
        except aiohttp.ClientConnectorError as e:
            msg = f"无法连接到LLM服务 {service_host}:{service_port}，请检查服务是否启动"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"LLM服务响应超时（{total}秒），请检查服务状态"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            msg = f"LLM服务发生错误: {e}"
            logger.error(msg)
            raise InternalServerError(msg)
        

    async def call_vlm_model(
            self,
            model_id: int,
            image: str | Image.Image,
            prompt: str,
            model_name: str | None = None,
            **kwargs
        ) -> str:
            """调用视觉语言模型进行图片解析。

            Args:
                model_id: 模型ID。
                image: 输入图片（文件路径或 PIL.Image 对象）。
                prompt: 完整的提示词文本（必填）。
                model_name: 模型名称（可选）。
                **kwargs: 推理的额外参数（如 temperature, max_tokens 等）。

            Returns:
                str: 模型生成的输出文本。
            """
            if not model_name:
                model_name = await self._get_model_name(model_id)
            service_host = self.vlm_config.service_host
            service_port = self.vlm_config.service_port
            
            # 1. 超时配置
            total_timeout = self.vlm_config.timeout
            timeout = aiohttp.ClientTimeout(total=total_timeout)
            
            url = f"http://{service_host}:{service_port}/v1/inference"
            headers = {"Content-Type": "application/json"}

            # 2. 图片编码（Base64）
            try:
                image_base64 = await ImageEncoder.encode(image)
            except Exception as e:
                msg = f"VLM 图片编码失败: {e}"
                logger.error(msg)
                raise InternalServerError(msg)

            # 3. 构建 OpenAI 兼容格式的消息体
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

            # 4. 构建请求体
            payload = {
                "model_name": model_name,
                "messages": messages,
                "stream": False,
                **kwargs
            }

            # 5. 执行请求
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        # 处理 HTTP 错误
                        if response.status != 200:
                            error_text = await response.text()
                            msg = f"VLM 服务 HTTP {response.status} 错误: {error_text}"
                            logger.error(msg)
                            raise InternalServerError(msg)

                        response_data = await response.json()
                        
                        # 提取内容
                        try:
                            content = response_data["choices"][0]["message"]["content"]
                            logger.info(f"VLM 解析成功 | 模型: {model_id} | Prompt长度: {len(prompt)}")
                            return content
                        except (KeyError, IndexError) as e:
                            msg = f"VLM 响应格式非法: {str(e)}"
                            logger.error(msg)
                            raise InternalServerError(msg)

            # 6. 异常分类捕获
            except aiohttp.ClientConnectorError:
                msg = f"无法连接到 VLM 服务 {service_host}:{service_port}"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except aiohttp.ServerTimeoutError:
                msg = f"VLM 服务响应超时 ({total_timeout}s)"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except Exception as e:
                msg = f"VLM 调用过程中发生异常: {str(e)}"
                logger.exception(msg)
                raise InternalServerError(msg)
            