import aiohttp
import re
import json
from datetime import datetime
from PIL import Image
from typing import AsyncGenerator
from dataclasses import dataclass
from decimal import Decimal
from loguru import logger
from typing import Any
from microservices.embedding.model.base import EmbeddingDataItem
from ..encoder import ImageEncoder
from core.config.settings import get_embed_config, get_llm_config, get_reranker_config, get_vlm_config, get_prompt_config
from core.exceptions import *


@dataclass
class LLMChunk:
    """标准化的 Chunk 对象，兼容原生推理字段"""
    content: str = ""
    reasoning_content: str | None = None


class AIModelClient():
    """模型微服务客户端"""
    def __init__(self):
        self.embedding_config = get_embed_config()  # 获取 Embedding 模型配置
        self.llm_config = get_llm_config()  # 获取 LLM 模型配置
        self.reranker_config = get_reranker_config()  # 获取 Reranker 模型配置
        self.vlm_config = get_vlm_config()  # 获取 VLMPrompt 模型配置
        self.prompt_config = get_prompt_config()  # 获取 Prompt 配置

    async def call_embedding_model(self,
                                model_name: str,
                                texts: list[str],
                                batch_size: int | None = None,
                                is_query: bool = True,
                                use_health_check_timeout: bool = False
                                ) -> list[EmbeddingDataItem]:
        """
        调用嵌入模型获取文本向量

        Args:
            model_name: 模型技术名称
            texts: 文本列表
            batch_size: 批处理大小
            is_query: 是否为查询模式
            use_health_check_timeout: 是否使用健康检查超时（较短）

        Returns:
            嵌入数据项列表
        """

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
        
    async def get_embedding(self, model_name: str, text: str) -> list[float]:
        """
        获取文本的嵌入向量

        Args:
            model_name: 模型技术名称
            text: 输入文本
            
        Returns:
            嵌入向量列表
        """
        embeddings = await self.call_embedding_model(
            model_name=model_name,
            texts=[text],
            is_query=True,
            batch_size=1
        )
        return embeddings[0].embedding
    
    async def compute_similarity(self, 
                                model_name: str, 
                                text1: str, 
                                text2: str, 
                                method: str = "cosine"
                                ) -> float:
        """
        计算两个文本之间的相似度
        
        Args:
            model_name: 用于计算相似度的模型技术名称
            text1: 第一个文本
            text2: 第二个文本
            method: 相似度计算方法，支持"cosine"(余弦相似度)和"dot"(点积)
            
        Returns:
            相似度分数
        """
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
                                  model_name: str,
                                  query: str,
                                  documents: list[str],
                                  top_k: int | None,
                                  use_health_check_timeout: bool = False
                                ) -> list[dict[str, Any]]:
        """
        调用重排序模型对文档进行重新排序

        Args:
            model_name: 用于重排序的模型技术名称
            query: 查询文本
            documents: 待重排序的文档列表
            top_k: 返回的顶部文档数量（None表示返回所有）
            use_health_check_timeout: 是否使用健康检查超时（较短）

        Returns:
            重排序结果列表
        """

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
            "top_k": top_k if top_k else 99999  # 设置一个很大的值，防止rerank返回的文档数小于top_k
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
        
    async def call_llm_model(self, model_name: str, prompt:  list[dict[str, str]] | str, **kwargs):
        """
        调用LLM微服务并处理SSE格式的响应

        Args:
            model_name: 模型技术名称
            prompt: 输入的提示信息
            **kwargs: 其他可选参数，如stream、temperature等

        Returns:
            异步生成器，逐块产生LLM的响应
        """

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

        # 处理额外参数（Decimal转float/int，datetime转ISO字符串）
        if kwargs:
            processed_kwargs = {}
            for k, v in kwargs.items():
                if v is not None:
                    if isinstance(v, Decimal):
                        processed_kwargs[k] = float(v) if v % 1 else int(v)
                    elif isinstance(v, datetime):
                        processed_kwargs[k] = v.isoformat()
                    else:
                        processed_kwargs[k] = v
            payload.update(processed_kwargs)

        # logger.debug(f"调用LLM服务，请求负载: {payload}")

        # 确保 payload 可以被 JSON 序列化
        payload = self._safe_json_payload(payload)

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

    @staticmethod
    def _safe_json_payload(obj: Any) -> Any:
        """递归转换 payload 中的非 JSON 可序列化对象（如 datetime）为安全类型。"""
        if isinstance(obj, dict):
            return {k: AIModelClient._safe_json_payload(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [AIModelClient._safe_json_payload(item) for item in obj]
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj) if obj % 1 else int(obj)
        return obj

    async def call_vlm_model(
            self,
            model_name: str,
            image: str | Image.Image,
            prompt: str,
            **kwargs
        ) -> AsyncGenerator:
            """调用视觉语言模型进行图片解析。

            Args:
                model_name: 模型技术名称。
                image: 输入图片（文件路径或 PIL.Image 对象）。
                prompt: 完整的提示词文本（必填）。
                **kwargs: 推理的额外参数（如 temperature, max_tokens 等）。

            Returns:
                str: 模型生成的输出文本。
            """
            service_host = self.vlm_config.service_host
            service_port = self.vlm_config.service_port
            
            # 1. 超时配置
            use_health_check_timeout = kwargs.pop("use_health_check_timeout", False)
            total = self.vlm_config.health_check_timeout if use_health_check_timeout else self.vlm_config.timeout
            timeout = aiohttp.ClientTimeout(total=total)
            
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
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }]

            # 4. 构建请求体
            payload = {
                "model_name": model_name,
                "messages": messages,
                "stream": kwargs.get("stream", False),
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

                        # 如果是流式请求，逐行 yield
                        if kwargs.get("stream"):
                            async for line in response.content:
                                yield line.decode('utf-8')
                        else:
                            # 如果非流式，直接返回整个 JSON（为了兼容老代码）
                            yield await response.text()

            # 6. 异常分类捕获
            except aiohttp.ClientConnectorError:
                msg = f"无法连接到 VLM 服务 {service_host}:{service_port}"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except aiohttp.ServerTimeoutError:
                msg = f"VLM 服务响应超时 ({total}s)"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except Exception as e:
                msg = f"VLM 调用过程中发生异常: {str(e)}"
                logger.exception(msg)
                raise InternalServerError(msg)
    
    async def get_llm_json(self, model_name: str, prompt:  list[dict[str, str]] | str, **kwargs) -> dict:
        """
        调用 LLM 并强制获取结构化 JSON 结果。
        内部自动处理非流式请求与 JSON 提取。
        """
        full_text = ""
        kwargs.pop('temperature', None)
        
        try:
            # 聚合 generator 产出的内容
            # 使用 stream=False，因为 json_object 模式不需要流式输出，且部分 LLM 后端对 stream+json 支持不佳
            async for chunk in self.call_llm_model(model_name=model_name, prompt=prompt, 
                                                   response_format="json_object", temperature=0, stream=False, **kwargs):
                line = chunk.strip()
                if not line or line == "data: [DONE]":
                    continue
                
                # 处理可能存在的 data: 前缀（标准 SSE）
                if line.startswith("data: "):
                    line = line[6:]
                
                try:
                    data = json.loads(line)
                    # 适配 OpenAI 格式的响应体
                    if "choices" in data and len(data["choices"]) > 0:
                        # 非流式用 message.content, 流式用 delta.content
                        choice = data["choices"][0]
                        if "message" in choice:
                            full_text += choice["message"].get("content", "")
                        elif "delta" in choice:
                            full_text += choice["delta"].get("content", "")
                except json.JSONDecodeError:
                    # 如果不是标准的 JSON chunk，则将其视为纯文本内容
                    full_text += line

            if not full_text:
                raise ValueError("LLM returned an empty response")

            # 鲁棒性 JSON 提取逻辑
            return self._extract_json_from_text(full_text)

        except Exception as e:
            logger.error(f"Failed to get JSON response from LLM: {e}")
            raise

    def _extract_json_from_text(self, text: str) -> dict:
        """
        从 LLM 的回复中提取 JSON。处理可能存在的 Markdown 标签。
        """
        text = text.strip()
        # 1. 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. 正则匹配 ```json { ... } ``` 或直接匹配 { ... }
        # 匹配最外层的花括号内容
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"Regex found potential JSON but failed to parse: {e}")
        
        raise ValueError(f"Could not parse valid JSON from LLM response: {text[:100]}...")
    
    async def get_llm_answer(self, model_name: str, prompt:  list[dict[str, str]] | str, **kwargs) -> str:
        """
        高层封装：直接获取 LLM 聚合后的纯文本字符串。
        自动处理 SSE 解析、过滤 metadata、拼接 Content。
        """
        full_content = []
        try:
            # 强制开启流式以复用 call_llm_model 的 SSE 解析逻辑
            kwargs["stream"] = True 
            
            async for raw_line in self.call_llm_model(model_name, prompt, **kwargs):
                line = raw_line.strip()
                
                # 1. 过滤空行、DONE 标志和非 data 行
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                
                try:
                    # 2. 解析 JSON 并提取内容
                    json_str = line[6:]
                    resp = json.loads(json_str)
                    choices = resp.get("choices", [])
                    if not choices:
                        continue
                    
                    # 3. 提取 delta.content (OpenAI/DeepSeek 标准)
                    content = choices[0].get("delta", {}).get("content", "")
                    if content:
                        full_content.append(content)
                        
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            return "".join(full_content).strip()
            
        except Exception as e:
            logger.error(f"get_llm_answer 失败: {e}")
            return ""
        
    async def get_vlm_answer(self, model_name: str, image: str | Image.Image, prompt: str, **kwargs) -> str:
        """
        通过复用 call_vlm_model 聚合流式输出。
        """
        full_content = []
        try:
            # 强制开启流式
            kwargs["stream"] = True 
            
            async for raw_line in self.call_vlm_model(model_name, image, prompt, **kwargs):
                line = raw_line.strip()
                
                # SSE 协议解析逻辑 (与 get_llm_answer 保持一致)
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                
                try:
                    json_str = line[6:]
                    resp = json.loads(json_str)
                    choices = resp.get("choices", [])
                    if not choices:
                        continue
                    
                    # 兼容 delta (流式) 和 message (非流式)
                    choice = choices[0]
                    content = ""
                    if "delta" in choice:
                        content = choice["delta"].get("content", "")
                    elif "message" in choice:
                        content = choice["message"].get("content", "")
                    
                    if content:
                        full_content.append(content)
                        
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

            return "".join(full_content).strip()
            
        except Exception as e:
            logger.error(f"get_vlm_answer 失败: {e}")
            return ""
        

    async def get_llm_stream_parsed(
        self, 
        model_name: str, 
        prompt: list[dict[str, str]] | str, 
        **kwargs
    ) -> AsyncGenerator[LLMChunk, None]:
        """
        基于 call_llm_model 的流式解析版本
        """
        # 强制开启流式开关
        kwargs["stream"] = True
        
        # 直接复用原有的 call_llm_model 逻辑
        # 注意：原方法返回的是 async for raw_chunk in response.content 的字符串流
        async for line in self.call_llm_model(model_name, prompt, **kwargs):
            # 这里的 line 已经是 decode('utf-8') 后的字符串
            # 但由于 response.content 的迭代可能包含多行或不完整行，需要处理前缀
            
            clean_line = line.strip()
            if not clean_line or not clean_line.startswith("data: "):
                continue
            
            data_str = clean_line[6:] # 截取 "data: " 之后的内容
            if data_str == "[DONE]":
                break

            try:
                data = json.loads(data_str)
                delta = data['choices'][0].get('delta', {})
                
                # 同时兼容普通内容和推理内容
                content = delta.get('content', '')
                # 适配 DeepSeek R1 常见的推理字段
                reasoning = delta.get('reasoning_content') or delta.get('thought_content')

                if content or reasoning:
                    yield LLMChunk(content=content, reasoning_content=reasoning)
                    
            except json.JSONDecodeError:
                continue