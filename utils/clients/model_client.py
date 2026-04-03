import aiohttp
import json
import re
from PIL import Image
from decimal import Decimal
from loguru import logger
from typing import Any
from microservices.embedding.model.base import EmbeddingDataItem
from ..encoder import ImageEncoder
from core.config.settings import get_embed_config, get_llm_config, get_reranker_config, get_vlm_config, get_prompt_config
from core.exceptions import *


class AIModelClient():
    """Client for AI model microservices.
    
    Provides asynchronous methods to interact with embedding, LLM, reranker, and VLM
    microservices, handling request construction, error handling, and response parsing.
    """
    def __init__(self):
        self.embedding_config = get_embed_config()  # Get Embedding model configuration
        self.llm_config = get_llm_config()          # Get LLM model configuration
        self.reranker_config = get_reranker_config()# Get Reranker model configuration
        self.vlm_config = get_vlm_config()          # Get VLM model configuration
        self.prompt_config = get_prompt_config()    # Get Prompt template configuration


    async def call_embedding_model(self,
                                model_name: str,
                                texts: list[str],
                                batch_size: int | None = None,
                                is_query: bool = True,
                                use_health_check_timeout: bool = False
                                ) -> list[EmbeddingDataItem]:
        """Call embedding model service to get text vector embeddings.

        Args:
            model_name: Name of the embedding model to use
            texts: List of text strings to generate embeddings for
            batch_size: Batch size for processing texts (None uses service default)
            is_query: Whether the texts are query inputs (True) or corpus documents (False)
            use_health_check_timeout: Whether to use shorter health check timeout instead of regular timeout

        Returns:
            list[EmbeddingDataItem]: List of embedding data items containing vectors and metadata
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
                        msg = f"Embedding service error: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    response_data = await response.json()
                    # Support both OpenAI-compatible format and legacy format
                    if "data" in response_data and isinstance(response_data["data"], list):  # OpenAI format
                        embeddings = [
                            EmbeddingDataItem(
                                embedding=item["embedding"],
                                index=item.get("index", i),
                                object=item.get("object", "embedding")
                            )
                            for i, item in enumerate(response_data["data"])
                        ]
                    elif isinstance(response_data, list):  # Legacy format
                        embeddings = [
                            EmbeddingDataItem(
                                embedding=item["embedding"],
                                index=item.get("index", i),
                                object=item.get("object", "embedding")
                            )
                            for i, item in enumerate(response_data)
                        ]
                    else:  # Unexpected format
                        msg = f"Embedding service returned unexpected response format: {response_data}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    logger.info("Successfully retrieved embedding vectors")
                    return embeddings

        except aiohttp.ClientConnectorError as e:
            msg = f"Failed to connect to embedding service {service_host}:{service_port}, please check if service is running"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"Embedding service response timed out ({total} seconds), please check service status"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            msg = f"Embedding service error occurred: {e}"
            logger.error(msg)
            raise InternalServerError(msg)

    async def call_reranker_model(self,
                                  model_name: str,
                                  query: str,
                                  documents: list[str],
                                  top_k: int | None,
                                  use_health_check_timeout: bool = False
                                ) -> list[dict[str, Any]]:
        """Call reranker model service to reorder documents based on relevance to query.

        Args:
            model_name: Name of the reranker model to use
            query: Query text to evaluate document relevance against
            documents: List of documents to reorder
            top_k: Number of top relevant documents to return (None returns all documents)
            use_health_check_timeout: Whether to use shorter health check timeout instead of regular timeout

        Returns:
            list[dict[str, Any]]: List of reranked documents with relevance scores
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
            "top_k": int(top_k) if top_k else 99999  # Set large value to prevent fewer results than requested
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"Reranker service error: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    response_data = await response.json()
                    rerank = response_data["rerankers"]
                    logger.info("Successfully retrieved reranked results")
                    return rerank
        except aiohttp.ClientConnectorError as e:
            msg = f"Failed to connect to reranker service {service_host}:{service_port}, please check if service is running"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"Reranker service response timed out ({total} seconds), please check service status"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            msg = f"Reranker service error occurred: {e}"
            logger.error(msg)
            raise InternalServerError(msg)
        
    async def call_llm_model(self, model_name: str, prompt: str, **kwargs):
        """Call LLM microservice and handle Server-Sent Events (SSE) responses.

        Args:
            model_name: Name of the LLM model to use
            prompt: Input prompt text for the model
            **kwargs: Additional optional parameters (e.g., stream, temperature, max_tokens)

        Yields:
            str: Chunked response text from LLM (decoded from raw bytes)
        """
        service_host = self.llm_config.service_host
        service_port = self.llm_config.service_port
        use_health_check_timeout = kwargs.pop("use_health_check_timeout", False)
        total = self.llm_config.health_check_timeout if use_health_check_timeout else self.llm_config.timeout
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Build request payload
        payload = {
            "model_name": model_name,
            "messages": prompt,
            "stream": kwargs.get("stream", True)  # Stream responses by default
        }

        # Process additional parameters (convert Decimal to float/int)
        if kwargs:
            processed_kwargs = {}
            for k, v in kwargs.items():
                if v is not None:
                    if isinstance(v, Decimal):
                        processed_kwargs[k] = float(v) if v % 1 else int(v)
                    else:
                        processed_kwargs[k] = v
            payload.update(processed_kwargs)

        # logger.debug(f"Calling LLM service with payload: {payload}")

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        text = await response.text()
                        msg = f"LLM service error: HTTP {response.status}, {text}"
                        logger.error(msg)
                        raise InternalServerError(msg)

                    # Read SSE response line by line and yield raw lines
                    async for line in response.content:
                        try:
                            yield line.decode('utf-8')
                        except UnicodeDecodeError as e:
                            logger.warning(f"Failed to decode chunk: {e}")
                            continue
        except aiohttp.ClientConnectorError as e:
            msg = f"Failed to connect to LLM service {service_host}:{service_port}, please check if service is running"
            logger.error(msg)
            raise InternalServerError(msg)
        except aiohttp.ServerTimeoutError:
            msg = f"LLM service response timed out ({total} seconds), please check service status"
            logger.error(msg)
            raise InternalServerError(msg)
        except Exception as e:
            logger.error(f"LLM service error occurred: {e}", exc_info=True)
            raise InternalServerError(f"LLM service error occurred: {e}")
        

    async def call_vlm_model(
            self,
            model_name: str,
            image: str | Image.Image,
            prompt: str,
            **kwargs
        ) -> str:
            """Call Vision-Language Model (VLM) for image analysis and interpretation.

            Args:
                model_name: Name of the VLM model to use
                image: Input image (file path string or PIL.Image object)
                prompt: Complete prompt text for the model (required)
                **kwargs: Additional inference parameters (e.g., temperature, max_tokens)

            Returns:
                str: Generated text output from the VLM model describing/analyzing the image
            """
            service_host = self.vlm_config.service_host
            service_port = self.vlm_config.service_port
            
            # 1. Timeout configuration
            total_timeout = self.vlm_config.timeout
            timeout = aiohttp.ClientTimeout(total=total_timeout)
            
            url = f"http://{service_host}:{service_port}/v1/inference"
            headers = {"Content-Type": "application/json"}

            # 2. Encode image to Base64 format
            try:
                image_base64 = await ImageEncoder.encode(image)
            except Exception as e:
                msg = f"VLM image encoding failed: {e}"
                logger.error(msg)
                raise InternalServerError(msg)

            # 3. Build OpenAI-compatible message payload
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

            # 4. Build complete request payload
            payload = {
                "model_name": model_name,
                "messages": messages,
                "stream": False,
                **kwargs
            }

            # 5. Execute request
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, headers=headers, json=payload) as response:
                        # Handle HTTP errors
                        if response.status != 200:
                            error_text = await response.text()
                            msg = f"VLM service HTTP {response.status} error: {error_text}"
                            logger.error(msg)
                            raise InternalServerError(msg)

                        response_data = await response.json()
                        
                        # Extract response content
                        try:
                            content = response_data["choices"][0]["message"]["content"]
                            # logger.info(f"VLM analysis successful | Model: {model_name} | Prompt length: {len(prompt)}")
                            return content
                        except (KeyError, IndexError) as e:
                            msg = f"VLM response format invalid: {str(e)}"
                            logger.error(msg)
                            raise InternalServerError(msg)

            # 6. Categorized exception handling
            except aiohttp.ClientConnectorError:
                msg = f"Failed to connect to VLM service {service_host}:{service_port}"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except aiohttp.ServerTimeoutError:
                msg = f"VLM service response timed out ({total_timeout}s)"
                logger.error(msg)
                raise InternalServerError(msg)
                
            except Exception as e:
                msg = f"VLM invocation exception occurred: {str(e)}"
                logger.exception(msg)
                raise InternalServerError(msg)
            
    async def call_llm_json(self, model_name: str, prompt: str, **kwargs) -> dict:
        """
        调用 LLM 并强制获取结构化 JSON 结果。
        内部自动处理非流式请求与 JSON 提取。
        """
        # 强制设为非流式，以便一次性获取完整响应（如果后端支持）
        # 或者为了兼容性，我们在此聚合 call_llm_model 的输出
        kwargs["stream"] = False 
        
        full_text = ""
        try:
            # 聚合 generator 产出的内容
            async for chunk in self.call_llm_model(model_name=model_name, prompt=prompt, **kwargs):
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
    
    async def get_llm_answer(self, model_name: str, prompt: str, **kwargs) -> str:
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