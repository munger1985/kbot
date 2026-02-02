import asyncio
import json
import oci
from pydantic import Field
from loguru import logger

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class OCIEmbeddingConfig(EmbeddingConfig):
    api_endpoint: str = Field(..., description="OCI Generative AI Endpoint")
    compartment_id: str = Field(..., description="OCI Compartment OCID")
    config_file: dict | str = Field(..., description="OCI Auth Config")
    input_type_query: str = Field("search_query")
    input_type_doc: str = Field("search_document")

class OCIEmbedding(BaseEmbedding[OCIEmbeddingConfig]):
    def __init__(self, config: OCIEmbeddingConfig):
        super().__init__(config)
        self.client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None
        self._is_initialized = False
        self.batch_size = 96 
        self._semaphore = asyncio.Semaphore(5)

    async def startup(self) -> None:
        if self._is_initialized: return
        try:
            oci_config = self.config.config_file
            if isinstance(oci_config, str):
                oci_config = json.loads(oci_config)

            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY
            )
            self._is_initialized = True
            logger.info("✅ OCI Embedding Client Initialized")
        except Exception as e:
            logger.error(f"❌ Initialization Failed: {e}")
            raise

    async def _embed_batch(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        async with self._semaphore:
            try:
                # 1. 彻底放弃使用具体的类，改用原始字典构造
                # 这能避开 SDK 内部 EmbeddingProvider.COHERE 的枚举校验
                cohere_request_dict = {
                    "inputs": texts,
                    "input_type": input_type,
                    "truncate": "END"
                }

                # 2. 使用基类包装字典
                # OCI SDK 的底层序列化器会根据字典内容自动识别 Provider
                embed_details = oci.generative_ai_inference.models.EmbedTextDetails()
                embed_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=self.config.model_name
                )
                embed_details.compartment_id = self.config.compartment_id
                
                # 直接赋值字典，SDK 在发送请求前会自动进行 JSON 序列化
                embed_details.embed_text_request = cohere_request_dict 

                # 3. 异步执行同步调用
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.embed_text(embed_details)
                )
                
                # 4. 结果提取 (V4 的返回结构通常与 V3 保持一致)
                embeddings = response.data.embeddings
                tokens = 0
                if hasattr(response.data, 'meta') and response.data.meta:
                    # 使用 getattr 安全获取，防止 meta 结构变化
                    billed_tokens = getattr(response.data.meta, 'billed_tokens', None)
                    if billed_tokens:
                        tokens = int(getattr(billed_tokens, 'tokens', 0))

                return embeddings, tokens

            except Exception as e:
                logger.error(f"❌ OCI Embedding V4 错误: {str(e)}")
                # 如果还是报错，打印出更详细的错误信息
                if hasattr(e, 'message'):
                    logger.error(f"详细错误信息: {e.message}") # type: ignore
                raise

    async def embed(self, texts: list[str], is_query: bool = True, **kwargs) -> EmbeddingResponse:
        if not self._is_initialized:
            await self.startup()
        
        input_type = self.config.input_type_query if is_query else self.config.input_type_doc
        
        # 分批处理
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tasks.append(self._embed_batch(batch, input_type))

        results = await asyncio.gather(*tasks)

        all_embeddings = []
        total_tokens = 0
        for embs, tks in results:
            all_embeddings.extend(embs)
            total_tokens += tks

        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.config.model_name,
            tokens=total_tokens
        )
    
    async def shutdown(self) -> None:
        self.client = None
        self._is_initialized = False
        logger.info("♻️ OCI Embedding 客户端已关闭")
