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
                # 1. 严格按照后端要求的 JSON 结构构造字典
                # 注意：input_type 在 OCI 上通常需要大写，如 'SEARCH_QUERY'
                # 某些模型（如 V4）对这个字段极其挑剔
                payload = {
                    "inputs": texts,
                    "inputType": input_type.upper(), # 尝试转大写以适配 OCI 枚举
                    "truncate": "END"
                }

                # 2. 直接构建 EmbedTextDetails
                # 根据 OCI SDK 的设计，EmbedTextDetails 应该直接包含所有参数
                embed_details = oci.generative_ai_inference.models.EmbedTextDetails()
                embed_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=self.config.model_name
                )
                embed_details.compartment_id = self.config.compartment_id
                embed_details.inputs = texts
                embed_details.input_type = input_type.upper()
                embed_details.truncate = "END"

                # 4. 执行异步调用
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.embed_text(embed_details)
                )
                
                # 5. 结果提取
                embeddings = response.data.embeddings
                tokens = 0
                if hasattr(response.data, 'meta') and response.data.meta:
                    # 兼容性获取 token 数
                    meta = response.data.meta
                    billed = getattr(meta, 'billed_tokens', None)
                    if billed:
                        tokens = int(getattr(billed, 'tokens', 0))

                return embeddings, tokens

            except Exception as e:
                logger.error(f"❌ OCI Embedding 请求失败: {e}")
                # 打印详细负载以便排查
                logger.debug(f"Payload was: {texts[:1]}... count: {len(texts)}")
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
