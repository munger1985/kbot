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
        """
        核心修正：使用字典构造模型，彻底绕过 import 限制
        """
        async with self._semaphore:
            # 1. 构造 Cohere 特有的请求负载字典
            # OCI SDK 的底层会根据此字典自动映射到对应的类
            cohere_payload = {
                "inputs": texts,
                "input_type": input_type,
                "truncate": "END"
            }

            # 2. 构造通用的 EmbedTextDetails
            # 注意：embed_text_request 虽然在 IDE 里报错，但运行时接受对象或字典
            embed_details = oci.generative_ai_inference.models.EmbedTextDetails(
                serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=self.config.model_name
                ),
                compartment_id=self.config.compartment_id,
                # 关键点：直接传入字典或动态构造
                embed_text_request=oci.generative_ai_inference.models.CohereEmbedTextRequest(**cohere_payload)  # type: ignore
                if hasattr(oci.generative_ai_inference.models, "CohereEmbedTextRequest") 
                else cohere_payload
            )

            loop = asyncio.get_event_loop()
            try:
                # 执行同步调用
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.embed_text(embed_details)
                )
                
                # 结果提取
                embeddings = response.data.embeddings
                tokens = 0
                # 尝试安全获取 billed_tokens
                if hasattr(response.data, 'meta') and response.data.meta:
                    meta = response.data.meta
                    if hasattr(meta, 'billed_tokens') and meta.billed_tokens:
                        tokens = int(meta.billed_tokens.tokens or 0)

                return embeddings, tokens
            except Exception as e:
                logger.error(f"❌ OCI Inference Error: {e}")
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