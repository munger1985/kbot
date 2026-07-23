import asyncio
import json
import oci
from pydantic import Field
from loguru import logger

from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse

class OCIEmbeddingConfig(EmbeddingConfig):
    api_endpoint: str = Field(..., description="OCI Generative AI Service Endpoint URL")
    compartment_id: str = Field(..., description="OCI Compartment OCID")
    config_file: dict | str = Field(..., description="OCI Authentication Configuration (dict or JSON string)")
    input_type_query: str = Field("search_query", description="Input type for query texts")
    input_type_doc: str = Field("search_document", description="Input type for document texts")

class OCIEmbedding(BaseEmbedding[OCIEmbeddingConfig]):
    """OCI Generative AI Embedding implementation"""
    def __init__(self, config: OCIEmbeddingConfig):
        super().__init__(config)
        self.client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None
        self._is_initialized = False
        self.batch_size = 96  # Optimized batch size for OCI API
        self._semaphore = asyncio.Semaphore(5)  # Rate limiting for concurrent API calls

    async def startup(self) -> None:
        """Initialize OCI Generative AI client with authentication"""
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
            logger.info("✅ OCI Embedding Client Initialized Successfully")
        except Exception as e:
            logger.error(f"❌ OCI Client Initialization Failed: {e}")
            raise

    async def _embed_batch(self, texts: list[str], input_type: str) -> tuple[list[list[float]], int]:
        """
        Process a single batch of texts with rate limiting
        
        Args:
            texts: Batch of texts to embed
            input_type: Input type (search_query/search_document)
            
        Returns:
            Tuple of (list of embeddings, total tokens used)
        """
        async with self._semaphore:
            try:
                # Build embed text request details following OCI API specifications
                # Note: OCI requires input_type to be uppercase (e.g., 'SEARCH_QUERY')
                # Some models (like V4) are very strict about this field
                embed_details = oci.generative_ai_inference.models.EmbedTextDetails()
                embed_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=self.config.model_name
                )
                embed_details.compartment_id = self.config.compartment_id
                embed_details.inputs = texts
                embed_details.input_type = input_type.upper()
                embed_details.truncate = "END"

                # Execute synchronous OCI API call in executor to avoid blocking event loop
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: self.client.embed_text(embed_details) # type: ignore
                )
                
                # Extract embeddings and token usage metrics
                embeddings = response.data.embeddings # type: ignore
                tokens = 0
                if hasattr(response.data, 'meta') and response.data.meta: # type: ignore
                    # Compatibility handling for token usage metadata
                    meta = response.data.meta # type: ignore
                    billed_tokens = getattr(meta, 'billed_tokens', None)
                    if billed_tokens:
                        tokens = int(getattr(billed_tokens, 'tokens', 0))

                return embeddings, tokens

            except Exception as e:
                logger.error(f"❌ OCI Embedding Request Failed: {e}")
                # Log partial payload for debugging (first text only to avoid clutter)
                logger.debug(f"请求载荷样例：{texts[:1]}... 当前批次数量：{len(texts)}")
                raise

    async def embed(self, texts: list[str], is_query: bool = True, **kwargs) -> EmbeddingResponse:
        """
        Generate embeddings for text list using OCI Generative AI API
        
        Args:
            texts: List of texts to embed
            is_query: Whether texts are query inputs (vs document inputs)
            **kwargs: Additional parameters (for interface compatibility)
            
        Returns:
            Standard OpenAI-formatted EmbeddingResponse
        """
        if not self._is_initialized:
            await self.startup()
        
        if not texts:
            return self._build_empty_response(self.config.model_name)
        
        input_type = self.config.input_type_query if is_query else self.config.input_type_doc
        
        # Process texts in batches for efficient API calls
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tasks.append(self._embed_batch(batch, input_type))

        # Execute all batch requests concurrently
        results = await asyncio.gather(*tasks)

        # Aggregate results from all batches
        all_embeddings = []
        total_tokens = 0
        for embeddings_batch, tokens_batch in results:
            all_embeddings.extend(embeddings_batch)
            total_tokens += tokens_batch

        # Return standardized OpenAI-formatted response
        return self._build_standard_response(
            embeddings=all_embeddings,
            model_name=self.config.model_name,
            tokens=total_tokens
        )
    
    async def shutdown(self) -> None:
        """Clean up OCI client resources"""
        self.client = None
        self._is_initialized = False
        logger.info("♻️ OCI Embedding Client Closed Successfully")
