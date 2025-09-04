import oci
from loguru import logger
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class OCIEmbeddingConfig(EmbeddingConfig):
    """Configuration for OCI embedding client."""
    compartment_id: str
    config_file: dict
    api_endpoint: str


class OCIEmbedding(BaseEmbedding):
    """OCI embedding client implementation."""

    def __init__(self, config: OCIEmbeddingConfig):
        """Initialize OCI embedding client.
        
        Args:
            config: OCI embedding configuration
        """
        self.config = config
        self.client = None
        self._is_running = False
    
    async def startup(self) -> None:
        """Initialize the OCI client."""
        try:
            if isinstance(self.config.config_file, str):  # type: ignore
                oci_config = json.loads(self.config.config_file) # type: ignore
            else:
                oci_config = self.config.config_file # type: ignore

            oci_config = oci_config

            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint, # type: ignore
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10,240))
            self._is_running = True
            logger.info("OCI client initialized")
        except Exception as e:
            logger.error(f"Error initializing OCI client: {str(e)}")
            raise RuntimeError(f"Error initializing OCI client: {str(e)}")
        
    async def shutdown(self) -> None:
        """Shutdown the OCI client."""
        if self.client:
            self.client = None
        self._is_running = False
        logger.info("OCI client shutdown")

    async def health_check(self) -> bool:
        """Check the health of the OCI client.
        
        Returns:
            bool: True if the client is healthy, False otherwise.
        """
        if not self._is_running or not self.client:
            return False
        try:
            # Perform a simple OCI service call to verify availability
            embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
            embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.config.model_name)
            embed_text_detail.inputs = ["test"]
            embed_text_detail.truncate = "NONE"
            embed_text_detail.compartment_id = self.config.compartment_id
            self.client.embed_text(embed_text_detail)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

        
    async def embed(
        self,
        texts: list[str],
        batch_size: int = 1
    ) -> EmbeddingResponse:
        """Embed a list of texts in batches."""
        if not self._is_running:
            await self.startup()
        if batch_size <= 0:
            batch_size = 1

        all_embeddings = []

        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{len(texts)//batch_size + 1}")

                embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
                embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.config.model_name)
                embed_text_detail.inputs = batch
                embed_text_detail.truncate = "NONE"
                embed_text_detail.compartment_id = self.config.compartment_id
                embed_text_response = self.client.embed_text(embed_text_detail) # type: ignore
                all_embeddings.extend(embed_text_response.data.embeddings) # type: ignore

            embeddings_data = [
                EmbeddingDataItem(
                    embedding=embedding,
                    index=idx,
                    object="embedding"
                ) for idx, embedding in enumerate(all_embeddings)
            ]
            return EmbeddingResponse(
                data=embeddings_data,
                model=self.config.model_name,
                object="list",
                usage={}
            )

        except Exception as e:
            logger.exception(f"Error in OCI embedding for batch: {e}")
            raise
        
