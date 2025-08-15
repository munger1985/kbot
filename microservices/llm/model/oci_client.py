import oci
from loguru import logger
from .base import LLMConfig, BaseLLM

class OCILLMConfig(LLMConfig):
    """Configuration for OCI LLM client."""
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    # top_k: int | None = None
    # frequency_penalty: float | None = None
    api_endpoint: str
    compartment_id: str
    config_profile: str = "DEFAULT"
    config_file: str = "~/.oci/config"


class OCIClient(BaseLLM):
    """OCI LLM client implementation."""
    
    def __init__(self, config: OCILLMConfig):
        """Initialize OCI LLM client.
        
        Args:
            config: OCI LLM configuration
        """
        super().__init__(config)
        self.client = None
        self._is_running = False
    
    async def startup(self) -> None:
        """Initialize the OCI client."""
        try:
            oci_config = oci.config.from_file(self.config.config_file, self.config.config_profile) # type: ignore
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

    
    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ):
        """Generate chat response with consistent return types.
        
        Args:
            messages: List of messages or single prompt string
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
        
        Returns:
            - chat repsonse
        
        Notes:
            This method requires the OCI chat service.
        
        """
        
        if not self._is_running:
            await self.startup()
        
        converted_messages = ""
        # Convert string to message list if needed
        if isinstance(messages, list):
            converted_messages = messages[0]['content'] if len(messages) > 0 else ''
        
        if isinstance(messages, str):
            converted_messages = messages
        
        if converted_messages == '':
            raise ValueError("No message provided to generate response for.")
        
        chat_request = oci.generative_ai_inference.models.CohereChatRequest()
        #chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        chat_request.message = converted_messages
        chat_request.max_tokens = kwargs.get('max_tokens', self.config.max_tokens) # type: ignore
        chat_request.temperature = kwargs.get('temperature', self.config.temperature) # type: ignore
        chat_request.top_p = kwargs.get('top_p', self.config.top_p) # type: ignore
        #chat_request.top_k = kwargs.get('top_k', self.config.top_k) # type: ignore
        #chat_request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty) # type: ignore
        chat_request.is_stream = stream

        chat_detail = oci.generative_ai_inference.models.ChatDetails()
        chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(model_id=self.config.model_name)
        chat_detail.chat_request = chat_request
        chat_detail.compartment_id = self.config.compartment_id # type: ignore

        try:
            response = self.client.chat(chat_detail) # type: ignore
            
            return response

        except Exception as e:
            self.ERROR_COUNTER.labels(provider="OCI").inc()
            logger.error(f"Error generating chat response with OCI: {str(e)}")
            raise Exception(f"Error generating chat response with OCI: {str(e)}")