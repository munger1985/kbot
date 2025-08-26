import oci
import json
from loguru import logger
from .base import LLMConfig, BaseLLM

class OCILLMConfig(LLMConfig):
    """Configuration for OCI LLM client."""
    temperature: float = 1.0
    max_tokens: int = 600
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float = 0
    presence_penalty: float = 0
    api_endpoint: str
    compartment_id: str
    config_file: dict


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
            # Parse config_file from JSON string to dict if needed
            if isinstance(self.config.config_file, str):  # type: ignore
                oci_config = json.loads(self.config.config_file) # type: ignore
            else:
                oci_config = self.config.config_file # type: ignore
            
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
        
        # OCI max_tokens has a max limit of 4000
        max_tokens = kwargs.get('max_tokens', self.config.max_tokens) # type: ignore
        

        

        if "cohere" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.CohereChatRequest()
            chat_request.message = converted_messages
            chat_request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty) # type: ignore
            chat_request.top_p = kwargs.get('top_p', self.config.top_p) # type: ignore
            chat_request.top_k = kwargs.get('top_k', self.config.top_k) # type: ignore

            if max_tokens and max_tokens > 4000:
                max_tokens = 4000
                logger.warning("OCI cohere model max_tokens limit exceeded, setting max_tokens to 4000.")

        elif "grok" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            content = oci.generative_ai_inference.models.TextContent()
            content.text = converted_messages
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            chat_request.messages = [message]
            
            chat_request.top_p = kwargs.get('top_p', self.config.top_p) # type: ignore
            chat_request.top_k = kwargs.get('top_k', self.config.top_k) # type: ignore
            if max_tokens and max_tokens > 20000:
                max_tokens = 20000
                logger.warning("OCI grok model max_tokens limit exceeded, setting max_tokens to 20000.")

        elif "llama" in self.config.model_name.lower():
            chat_request = oci.generative_ai_inference.models.GenericChatRequest()
            chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
            content = oci.generative_ai_inference.models.TextContent()
            content.text = converted_messages
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            chat_request.messages = [message]
            
            chat_request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty) # type: ignore
            chat_request.presence_penalty = kwargs.get('presence_penalty', self.config.presence_penalty) # type: ignore
            chat_request.top_p = kwargs.get('top_p', self.config.top_p) # type: ignore
            if max_tokens and max_tokens > 600:
                max_tokens = 600
                logger.warning("OCI llama model max_tokens limit exceeded, setting max_tokens to 600.")
        else:
            raise ValueError(f"Unsupported OCI provider: {self.provider}")

        chat_request.max_tokens = max_tokens
        chat_request.temperature = kwargs.get('temperature', self.config.temperature) # type: ignore
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