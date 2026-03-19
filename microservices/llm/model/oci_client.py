import oci
import json
import asyncio
from typing import Any
from pydantic import Field
from loguru import logger

from .base import LLMConfig, BaseLLM

class OCILLMConfig(LLMConfig):
    """Configuration class for OCI Generative AI LLM client.
    
    Extends the base LLM configuration with OCI-specific parameters and 
    model generation settings compatible with OCI's Generative AI service.
    """
    temperature: float | None = Field(
        None, 
        ge=0, 
        le=2, 
        description="Temperature parameter controlling output randomness (0-2)"
    )
    top_p: float | None = Field(
        None, 
        ge=0, 
        le=1, 
        description="Top-p sampling parameter for nucleus sampling (0-1)"
    )
    top_k: int | None = Field(
        None, 
        ge=0, 
        description="Top-k sampling parameter for token selection (0+)"
    )
    frequency_penalty: float | None = Field(
        None, 
        ge=-2, 
        le=2, 
        description="Frequency penalty to reduce repetitive text (-2 to 2)"
    )
    presence_penalty: float | None = Field(
        None, 
        ge=-2, 
        le=2, 
        description="Presence penalty to encourage new topics (-2 to 2)"
    )
    api_endpoint: str = Field(
        ..., 
        description="OCI Generative AI service endpoint URL"
    )
    compartment_id: str = Field(
        ..., 
        description="OCI Compartment OCID for resource authorization"
    )
    config_file: dict | str = Field(
        ..., 
        description="OCI authentication configuration (dict or JSON string)"
    )

class OCIClient(BaseLLM[OCILLMConfig]):
    """
    OCI Generative AI optimized LLM implementation.
    
    Provides asynchronous interface to OCI's Generative AI service with 
    model-specific adaptations for Cohere, Llama 3, Grok, and other models 
    available on OCI. Handles authentication, request formatting, and 
    response processing while maintaining compatibility with the BaseLLM interface.
    """
    
    def __init__(self, config: OCILLMConfig):
        """Initialize OCI LLM client with configuration.
        
        Args:
            config: OCI-specific LLM configuration object
        """
        super().__init__(config)
        self.client: oci.generative_ai_inference.GenerativeAiInferenceClient | None = None
        self._is_initialized = False
        self.config = config

    async def startup(self) -> None:
        """Asynchronously initialize OCI Generative AI client.
        
        Handles configuration parsing and client initialization with retry 
        strategy and timeout settings. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If client initialization fails (authentication, endpoint, etc.)
        """
        if self._is_initialized:
            return

        try:
            # Parse OCI authentication configuration
            oci_config = self.config.config_file
            if isinstance(oci_config, str):
                oci_config = json.loads(oci_config)
            
            # Initialize inference client with retry and timeout settings
            # Note: OCI SDK is synchronous but connection setup is typically fast
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,
                retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,
                timeout=(10, 240)  # Connect timeout: 10s, read timeout: 240s
            )
            self._is_initialized = True
            logger.info(f"✅ OCI LLM client initialized successfully (Model: {self.config.model_name})")
        except Exception as e:
            logger.error(f"❌ OCI client initialization failed: {e}")
            raise

    def _convert_to_oci_messages(self, messages: list[dict[str, str]] | str) -> list[Any]:
        """Convert standard message format to OCI Message objects.
        
        Transforms either a list of message dictionaries (role/content pairs)
        or a single string prompt into OCI's required message format with
        proper role normalization (USER/ASSISTANT in uppercase).
        
        Args:
            messages: Input messages in standard format (list of dicts or string)
            
        Returns:
            List of OCI Message objects compatible with the API
        """
        # Convert single string to standard user message
        if isinstance(messages, str):
            messages = [{"role": "USER", "content": messages}]
        
        oci_messages = []
        for msg in messages:
            # Create text content object
            content = oci.generative_ai_inference.models.TextContent()
            content.text = msg.get("content", "")
            
            # Create OCI message object with normalized role
            oci_msg = oci.generative_ai_inference.models.Message()
            oci_msg.role = msg.get("role", "USER").upper()  # OCI requires uppercase roles
            oci_msg.content = [content]
            oci_messages.append(oci_msg)
            
        return oci_messages

    def _build_chat_request(self, messages: list[dict[str, str]] | str, **kwargs) -> Any:
        """Build model-specific chat request object for OCI API.
        
        Creates appropriate request object (CohereChatRequest or GenericChatRequest)
        based on model name, with model-specific token limits and parameter handling.
        
        Args:
            messages: Input messages in standard format
            **kwargs: Additional generation parameters to override config values
            
        Returns:
            OCI chat request object (CohereChatRequest or GenericChatRequest)
        """
        model_name = self.config.model_name.lower()
        
        # Base generation parameters with config defaults and kwargs overrides
        base_params = {
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "temperature": kwargs.get('temperature', self.config.temperature),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "top_k": kwargs.get('top_k', self.config.top_k),
        }

        # Cohere model specific request formatting
        if "cohere" in model_name:
            request = oci.generative_ai_inference.models.CohereChatRequest()
            # Cohere on OCI typically accepts a single message string
            request.message = messages[-1]['content'] if isinstance(messages, list) else messages
            request.frequency_penalty = kwargs.get('frequency_penalty', self.config.frequency_penalty)
            # Enforce Cohere token limit
            base_params["max_tokens"] = min(base_params["max_tokens"], 4000)
            
        # Llama/Grok/Generic model request formatting
        else:
            request = oci.generative_ai_inference.models.GenericChatRequest()
            request.api_format = oci.generative_ai_inference.models.GenericChatRequest.API_FORMAT_GENERIC
            request.messages = self._convert_to_oci_messages(messages)
            
            # Apply model-specific token limits
            if "llama" in model_name:
                base_params["max_tokens"] = min(base_params["max_tokens"], 4096)  # Llama 3 limit
            elif "grok" in model_name:
                base_params["max_tokens"] = min(base_params["max_tokens"], 20000)  # Grok limit

        # Inject valid parameters into request object
        for param_name, param_value in base_params.items():
            if param_value is not None and hasattr(request, param_name):
                setattr(request, param_name, param_value)
        
        return request

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """Generate chat completion response asynchronously via OCI API.
        
        Handles request construction, asynchronous execution of the synchronous
        OCI SDK call (via thread executor), and response return. Supports both
        streaming and non-streaming responses.
        
        Args:
            messages: Input messages (list of role/content dicts or single string)
            stream: Whether to enable streaming responses (True/False)
            **kwargs: Additional generation parameters to override config values
            
        Returns:
            OCI chat response object (complete or streaming)
            
        Raises:
            ValueError: If client is not initialized
            Exception: If API call fails (network, authentication, validation)
        """
        # Ensure client is initialized
        if not self._is_initialized:
            await self.startup()

        if self.client is None:
            raise ValueError("OCI Generative AI client not initialized")

        # Build complete chat request
        chat_request = self._build_chat_request(messages, **kwargs)
        chat_request.is_stream = stream

        # Create chat details with serving mode and compartment info
        chat_details = oci.generative_ai_inference.models.ChatDetails()
        chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=self.config.model_name
        )
        chat_details.chat_request = chat_request
        chat_details.compartment_id = self.config.compartment_id

        try:
            # Execute synchronous OCI SDK call asynchronously to avoid event loop blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.client.chat(chat_details)  # type: ignore
            )
            return response
        except Exception as e:
            logger.error(f"❌ OCI chat completion failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Clean up OCI client resources.
        
        Resets client instance and initialization flag, ensuring proper
        resource cleanup for graceful shutdown.
        """
        self.client = None
        self._is_initialized = False
        logger.info("♻️ OCI LLM client shutdown completed")