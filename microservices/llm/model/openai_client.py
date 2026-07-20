from typing import AsyncGenerator
from openai import AsyncOpenAI, APIError
from loguru import logger
from pydantic import Field

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageParam
)
from .base import LLMConfig, BaseLLM


class OpenaiLLMConfig(LLMConfig):
    """Configuration class for OpenAI LLM client.
    
    Extends the base LLM configuration with OpenAI-specific generation parameters
    and API connection settings. All parameters support None values to allow
    using OpenAI API defaults when not explicitly configured.
    """
    temperature: float | None = Field(
        None, 
        ge=0, 
        le=2, 
        description="Temperature parameter controlling output randomness (0-2, higher=more random)"
    )
    top_p: float | None = Field(
        None, 
        ge=0, 
        le=1, 
        description="Top-p sampling parameter for nucleus sampling (0-1)"
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
    timeout: int | None = Field(
        None, 
        gt=0, 
        description="Request timeout in seconds (must be greater than 0)"
    )
    api_endpoint: str = Field(
        ..., 
        description="Base URL for OpenAI-compatible API endpoint"
    )
    api_key: str = Field(
        ..., 
        description="API key for authentication with OpenAI service"
    )


class OpenaiClient(BaseLLM[OpenaiLLMConfig]):
    """
    OpenAI-compatible LLM client implementation with enhanced features.
    
    Key optimizations:
    - Dynamic parameter resolution (kwargs override config)
    - Robust message format validation and normalization
    - Comprehensive logging for request/response tracking
    - Proper resource management with async client lifecycle
    - Support for streaming responses and tool calls
    """
    
    def __init__(self, config: OpenaiLLMConfig):
        """Initialize OpenAI LLM client with configuration.
        
        Args:
            config: OpenAI-specific LLM configuration object
        """
        super().__init__(config)
        self._client: AsyncOpenAI | None = None
        self._is_initialized = False
    
    async def startup(self) -> None:
        """Asynchronously initialize OpenAI async client.
        
        Creates the AsyncOpenAI client instance with configured API key,
        base URL, and timeout settings. Idempotent - safe to call multiple times.
        
        Raises:
            Exception: If client initialization fails (invalid config, network, etc.)
        """
        if self._is_initialized:
            return

        try:
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_endpoint,
                timeout=self.config.timeout
            )
            self._is_initialized = True
            logger.info(f"✅ OpenAI client ready: {self.config.api_endpoint}")
        except Exception as e:
            logger.error(f"❌ OpenAI client initialization failed: {e}")
            raise
        
    async def shutdown(self) -> None:
        """Clean up client resources gracefully.
        
        Closes the async client connection and resets initialization state.
        Ensures proper resource release during application shutdown.
        """
        if self._client:
            await self._client.close()
            self._client = None
        self._is_initialized = False
        logger.info("♻️ OpenAI client closed safely")

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion | AsyncGenerator[ChatCompletionChunk, None]:
        """
        Unified chat completion interface for OpenAI API.
        
        Handles message normalization, parameter aggregation, and API execution
        with support for both streaming and non-streaming responses.
        
        Args:
            messages: Input messages - either a list of role/content dictionaries
                or a single string (interpreted as user message)
            stream: If True, returns async generator of ChatCompletionChunk;
                if False, returns complete ChatCompletion object
            **kwargs: Additional generation parameters to override config values
                (temperature, max_tokens, tools, response_format, etc.)
                
        Returns:
            ChatCompletion: Complete response object (when stream=False)
            AsyncGenerator[ChatCompletionChunk]: Streaming response chunks (when stream=True)
                
        Raises:
            ValueError: If client is not initialized
            APIError: For OpenAI API-specific errors
            Exception: For other unexpected errors during generation
        """
        # Ensure client is initialized before processing request
        if not self._is_initialized:
            await self.startup()

        if self._client is None:
            raise ValueError("OpenAI client not initialized")

        # Step 1: Standardize message format to OpenAI spec
        prepared_messages = self._prepare_messages(messages)

        # Step 2: Aggregate parameters (kwargs take precedence over config)
        # Only include non-None values to use API defaults when appropriate
        base_params = {
            "model": self.config.model_name,
            "messages": prepared_messages,
            "stream": stream,
            "temperature": kwargs.get('temperature', self.config.temperature),
            "max_tokens": kwargs.get('max_tokens', self.config.max_tokens),
            "top_p": kwargs.get('top_p', self.config.top_p),
            "frequency_penalty": kwargs.get('frequency_penalty', self.config.frequency_penalty),
            "presence_penalty": kwargs.get('presence_penalty', self.config.presence_penalty),
            "tools": kwargs.get('tools'),
            "tool_choice": kwargs.get('tool_choice'),
            "response_format": kwargs.get('response_format'),  # Support for JSON mode
        }
        
        # Filter out None values to use API defaults
        api_params = {k: v for k, v in base_params.items() if v is not None}
        
        # Remove penalty parameters at neutral (0.0) values — some models (e.g.
        # Grok, Claude) reject these parameters entirely, and 0.0 means "no
        # penalty applied" which matches the API default behavior.
        for _key in ('frequency_penalty', 'presence_penalty'):
            if _key in api_params and api_params[_key] == 0.0:
                del api_params[_key]

        try:
            logger.debug(
                f"🚀 Sending request to [{self.config.model_name}] - Stream: {stream}"
                f" | response_format={api_params.get('response_format')!r}"
            )
            response = await self._client.chat.completions.create(**api_params)

            # Step 3: Enhanced logging for non-streaming responses
            if not stream:
                self._log_completion_info(response)  # type: ignore
            
            return response
            
        except APIError as e:
            logger.error(f"❌ OpenAI API error: {e.code} | {e.message}")
            raise
        except Exception as e:
            logger.error(f"💥 Unexpected generation error: {str(e)}")
            raise

    def _prepare_messages(self, messages: list[dict[str, str]] | str) -> list[ChatCompletionMessageParam]:
        """Normalize input messages to conform to OpenAI message format.
        
        Converts string input to user message, validates role values, and
        ensures proper message structure for the API.
        
        Args:
            messages: Input messages (string or list of role/content dicts)
            
        Returns:
            List of ChatCompletionMessageParam objects conforming to OpenAI spec
            
        Notes:
            Automatically corrects invalid roles to 'user' with warning logging
        """
        # Convert single string to standard user message
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        
        # Validate and normalize message list
        processed_messages = []
        for msg in messages:
            # Get role with fallback to 'user'
            role = msg.get("role", "user")
            
            # Validate and correct non-standard roles
            if role not in ["system", "user", "assistant", "tool"]:
                logger.warning(f"Detected non-standard role: {role}, auto-corrected to 'user'")
                role = "user"
                
            # Create standardized message with empty content fallback
            processed_messages.append({
                "role": role,
                "content": msg.get("content", "")
            })
            
        return processed_messages  # type: ignore

    def _log_completion_info(self, response: ChatCompletion) -> None:
        """Log detailed completion metadata for monitoring and debugging.
        
        Tracks token usage, tool calls, and other important response metadata
        to provide visibility into API usage and costs.
        
        Args:
            response: Complete ChatCompletion response object from OpenAI API
        """
        # Log token usage metrics if available
        usage = response.usage
        if usage:
            logger.info(
                f"📊 Token usage - Prompt: {usage.prompt_tokens}, "
                f"Completion: {usage.completion_tokens}, "
                f"Total: {usage.total_tokens}"
            )
        
        # Log tool calls if present
        message = response.choices[0].message
        if message.tool_calls:
            tool_names = [tc.function.name for tc in message.tool_calls] # type: ignore
            logger.info(f"🛠️ Tool calls triggered: {tool_names}")

    @property
    def is_initialized(self) -> bool:
        """Check if client is properly initialized and ready for requests.
        
        Returns:
            True if client is initialized, False otherwise
        """
        return self._is_initialized