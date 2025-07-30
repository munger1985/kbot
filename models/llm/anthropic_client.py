"""Anthropic LLM client implementation."""

import anthropic
from loguru import logger
from typing import Any
from pydantic import Field
from models.llm.base import BaseLLM, LLMConfig
from core.config import settings


class AnthropicLLMConfig(LLMConfig):
    """Anthropic LLM configuration."""

    api_key: str = Field(..., description="Anthropic API key")
    model_name: str = Field(default="claude-3-opus-20240229", description="Default model to use")


class AnthropicClient(BaseLLM):
    """Anthropic LLM client."""

    def __init__(self, config: AnthropicLLMConfig) -> None:
        """Initialize Anthropic client.

        Args:
            config: Anthropic LLM configuration
        """
        super().__init__(config)
        self.config = config
        self.client = None

    async def startup(self) -> None:
        """Initialize the client."""
        self.client = anthropic.Anthropic(api_key=self.config.api_key)
        logger.info("Anthropic client initialized")

    async def shutdown(self) -> None:
        """Shutdown the client."""
        self.client = None
        logger.info("Anthropic client shutdown")

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Generated text

        Raises:
            RuntimeError: If client is not initialized
        """
        if self.client is None:
            raise RuntimeError("Anthropic client not initialized")

        # Used for health check
        if prompt == "":
            return ""
        
        # Set default values
        model = self.config.model_name
        max_tokens = max_tokens or settings["llm"]["max_tokens"]
        temperature = temperature or settings["llm"]["temperature"]

        try:
            # Create message
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs,
            )
            
            # Extract text from response
            return message.content[0].text
        except Exception as e:
            logger.exception(f"Error generating text with Anthropic: {e}")
            raise

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a chat response.

        Args:
            messages: List of messages
            model: Model name
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Chat response text

        Raises:
            RuntimeError: If client is not initialized
        """
        if self.client is None:
            raise RuntimeError("Anthropic client not initialized")

        # Set default values
        model = self.config.model_name
        max_tokens = max_tokens or settings["llm"]["max_tokens"]
        temperature = temperature or settings["llm"]["temperature"]

        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                # Map roles to Anthropic format
                if role == "system":
                    # For system messages, we'll use a special format
                    # that Anthropic recommends for system instructions
                    anthropic_messages.append({
                        "role": "user",
                        "content": f"<system>\n{content}\n</system>\n\nPlease acknowledge the above system instructions."
                    })
                    anthropic_messages.append({
                        "role": "assistant",
                        "content": "I'll follow the system instructions provided."
                    })
                else:
                    # For user and assistant messages, use them directly
                    anthropic_messages.append({
                        "role": role,
                        "content": content
                    })
            
            # Create message
            message = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=anthropic_messages,
                **kwargs,
            )
            
            # Extract text from response
            return message.content[0].text
        except Exception as e:
            logger.exception(f"Error generating chat response with Anthropic: {e}")
            raise