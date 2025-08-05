"""Hugging Face LLM client implementation."""

from typing import Any
import requests
from pydantic import Field
from loguru import logger
from models.llm.base import BaseLLM, LLMConfig
from core.config import settings


class HuggingFaceLLMConfig(LLMConfig):
    """Hugging Face LLM configuration."""

    api_key: str = Field(..., description="Hugging Face API token")
    model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2", description="Default model to use")
    api_url: str = Field(default="https://api-inference.huggingface.co/models", description="Hugging Face API URL")


class HuggingFaceClient(BaseLLM):
    """Hugging Face LLM client."""

    def __init__(self, config: HuggingFaceLLMConfig) -> None:
        """Initialize Hugging Face client.

        Args:
            config: Hugging Face LLM configuration
        """
        super().__init__(config)
        self.config = config
        self.headers = None
        self._is_initialized = False

    async def startup(self) -> None:
        """Initialize the client."""
        self.headers = {"Authorization": f"Bearer {self.config.api_key}"}
        self._is_initialized = True
        logger.info("Hugging Face client initialized")

    async def shutdown(self) -> None:
        """Shutdown the client."""
        self.headers = None
        self._is_initialized = False
        logger.info("Hugging Face client shutdown")

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
        if self.headers is None:
            raise RuntimeError("Hugging Face client not initialized")
        
        # Set default values
        model = self.config.model_name
        
        # Prepare payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens or settings["llm"]["max_tokens"],
                "temperature": temperature or settings["llm"]["temperature"],
                "return_full_text": False,
                **kwargs
            }
        }

        try:
            # Make API request
            response = requests.post(
                f"{self.config.api_url}/{model}",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                if "generated_text" in result[0]:
                    return result[0]["generated_text"]
                else:
                    return str(result[0])
            else:
                return str(result)
        except Exception as e:
            logger.exception(f"Error generating text with Hugging Face: {e}")
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
        if self.headers is None:
            raise RuntimeError("Hugging Face client not initialized")

        # Set default values
        model = self.config.model_name
        
        # Format messages into a prompt
        prompt = self._format_chat_messages(messages)
        
        # Use the generate method to get a response
        return await self.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens or settings["llm"]["max_tokens"],
            temperature=temperature or settings["llm"]["temperature"],
            **kwargs
        )

    def _format_chat_messages(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages into a prompt.

        Args:
            messages: list of messages

        Returns:
            Formatted prompt
        """
        formatted_messages = []
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_messages.append(f"<|system|>\n{content}")
            elif role == "user":
                formatted_messages.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted_messages.append(f"<|assistant|>\n{content}")
            else:
                formatted_messages.append(f"<|{role}|>\n{content}")
        
        # Add the final assistant prompt
        formatted_messages.append("<|assistant|>")
        
        return "\n".join(formatted_messages)
    