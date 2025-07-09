import openai
from functools import lru_cache
from .base import CloudLLMConfig, BaseLLM

class CloudLLM(BaseLLM):
    PROVIDER_MAP = {
        'openai': openai.OpenAI,
        'azure': openai.AzureOpenAI
    }
    
    def __init__(self, config: CloudLLMConfig):
        self.config = config
        self.client = None

    async def startup(self):
        try:
            client_class = self.PROVIDER_MAP.get(self.config.provider)
            if not client_class:
                raise ValueError(f"Unsupported provider: {self.config.provider}")
                
            self.client = client_class(
                api_key=self.config.api_key,
                base_url=self.config.endpoint,
                timeout=self.config.timeout
            )
        except Exception as e:
            self.ERROR_COUNTER.labels(provider=self.config.provider).inc()
            raise

    @lru_cache(maxsize=1000)
    async def generate(self, prompt: str, **kwargs) -> str:
        with self.LATENCY_HIST.labels(model_type='cloud').time():
            try:
                response = await self.client.completions.create( # type: ignore
                    model=self.config.model_name,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    **kwargs
                )
                return response.choices[0].text
            except Exception as e:
                self.ERROR_COUNTER.labels(provider=self.config.provider).inc()
                raise