import torch
from functools import lru_cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import LocalLLMConfig, BaseLLM

class LocalLLM(BaseLLM):
    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.compiled_model = None

    async def startup(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map=self.config.device,
                torch_dtype=torch.float16 if 'cuda' in self.config.device else torch.float32
            ).eval()
            
            if self.config.compile_model:
                self.compiled_model = torch.compile(
                    self.model,
                    mode='max-autotune',
                    fullgraph=True
                )
                
        except (OSError, ValueError, ImportError) as e:
            self.ERROR_COUNTER.labels(provider='local').inc()
            raise RuntimeError(f"Model init failed: {str(e)}")
        except RuntimeError as e:
            self.ERROR_COUNTER.labels(provider='local').inc()
            if isinstance(e, torch.cuda.OutOfMemoryError):
                raise RuntimeError(f"CUDA out of memory: {str(e)}") from e
            raise RuntimeError(f"Runtime error during model init: {str(e)}") from e

    @lru_cache(maxsize=1000)
    async def generate(self, prompt: str, **kwargs) -> str:
        with self.LATENCY_HIST.labels(model_type='local').time():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device) # type: ignore
            model = self.compiled_model or self.model
            
            with torch.no_grad():
                outputs = model.generate( # type: ignore
                    **inputs,
                    max_new_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    **kwargs
                )
                
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True) # type: ignore