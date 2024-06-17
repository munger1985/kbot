from typing import Any, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

import torch

from loguru import logger
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

class QwenLLM(LLM):
    # 基于本地 Qwen 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model_path: str = 'Qwen/Qwen1.5-7B-Chat'):
        super().__init__()
        logger.info("正在从本地加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        logger.info("完成本地模型的加载")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="pt").to('cuda')
        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=8192)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    @property
    def _llm_type(self) -> str:
        return "Qwen2_LLM"