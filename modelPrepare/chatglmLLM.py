from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, List, Mapping, Optional, Dict

from loguru import logger
from langchain_core.language_models import LLM

class ChatGLM3(LLM):
    '''
    local chatglm
    '''
    max_token: int = 8192
    do_sample: bool = False
    temperature: float = 0.1
    top_p = 0.8
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False
    model_name: str

    def __init__(self, **kwargs):
        # logger.info("##llm kwargs",kwargs)
        super().__init__(**kwargs)
        self.model_name = kwargs['model_name']
        self.load_model(model_name_or_path=self.model_name)

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"

    def load_model(self, model_name_or_path=None):
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        # self.model = AutoModel.from_pretrained(
        #    model_name_or_path, config=model_config, trust_remote_code=True
        # ).half().cuda()
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True
        ).cuda()

    def _call(self,
              prompt: str,
              history: List = [],
              stop: Optional[List[str]] = ["<|user|>"]):
        # logger.info("======prompt:")
        # logger.info(prompt)
        # logger.info("======")
        query = prompt
        logger.info("======history:")
        logger.info(self.history)
        logger.info("======")
        _, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
        )
        response = self.history[-1]["content"]
        self.history = []
        # logger.info("#response:",response)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        logger.info("##self.max_token,self.temperature,self.top_p,self.do_sample:", self.max_token, self.temperature,
              self.top_p, self.do_sample)
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
        }