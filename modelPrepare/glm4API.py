from typing import Any, List, Mapping, Optional, Dict
from zhipuai import ZhipuAI
import llm_keys
from loguru import logger
from langchain_core.language_models import LLM

query_llm = ZhipuAI(api_key=llm_keys.zhipu_api_key)

class GLM4(LLM):
    '''
    local chatglm
    '''
    max_token: int = 8192
    do_sample: bool = False
    temperature: float = 0.1
    top_p = 0.7
    tokenizer: object = None
    model: object = None
    history: List = []
    tool_names: List = []
    has_search: bool = False
    model_name: str = 'glm4'

    def __init__(self, **kwargs):
        # logger.info("##llm kwargs",kwargs)
        super().__init__(**kwargs)


    @property
    def _llm_type(self) -> str:
        return "GLM4"



    def _call(self,
              prompt: str,
              history: List = [],
              stop: Optional[List[str]] = ["<|user|>"]):
        # logger.info("======prompt:")
        # logger.info(prompt)
        # logger.info("======")

        # system_message_content = re.findall(r"SystemMessage\(content='(.*?)'\)", prompt)
        # ai_message_content = re.findall(r"AIMessage\(content='(.*?)'\)", prompt)

        # 抽取HumanMessage的内容
        # human_message_content = re.findall(r"HumanMessage\(content='(.*?)'\)", prompt)

        # logger.info(f"SystemMessage内容：{system_message_content}")
        # logger.info(f"HumanMessage内容：{human_message_content}")
        response = query_llm.chat.completions.create(
            model="glm-4",
            messages=[
                # {
                #     "role": "system",
                #     "content": system_message_content,
                # },
                {"role": "user", "content": prompt},
            ],
            top_p=self.top_p,
            temperature=self.temperature,
            ##max_tokens=8096,
        )
        llm_response_output = response.choices[0].message.content
        # logger.info("#response:",response)
        return llm_response_output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        logger.info("##self.max_token,self.temperature,self.top_p,self.do_sample:", self.max_token, self.temperature,
              self.top_p, self.do_sample)
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }