import logging
from typing import Any, List, Mapping, Optional

import requests
import random
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

logger = logging.getLogger(__name__)
from loguru import logger

import _thread as thread
import base64
import datetime
import hashlib
import hmac
import json
from urllib.parse import urlparse
import ssl
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
import websocket  # 使用websocket_client

'''配置授权
'''
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from ads.common.auth import default_signer
from ads.config import OCI_RESOURCE_PRINCIPAL_VERSION
import ads

DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 0.3

_END_MARKER = "[DONE]"


class StreamingException(Exception):
    INTERNAL_ERROR_MESSAGE = "Internal error"
    INTERNAL_ERROR_CODE = 500

    def __init__(self, json_response: dict):
        if json_response and isinstance(json_response, dict):
            if 'message' in json_response:
                self.message = json_response['message']
            else:
                self.message = self.INTERNAL_ERROR_MESSAGE
            if 'status_code' in json_response:
                self.status_code = json_response['status_code']
            else:
                self.status_code = self.INTERNAL_ERROR_CODE

        else:
            self.message = self.INTERNAL_ERROR_MESSAGE
            self.status_code = self.INTERNAL_ERROR_CODE

        super().__init__(f"{self.status_code}: {self.message}")


class ModelInvoker:

    def __init__(
            self,
            endpoint: str,
            prompt: str,
            params: dict,
            retries: int = DEFAULT_RETRIES,
            backoff_factor: float = DEFAULT_BACKOFF_FACTOR
    ):
        # TODO: This should accept all types of authentication
        # if OCI_RESOURCE_PRINCIPAL_VERSION:
        ads.set_auth("instance_principal")
        self.auth = default_signer()
        self.endpoint = endpoint
        self.prompt = prompt
        self.params = params
        self.retries = retries
        self.backoff_factor = backoff_factor
        self.session = self._create_session_with_retries(retries, backoff_factor)

    def _create_session_with_retries(
            self, retries: int, backoff_factor: float
    ) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=backoff_factor,
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def invoke(self):
        headers = {
            "Content-Type": "application/json",
            "enable-streaming": "true",
        }

        try:
            response = self.session.post(
                self.endpoint,
                auth=self.auth["signer"],
                headers=headers,
                json={"prompt": self.prompt, **self.params},
                stream=True,
            )
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    yield line.decode("utf-8")

        except requests.RequestException as e:
            yield json.dumps({"object": "error", "message": str(e)})


def invoke_streaming_md(endpoint: str, prompt: str, params: dict):
    params['model'] = 'odsc-llm'
    params['stream'] = False
    model_invoker = ModelInvoker(
        endpoint=endpoint,
        prompt=prompt,
        params=params,
    )
    res = ""
    for item in model_invoker.invoke():
        if item.startswith("data"):
            if _END_MARKER in item:
                continue
            item_json = json.loads(item[len(_END_MARKER):])
        else:
            item_json = json.loads(item)

        if 'choices' in item_json:
            res+= item_json['choices'][0]['text']
            # logger.info(item_json['choices'][0]['text'], end='')
        else:
            raise StreamingException(item_json)
    return res

class AIQuickActions(LLM):
    user_id: str  =   random.randint(0, 111119)
    text  =  []
    model_kwargs: Optional[dict] = None
    answer :str   = ""
    endpoint:str
    """Key word arguments to pass to the model."""
    # max_token: int = 4000
    """Max token allowed to pass to the model.在真实应用中考虑启用"""
    # temperature: float = 0.75
    """LLM model temperature from 0 to 10.在真实应用中考虑启用"""
    # history: List[List] = []
    """History of the conversation.在真实应用中可以考虑是否启用"""
    # top_p: float = 0.85
    """Top P for nucleus sampling from 0 to 1.在真实应用中考虑启用"""
    # with_history: bool = False
    """Whether to use history or not.在真实应用中考虑启用"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.endpoint = kwargs.get("endpoint")
    def  getText(self, role,content):
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        self.text.append(jsoncon)
        return self.text
    @property
    def _llm_type(self) -> str:
        return "Aqua"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"user_id": self.user_id},
            **{"model_kwargs": _model_kwargs},
        }

    # 收到websocket消息的处理
    def on_message(self,ws, message):
        # logger.info(message)
        data = json.loads(message)
        code = data['header']['code']
        if code != 0:
            logger.info(f'请求错误: {code}, {data}')
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            logger.info(content, end="")
            self.answer += content
            if status == 2:
                ws.close()
    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        self.answer=""
        try:
            self.answer=  invoke_streaming_md(
            endpoint= self.endpoint+"/predict",
            prompt=prompt,
            params={"max_tokens": 500, "temperature": 0.7, "top_k": 50, "top_p": 1, "stop": [], "frequency_penalty": 0,
                    "presence_penalty": 0})
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"SparkLLM response: ")

        return self.answer