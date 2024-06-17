from typing import Any, List, Mapping, Optional

import requests
import random
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
import llm_keys
import websocket  # 使用websocket_client
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LLM

'''配置授权
'''

appid=llm_keys.xinghuo_appid
api_secret=llm_keys.xinghuo_api_secret
api_key=llm_keys.xinghuo_api_key
#用于配置大模型版本，默认“general/generalv2”
domain = llm_keys.xinghuo_domain
#云端环境的服务地址
Spark_url = llm_keys.xinghuo_spark_url



class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Spark_url):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = urlparse(Spark_url).netloc
        self.path = urlparse(Spark_url).path
        self.Spark_url = Spark_url

    # 生成url
    def create_url(self):
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + self.path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')

        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": self.host
        }
        # 拼接鉴权参数，生成url
        url = self.Spark_url + '?' + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url


# 收到websocket错误的处理
def on_error(ws, error):
    logger.info("### error:", error)


# 收到websocket关闭的处理
def on_close(ws,one,two):
    logger.info(" ")


# 收到websocket连接建立的处理
def on_open(ws):
    thread.start_new_thread(run, (ws,))


def run(ws, *args):
    data = json.dumps(gen_params(appid=ws.appid, domain= ws.domain,question=ws.question))
    ws.send(data)


# # 收到websocket消息的处理
# def on_message(ws, message):
#     # logger.info(message)
#     data = json.loads(message)
#     code = data['header']['code']
#     if code != 0:
#         logger.info(f'请求错误: {code}, {data}')
#         ws.close()
#     else:
#         choices = data["payload"]["choices"]
#         status = choices["status"]
#         content = choices["text"][0]["content"]
#         logger.info(content,end ="")
#         answer += content
#         # logger.info(1)
#         if status == 2:
#             ws.close()
text=[]
# def getText(role,content):
#     jsoncon = {}
#     jsoncon["role"] = role
#     jsoncon["content"] = content
#     text.append(jsoncon)
#     return text

def gen_params(appid, domain,question):
    """
    通过appid和用户的提问来生成请参数
    """
    data = {
        "header": {
            "app_id": appid,
            "uid": "1234"
        },
        "parameter": {
            "chat": {
                "domain": domain,
                "temperature": 0.5,
                "max_tokens": 2048
            }
        },
        "payload": {
            "message": {
                "text": question
            }
        }
    }
    return data


def main(appid, api_key, api_secret, Spark_url,domain, question,sparkllm):
    # logger.info("星火:")
    wsParam = Ws_Param(appid, api_key, api_secret, Spark_url)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=sparkllm.on_message, on_error=on_error, on_close=on_close, on_open=on_open)
    ws.appid = appid
    ws.question = question
    ws.domain = domain
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})






def getlength(text):
    length = 0
    for content in text:
        temp = content["content"]
        leng = len(temp)
        length += leng
    return length

def checklen(text):
    while (getlength(text) > 8000):
        del text[0]
    return text


class SparkLLM(LLM):
    """Define the custom LLM wrapper for Xunfei SparkLLM to get support of LangChain
    """
    user_id: str  =   random.randint(0, 111119)
    text  =  []
    """Endpoint URL to use.此URL指向部署的调用星火大模型的FastAPI接口地址"""
    model_kwargs: Optional[dict] = None
    answer :str   = ""
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
    def  getText(self, role,content):
        jsoncon = {}
        jsoncon["role"] = role
        jsoncon["content"] = content
        self.text.append(jsoncon)
        return self.text
    @property
    def _llm_type(self) -> str:
        return "SparkLLM"

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
        question = checklen(self.getText("user",prompt))
        # call api
        try:
             main(appid, api_key, api_secret, Spark_url, domain, question,self)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        logger.debug(f"SparkLLM response: ")

        return self.answer