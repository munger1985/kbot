from config import config
from langchain_community.llms import QianfanLLMEndpoint
import os

os.environ["QIANFAN_AK"] = config.qianfan_ak
os.environ["QIANFAN_SK"] = config.qianfan_sk

def qianfanLLM(model_name="ERNIE-4.0-8K-Latest",temperature=0.1):
    llm = QianfanLLMEndpoint(
        model=model_name,
        endpoint="eb-instant",
        #qianfan_ak=config.qianfan_ak,
        #qianfan_sk=config.qianfan_sk,
        temperature=temperature,
        #max_tokens=max_tokens
    )
    return llm