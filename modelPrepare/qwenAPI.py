import llm_keys
from langchain_community.llms import Tongyi

# qwen-turbo,qwen-plus,qwen-max,qwen-max-longcontext
# https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction?spm=a2c4g.11186623.0.0.1568140bOMYZzf
# https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes?spm=a2c4g.11186623.0.0.6ec95018FuKzlW


def QwenPlus(model_name="qwen-plus",temperature=0, max_tokens=2000,top_p=0.9):
    text_gen_params = {
        'temperature': temperature,
        #'max_tokens': max_tokens,
        #'top_p':top_p
    }
    llm = Tongyi(model_name=model_name,
                 dashscope_api_key=llm_keys.qwen_api_key,
                 model_kwargs=text_gen_params)
    return llm
