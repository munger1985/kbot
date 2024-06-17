# coding = utf-8
import torch, transformers
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline, Cohere
from transformers import AutoTokenizer
from lazy_load import lazy_func
from langchain_community.llms.chatglm3 import ChatGLM3 as ChatGlm3Remote
from langchain_community.llms import Tongyi
from modelPrepare import  qwenLLM
import llm_keys
from langchain_core.language_models import LLM

@lazy_func
def load_llm_model(model_alias='hf_pipeline', model_path:str= "") -> LLM:
    """

    :param model_alias:  custom names
    :param model_path:  fill in hf model id or local path
    :return:
    """
    if model_alias == "Cohere":
        llm = Cohere(model="command-r-plus",
                    cohere_api_key=llm_keys.cohere_api_key, 
                    temperature=0,
                    max_tokens=102400)
    elif model_alias == 'Chatglm3Remote':
        endpoint_url = "http://138.2.237.212:8000/v1/chat/completions"
        llm = ChatGlm3Remote(
            endpoint_url=endpoint_url,
            max_tokens=4000,
            # prefix_messages=messages,
            top_p=0.9,
            temperature=0,
            model='THUDM/chatglm3-6b'
        )
    elif model_alias == 'Qwen-plus':
        text_gen_params = {
            'temperature'   : 0,
            #'max_tokens': 20480,
            #'top_p':0.9
        }
        #qwen-turbo,qwen-plus,qwen-max,qwen-max-longcontext
        #https://help.aliyun.com/zh/dashscope/developer-reference/model-introduction?spm=a2c4g.11186623.0.0.1568140bOMYZzf
        #https://help.aliyun.com/zh/dashscope/developer-reference/tongyi-qianwen-7b-14b-72b-api-detailes?spm=a2c4g.11186623.0.0.6ec95018FuKzlW
        llm = Tongyi(model_name="qwen-plus",
                    dashscope_api_key=llm_keys.qwen_api_key,
                    model_kwargs=text_gen_params)
    elif model_alias == 'Qwen':
        llm = qwenLLM.QwenLLM(model_path=model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,trust_remote_code=True
        )
        global streamer
        streamer = None
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
        pipeline = transformers.pipeline(
            "text-generation",  # task
            model=model_path,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            # max_length=2048,
            max_length=4096,
            #temperature=0.5,
            do_sample=False,
            # top_k=30,
            truncation=True,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            #model_kwargs={'temperature': 0.1}
            streamer=streamer,
        )


        #model_kwargs={"temperature": 0.1, "max_length":2048}
        # pipeline
        llm = HuggingFacePipeline(pipeline=pipeline)
    return llm
