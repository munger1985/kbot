from modelPrepare import chatgptAPI
from modelPrepare import glm4API
from modelPrepare import qianfanAPI
from modelPrepare import sparkAPI
from modelPrepare.aquaAPI import AIQuickActions
from modelPrepare import ociGenAIAPI
from modelPrepare.huggingfaceLLM import HfLLM
from modelPrepare.qwenAPI import QwenPlus
from modelPrepare.qwenLLM import QwenLLM
from modelPrepare.remoteAPI import *

import torch
from lazy_load import lazy_func
from langchain_core.language_models import LLM
from FlagEmbedding import FlagReranker
from config import config

#######  this module is for complex model loading
device = "cuda" if torch.cuda.is_available() else "cpu"

bgeRerankerModel = None


@lazy_func
def load_bge_reranker(model_name_or_path):
    global bgeRerankerModel
    if bgeRerankerModel:
        return bgeRerankerModel
    else:
        bgeRerankerModel = FlagReranker(model_name_or_path,
                                        use_fp16=True)  # Setting use_fp16 to True speeds up computation with a slight performance degradation
        return bgeRerankerModel


@lazy_func
def load_llm_model(model_id_or_path: str = "") -> LLM:
    """

    :param model_alias:  custom names
    :param model_id_or_path:  fill in hf model id or local path
    :return:
    """

    return HfLLM(model_id_or_path)


MODEL_DICT = {
    'NoneLLM': 'NoneLLM',
    ######################      API models        #############################################
    # 'OCI-cohere.command-r-16k': ociGenAIAPI.ociCMDR,
    # 'OCI-cohere.command-r-plus': ociCMDRPlus,
    'OCI-cohere.command-a-03-2025': ociGenAIAPI.ociCMDA032025,
    'OCI-cohere.command-r-plus-08-2024': ociGenAIAPI.ociCMDRPlus082024,
    'OCI-cohere.command-r-08-2024': ociGenAIAPI.ociCMDR082024,
    'OCI-meta.llama-3.1-70b-instruct': ociGenAIAPI.ociGenAILlama3_1_70B,
    'OCI-meta.llama-3.1-405b-instruct': ociGenAIAPI.ociGenAILlama3_1_405B,
    'OCI-meta.llama-3.2-90b-vision-instruct': ociGenAIAPI.ociGenAILlama3_2_90b,
    'OCI-meta.llama-3.3-70b-instruct': ociGenAIAPI.ociGenAILlama3_3_70b,
    'OCI-meta.llama-4-marverick-17b': ociGenAIAPI.ociGenAILlama4_maverick_17b,
    'OCI-meta.llama-4-scout-17b': ociGenAIAPI.ociGenAILlama4_scout_17b,
    'Qwen2-7B-Instruct': remoteModel('Qwen2-7B-Instruct', 'http://146.235.226.110:8098/v1', '123456', 512, 0),
    'Qwen2-7B-Instruct_Https': remoteModel('Qwen2-7B-Instruct', 'https://chat.oracle.k8scloud.site:8098/v1', '123456',
                                           512, 0),
    'Llama-3-8B-Instruct': remoteModel('/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct',
                                       'http://146.235.226.110:8098/v1', '123456', 256, 0),
    'Llama-3-70B-Instruct': remoteModel('meta-llama/Meta-Llama-3-70B-Instruct', 'http://141.147.8.181:8098/v1',
                                        '123456', 256, 0),
    'ooba': remoteModel('gpt-3.5-turbo', 'http://10.145.141.77:5000/v1', '123456', 256, 0),
    'mistral-aqua': AIQuickActions(
        endpoint='https://modeldeployment.us-sanjose-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.us-sanjose-1.amaaaaaaak7gbriae3l7phztdfyhlwjqjgkfcqxwhu62gs5vur4sqryi5nvq'),
    '星火大模型3.0': sparkAPI.SparkLLM(),
    'ChatGLM4': glm4API.glm4,
    'ChatGPT': chatgptAPI.gpt3,
    'Qwen-plus': QwenPlus(),
    'DeepSeek_V3': remoteModel('deepseek-chat', 'https://api.deepseek.com/v1', config.deepseek_api_key, 512, 0),
    'DeepSeek_R1': remoteModel('deepseek-reasoner', 'https://api.deepseek.com/v1', config.deepseek_api_key, 512, 0),
    'ERNIE-4.0-8K-Latest': qianfanAPI.qianfanLLM('ERNIE-4.0-8K-Latest'),
    ######################      local models      ###########################################
    ###   format 1) : local path
    # e.g.  'llama-2-7b-chat':   load_llm_modeeibccbhvltvuieifgggchbujrfnttjnvvbchbvdhngfb
    # l("/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf"),
    # 'Llama-2-7B-Chat': load_llm_model('Llama-2-7B-Chat','/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf'),
    # 'Gemma-7B-IT': load_llm_model('Gemma-7B-IT','/home/ubuntu/ChatGPT/Models/google/gemma-7b-it'),
    # 'Mistral-7B-Instruct-v0.2': load_llm_model('Mistral','/home/ubuntu/ChatGPT/Models/mistral/Mistral-7B-Instruct-v0.2'),
    # 'Qwen1.5-7B-Chat': load_llm_model('Qwen','/home/ubuntu/ChatGPT/Models/Qwen/Qwen1.5-7B-Chat'),
    # 'Llama-3-8B-Instruct': load_llm_model('Llama3','/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct'),
    # 'Qwen3-8B': QwenLLM('Qwen/Qwen3-8B'),
    # 'Qwen3-8Bhf': HfLLM('Qwen/Qwen3-8B'),

    ###   format 2) : huggingface model id
    # 'Llama-2-7B-Chat': load_llm_model( 'meta-llama/Llama-2-7b-chat-hf'),
    # 'Gemma-7B-IT': load_llm_model( 'google/gemma-7b-it'),
    # 'Mistral-7B-Instruct-v0.2': load_llm_model( 'mistralai/Mistral-7B-Instruct-v0.2'),
    # 'Qwen3-8B': load_llm_model('Qwen/Qwen3-8B'),

}

