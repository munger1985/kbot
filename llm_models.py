import torch
from langchain_community.embeddings import HuggingFaceEmbeddings

from modelPrepare import cohereAPI
from llm_prepare import load_llm_model,load_embedding_model
from modelPrepare import chatgptAPI
from modelPrepare import glm4API
from modelPrepare import qianfanAPI
from modelPrepare import sparkAPI
from modelPrepare.aquaAPI import AIQuickActions
from modelPrepare.cohereAPI import cohereEmbedding
from modelPrepare  import  ociGenAIAPI     
from modelPrepare.qwenAPI import QwenPlus
from modelPrepare.remoteAPI import *
from config import config
device = "cuda" if torch.cuda.is_available() else "cpu"






MODEL_DICT = {
    'NoneLLM': 'NoneLLM',
    ######################      API models        #############################################
    # 'OCI-cohere.command-r-16k': ociGenAIAPI.ociCMDR,
    # 'OCI-cohere.command-r-plus': ociCMDRPlus,
    'OCI-cohere.command-r-plus082024': ociGenAIAPI.ociCMDRPlus082024,
    'OCI-cohere.command-r082024': ociGenAIAPI.ociCMDR082024,
    'OCI-meta.llama-3.1-70b-instruct': ociGenAIAPI.ociGenAILlama3_1_70B,
    'OCI-meta.llama-3.1-405b-instruct': ociGenAIAPI.ociGenAILlama3_1_405B,
    'Qwen2-7B-Instruct': remoteModel('Qwen2-7B-Instruct','http://146.235.226.110:8098/v1','123456',512,0),
    'Qwen2-7B-Instruct_Https': remoteModel('Qwen2-7B-Instruct','https://chat.oracle.k8scloud.site:8098/v1','123456',512,0),
    'Llama-3-8B-Instruct':  remoteModel('/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct','http://146.235.226.110:8098/v1','123456',256,0),
    'Llama-3-70B-Instruct':  remoteModel('meta-llama/Meta-Llama-3-70B-Instruct','http://141.147.8.181:8098/v1','123456',256,0),
    'ooba':  remoteModel('gpt-3.5-turbo','http://10.145.141.77:5000/v1','123456',256,0),
    'mistral-aqua': AIQuickActions(endpoint='https://modeldeployment.us-sanjose-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.us-sanjose-1.amaaaaaaak7gbriae3l7phztdfyhlwjqjgkfcqxwhu62gs5vur4sqryi5nvq'),
    '星火大模型3.0': sparkAPI.SparkLLM(),
    'ChatGLM4':  glm4API.GLM4(),
    'ChatGPT' : chatgptAPI.gpt3,
    'Qwen-plus': QwenPlus(),
    'DeepSeek_V3':  remoteModel('deepseek-chat','https://api.deepseek.com/v1',config.deepseek_api_key,512,0),
    'DeepSeek_R1':  remoteModel('deepseek-reasoner','https://api.deepseek.com/v1',config.deepseek_api_key,512,0),
    #'Cohere-CommandR+': cohereAPI.commandRPlus(),
    'ERNIE-4.0-8K-Latest':qianfanAPI.qianfanLLM('ERNIE-4.0-8K-Latest'),
    ######################      local models      ###########################################
    ###   format 1) : local path
    # e.g.  'llama-2-7b-chat':   load_llm_model("/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf"),
    # 'Llama-2-7B-Chat': load_llm_model('Llama-2-7B-Chat','/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf'),
    #'Gemma-7B-IT': load_llm_model('Gemma-7B-IT','/home/ubuntu/ChatGPT/Models/google/gemma-7b-it'),
    #'Mistral-7B-Instruct-v0.2': load_llm_model('Mistral','/home/ubuntu/ChatGPT/Models/mistral/Mistral-7B-Instruct-v0.2'),
    #'Qwen1.5-7B-Chat': load_llm_model('Qwen','/home/ubuntu/ChatGPT/Models/Qwen/Qwen1.5-7B-Chat'),
    #'Llama-3-8B-Instruct': load_llm_model('Llama3','/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct'),

    ###   format 2) : huggingface model id
    #'Llama-2-7B-Chat': load_llm_model('Llama2','meta-llama/Llama-2-7b-chat-hf'),
    #'Gemma-7B-IT': load_llm_model('Gemma','google/gemma-7b-it'),
    #'Mistral-7B-Instruct-v0.2': load_llm_model('Mistral','mistralai/Mistral-7B-Instruct-v0.2'),
    'Qwen2-1.5B-Instruct': load_llm_model('Qwen','Qwen/Qwen2-1.5B-Instruct'),

    ###   format 3) : llm_prepare.py special configured model, sometimes we need special model params
    #'ChatGLM3-6B': load_llm_model(model_alias= 'chatglm3')
    # 'Chatglm3Remote': load_llm_model(model_alias = 'chatglm3Remote'),

}

