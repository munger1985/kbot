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
from modelPrepare.ociGenAIAPI import *
from modelPrepare.qwenAPI import QwenPlus
from modelPrepare.remoteAPI import *
import  llm_keys
device = "cuda" if torch.cuda.is_available() else "cpu"

#######  DEV #################
#http_prefix  is the url prefix for downloading the docs, is kbot server itself
http_prefix = 'http://150.230.37.250:8093/'
DOC_VIEWER_FLAG='N' ##Y|N,If config `Y`, it will use the http_doc_viewer to view the document, otherwise it will http_prefix to download the file.
http_doc_viewer = "http://150.230.37.250/HysunDocuViewer/?src=http://150.230.37.250:8080/"
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector/vector@129.159.40.144:1521/orclpdb1"
ORACLE_AI_VECTOR_CONNECTION_STRING="vectorxx_dev/xxxpassword@165.1.65.228:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"

# ADW AI Vector Search
ADW_VECTOR_SEARCH_USER = "vectoxxr_dev"
ADW_VECTOR_SEARCH_PASSWORD = "xxxpassword"
ADW_VECTOR_SEARCH_DSN = "kbotadw23ai_medium"
ADW_VECTOR_SEARCH_WALLET_LOCATION = "/home/ubuntu/kbot/keys/adwvectordb"  # Wallet zip文件解压缩后的目录
ADW_VECTOR_SEARCH_WALLET_PASSWORD = "xxxpassword"

OCI_OPEN_SEARCH_URL="https://amaaaaaaak7gbrialufa2y2ozyzfflp5ox2g5roy5aw5b6f7h3j2ee5z2zva.opensearch.ap-melbourne-1.oci.oraclecloud.com:9200"
OCI_OPEN_SEARCH_USER='opc'
OCI_OPEN_SEARCH_PASSWD='xxxpassword'

####### opensearch docker in ai-dev
OCI_OPEN_SEARCH_URL="https://localhost:9200"
OCI_OPEN_SEARCH_USER='admin'
OCI_OPEN_SEARCH_PASSWD='admin'


# HeatWave VectorStore
HEATWAVE_CONNECTION_PARAMS = {
    "user": "admin",
    "password": "xxxpassword",
    "host": "192.9.158.173",
    "database": "kbot_dev",
}
HEATWAVE_VECTOR_STORE_POOL_NAME = "heatwave_vectorstore_pool"
HEATWAVE_VECTOR_STORE_POOL_SIZE = 6

#######  PRD #################
#http_prefix = 'https://prd.oracle.k8scloud.site/'
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector_prd/xxxpassword@165.1.65.228:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"


#######  API auth method   #####################
auth_type= 'none'
# auth_type= 'API_KEY'
auth_type= 'INSTANCE_PRINCIPAL'


#######  the knowledge base root directory    #####################
# KB_ROOT_PATH = '/home/ubuntu/kbroot'
#######  if use auto, the kbroot will be automatically set  in the same directory where kbot/ locates   ######################
KB_ROOT_PATH = 'auto'

#######  sqlite parent directory    #######################################
sqlite_path = KB_ROOT_PATH



#######  Vector Store setting    #######################################
score_threshold =  0.6
vector_store_limit= 10

#######  Reranker model setting    #######################################
#rerankerModel = 'bgeReranker'
#BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-large"  #BAAI/bge-reranker-large
BGE_RERANKER="BAAI/bge-reranker-v2-m3"
# disableReranker, ociCohereReranker,cohereReranker
rerankerModel = 'ociCohereReranker'
reranker_topk= 2

#######  the memory window for chat history   #####################
history_k = 5

#######  Embedding model setting    #######################################
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
#e5_large_v2 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/e5-large-v2", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-large-zh-v1.5", model_kwargs={'device': device})
# e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
# bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
# bge_large_zh_v15 = HuggingFaceEmbeddings(model_name=, model_kwargs={'device': device})


EMBEDDING_DICT = {
    'bge_m3': load_embedding_model("BAAI/bge-m3",device),
    'bge_large_zh_v15': load_embedding_model("BAAI/bge-large-zh-v1.5",device),
    'e5_large_v2': load_embedding_model("intfloat/e5-large-v2",device),
    'OCI-cohere.embed-multilingual-v3.0': genaiEmbedding,
    'cohere_embed':cohereEmbedding,
    'text-embedding-3-large':chatgptAPI.openaiEmbeddings,
    'cohereRemoteVM': remoteEmbedding(model_name='OCI-cohere.embed-multilingual-v3.0',
                                      openai_api_key='xxxx',
                                      openai_api_base='http://localhost:8093/v1')
}

#######  llm model setting          #######################################
# use default authN method   INSTANCE_PRINCIPAL

MODEL_DICT = {
    'NoneLLM': 'NoneLLM',
    ######################      API models        #############################################
    'OCI-cohere.command-r-16k': ociCMDR,
    'OCI-cohere.command-r-plus': ociCMDRPlus,
    'OCI-cohere.command-r-plus082024': ociCMDRPlus082024,
    'OCI-cohere.command-r082024': ociCMDR082024,
    'OCI-meta.llama-3.1-70b-instruct': ociGenAILlama3_1_70B,
    'OCI-meta.llama-3.1-405b-instruct': ociGenAILlama3_1_405B,
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
    'DeepSeek_V3':  remoteModel('deepseek-chat','https://api.deepseek.com/v1',llm_keys.deepseek_api_key,512,0),
    'DeepSeek_R1':  remoteModel('deepseek-reasoner','https://api.deepseek.com/v1',llm_keys.deepseek_api_key,512,0),
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

########  VECTOR_STORE Types  #######################################
VECTOR_STORE_DICT = [
    'faiss',
    'oracle',
    'adb',
    'opensearch',
    'heatwave'
]
########## summary model #################################
SUMMARY_MODEL ="OCI-cohere.command-r-plus082024"

######## Select AI ########
selectai_pool = None
#selectai_pool = oracledb.create_pool(
#    user="WKSP_XH",
#    password="xxxpassword",
#    dsn="cntech_medium",
#    wallet_location="/home/ubuntu/qq/Keys",
#    config_dir="/home/ubuntu/qq/Keys",
#    wallet_password="xxxpassword",
#)