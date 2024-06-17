from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from modelPrepare.cohereAPI import commandRPlus
from modelPrepare.qwenAPI import QwenPlus
from llm_prepare import load_llm_model
from langchain_community.llms import OCIGenAI
import torch
from modelPrepare import sparkAPI
from modelPrepare import glm4API
from modelPrepare import chatgptAPI
from modelPrepare.aquaAPI import AIQuickActions
from modelPrepare.ociGenAIAPI import *
from langchain_community.embeddings import CohereEmbeddings
import llm_keys
from modelPrepare.remoteAPI import *
import modelPrepare.ociGenAIAPI
import oracledb
device = "cuda" if torch.cuda.is_available() else "cpu"

#######  DEV #################
#http_prefix = 'https://dev.oracle.k8scloud.site/'
http_prefix = 'http://150.230.37.250:8093/'
ORACLE_AI_VECTOR_CONNECTION_STRING="dd/fgt#_123@165.1.12.228:1521/kbpdb1.f.fff.oraclevcn.com"

OCI_OPEN_SEARCH_URL="https://amaaaaaaak7gbrialufa2y2ozyzfflp5ox2g5roy5aw5b6f7h3j2ee5z2zva.opensearch.ap-melbourne-1.oci.oraclecloud.com:9200"
OCI_OPEN_SEARCH_USER='opc'
OCI_OPEN_SEARCH_PASSWD='Qartrz!66'



#######  API auth method   #####################
auth_type= 'none'
# auth_type= 'API_KEY'
auth_type= 'INSTANCE_PRINCIPAL'


#######  the knowledge base root directory    #####################
KB_ROOT_PATH = '/home/ubuntu/kbroot'
#######  if use auto, the kbroot will be automatically set  in the same directory where kbot/ locates   ######################
#KB_ROOT_PATH = 'auto'

#######  sqlite parent directory    #######################################
sqlite_path = KB_ROOT_PATH

#######  OCI genAI Settings    #####################
compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"
GenAIEndpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

#######  Vector Store setting    #######################################
score_threshold =  0.6
vector_store_limit= 10

#######  Reranker model setting    #######################################
#rerankerModel = 'bgeReranker'
#BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-large"  #BAAI/bge-reranker-large
BGE_RERANK_PATH="BAAI/bge-reranker-large"
rerankerModel = 'cohereReranker'
reranker_topk= 2

#######  the memory window for chat history   #####################
history_k = 5

#######  Embedding model setting    #######################################
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
#e5_large_v2 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/e5-large-v2", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-large-zh-v1.5", model_kwargs={'device': device})
e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': device})

genaiEmbedding=KbotOCIGenAIEmbeddings(model_id="cohere.embed-multilingual-v3.0",
                                      service_endpoint=GenAIEndpoint,
                                      compartment_id=compartment_id,
                                      auth_type=auth_type)
cohereEmbedding = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=llm_keys.cohere_api_key)

## no need for a certain model, comment it
EMBEDDING_DICT = {
    # 'm3e-base': m3eEmbedding,
    'bge_m3': bge_m3,
    'bge_large_zh_v15': bge_large_zh_v15,
    'oci_genai_embed': genaiEmbedding,
    'e5_large_v2': e5_large_v2,
    'cohere_embed':cohereEmbedding
}

#######  llm model setting          #######################################
# use default authN method   INSTANCE_PRINCIPAL
ociGenAICohere = KbotOCIGenAI(
    model_id="cohere.command",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs={'max_tokens': 4096,
                  'temperature': 0,
                  }
)

ociGenAILlama2 =  KbotOCIGenAI(
    model_id="meta.llama-2-70b-chat",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs = {
      'max_tokens': 4096,
     'temperature'   : 0.10,
     }
)

MODEL_DICT = {
    'NoneLLM': 'NoneLLM',
    ######################      API models        #############################################
    'OCIGenAICohere': ociGenAICohere,
    'OCIGenAILlama2': ociGenAILlama2,
    'Llama3-8B':  remoteModel('/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct','http://146.235.214.184:8098/v1','123456'),
    'Llama-3-70B-Instruct':  remoteModel('meta-llama/Meta-Llama-3-70B-Instruct','http://141.147.8.181:8098/v1','123456',256,0),
    'ooba':  remoteModel('gpt-3.5-turbo','http://10.145.141.77:5000/v1','123456',256,0),
    'mistral-aqua': AIQuickActions(endpoint='https://modeldeployment.us-sanjose-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.us-sanjose-1.amaaaaaaak7gbriae3l7phztdfyhlwjqjgkfcqxwhu62gs5vur4sqryi5nvq'),
    'XingHuo': sparkAPI.SparkLLM(),
    'ChatGLM4':  glm4API.GLM4(),
    'ChatGPT' : chatgptAPI.gpt3,
    'Qwen-plus': QwenPlus(),
    'Cohere-CommandR+': commandRPlus(),
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
    'Qwen1.5-7B-Chat': load_llm_model('Qwen','Qwen/Qwen1.5-7B-Chat'),

    ###   format 3) : llm_prepare.py special configured model, sometimes we need special model params
    #'ChatGLM3-6B': load_llm_model(model_alias= 'chatglm3')
    # 'Chatglm3Remote': load_llm_model(model_alias = 'chatglm3Remote'),

}

########  VECTOR_STORE Types  #######################################
VECTOR_STORE_DICT = [
    'faiss',
    'oracle',
    'opensearch'
]


######## Select AI ########
selectai_pool = None
#selectai_pool = oracledb.create_pool(
#    user="WKSP_XH",
#    password="Cntech!123456#",
#    dsn="cntech_medium",
#    wallet_location="/home/ubuntu/qq/Keys",
#    config_dir="/home/ubuntu/qq/Keys",
#    wallet_password="admin1234",
#)