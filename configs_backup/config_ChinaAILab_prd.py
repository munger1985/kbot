from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from modelPrepare.cohereAPI import commandRPlus
from modelPrepare.qwenAPI import QwenPlus
from llm_prepare import load_llm_model
from langchain_community.llms import OCIGenAI
import torch
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from modelPrepare import sparkAPI
from modelPrepare import glm4API
from modelPrepare import chatgptAPI
from modelPrepare.aquaAPI import AIQuickActions
from modelPrepare.ociGenAIAPI import *
from modelPrepare import qianfanAPI
from langchain_community.embeddings import CohereEmbeddings
import llm_keys
from modelPrepare.remoteAPI import *
device = "cuda" if torch.cuda.is_available() else "cpu"

#######  DEV #################
#http_prefix = 'https://dev.oracle.k8scloud.site/'
#http_prefix = 'http://150.230.37.250:8899/'
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector/vector@129.159.40.144:1521/orclpdb1"
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector_dev/VEctor#_123@165.1.65.228:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"

#OCI_OPEN_SEARCH_URL="https://amaaaaaaak7gbrialufa2y2ozyzfflp5ox2g5roy5aw5b6f7h3j2ee5z2zva.opensearch.ap-melbourne-1.oci.oraclecloud.com:9200"
#OCI_OPEN_SEARCH_USER='opc'
#OCI_OPEN_SEARCH_PASSWD='Qartrz!66'
# HeatWave VectorStore
HEATWAVE_CONNECTION_PARAMS = {
    "user": "admin",
    "password": "BotWelcome123##",
    "host": "192.9.158.173",
    "database": "kbot_prd",
}
HEATWAVE_VECTOR_STORE_POOL_NAME = "heatwave_vectorstore_pool"
HEATWAVE_VECTOR_STORE_POOL_SIZE = 6
#######  PRD #################
#http_prefix = 'https://prd.oracle.k8scloud.site/'
DOC_VIEWER_FLAG='Y' ##Y|N,If config `Y`, it will use the http_doc_viewer to view the document, otherwise it will http_prefix to download the file.
http_prefix = 'https://kbot.china.sehub.tech/'
http_doc_viewer = "https://kbot.china.sehub.tech/HysunDocuViewer/?src=https://kbot.china.sehub.tech:8080/"
ORACLE_AI_VECTOR_CONNECTION_STRING="vector_prd/AILab1_#123New@150.230.199.230:1521/kbpdb1.sub12210155490.ailabvcn.oraclevcn.com"

#######  API auth method   #####################
auth_type= 'none'
# auth_type= 'API_KEY'
auth_type= 'INSTANCE_PRINCIPAL'

#######  the knowledge base root directory    #####################
KB_ROOT_PATH = '/home/ubuntu/kbroot'

#######  sqlite parent directory    #######################################
sqlite_path = KB_ROOT_PATH

#######  OCI genAI Settings    #####################
compartment_id = "ocid1.compartment.oc1..aaaaaaaasxmj2xt6gy7t7pf26nks6ehn3adfdakpbdgowuiage73xin3td2a"
GenAIEndpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

#######  Vector Store setting    #######################################
score_threshold =  0.6
vector_store_limit= 10

#######  Reranker model setting    #######################################
#rerankerModel = 'bgeReranker'
BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-large"  #BAAI/bge-reranker-large
#BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-v2-m3"  #BAAI/bge-reranker-v2-m3
#BGE_RERANK_PATH="BAAI/bge-reranker-large"
rerankerModel = 'cohereReranker'
reranker_topk= 2

#######  the memory window for chat history   #####################
history_k = 5

#######  Embedding model setting    #######################################
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
#e5_large_v2 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/e5-large-v2", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-large-zh-v1.5", model_kwargs={'device': device})
#e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
bge_m3 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-m3", model_kwargs={'device': device})
#bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': device})
bge_base_en_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-base-en-v1.5", model_kwargs={'device': device})
text2vec_large_chinese = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/text2vec-large-chinese", model_kwargs={'device': device})

genaiEmbedding=KbotOCIGenAIEmbeddings(model_id="cohere.embed-multilingual-v3.0",
                                      service_endpoint=GenAIEndpoint,
                                      compartment_id=compartment_id,
                                      auth_type=auth_type)

genaiEmbedding_light=KbotOCIGenAIEmbeddings(model_id="cohere.embed-multilingual-light-v3.0",
                                      service_endpoint=GenAIEndpoint,
                                      compartment_id=compartment_id,
                                      auth_type=auth_type)

## no need for a certain model, comment it
## no need for a certain model, comment it
EMBEDDING_DICT = {
    # 'm3e-base': m3eEmbedding,
    'bge_m3': bge_m3,
    #'bge_large_zh_v15': bge_large_zh_v15,
    'oci_genai_embed': genaiEmbedding,
    'bge_base_en_v15': bge_base_en_v15,
    'text2vec_large_chinese': text2vec_large_chinese,
    #'e5_large_v2': e5_large_v2,
    #'cohere_embed':cohereEmbedding
}
#######  llm model setting          #######################################
# use default authN method   INSTANCE_PRINCIPAL
ociCMDR = KbotChatOCIGenAI(
    model_id="cohere.command-r-16k",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
    }
)

ociCMDRPlus = KbotChatOCIGenAI(
    model_id="cohere.command-r-plus",
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
    }
)

ociGenAILlama3 =  KbotChatOCIGenAI(
    model_id="meta.llama-3-70b-instruct",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)


MODEL_DICT = {
    'NoneLLM': 'NoneLLM',
    ######################      API models        #############################################
    'OCIGenAICohereCmdR':ociCMDR,
    'OCIGenAICohereCmdR+':ociCMDRPlus,
    'OCIGenAILlama3': ociGenAILlama3,
    #'Chatglm3Remote': load_llm_model('chatglm3Remote'),
    'XingHuo': sparkAPI.SparkLLM(),
    'ChatGLM4':  glm4API.GLM4(),
    'ChatGPT' : chatgptAPI.gpt3,
    'Qwen-plus': load_llm_model('Qwen-plus'),
    ######################      local models      ###########################################
    ###   format 1) : local path
    # e.g.  'llama-2-7b-chat':   load_llm_model("/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf"),
    #'Llama-2-7B-Chat': load_llm_model('Llama-2-7B-Chat','/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf'),
    #'Gemma-7B-IT': load_llm_model('Gemma-7B-IT','/home/ubuntu/ChatGPT/Models/google/gemma-7b-it'),
    #'Mistral-7B-Instruct-v0.2': load_llm_model('Mistral','/home/ubuntu/ChatGPT/Models/mistral/Mistral-7B-Instruct-v0.2'),
    #'Llama-3-8B-Instruct': load_llm_model('Llama3','/home/ubuntu/ChatGPT/Models/meta/Meta-Llama-3-8B-Instruct'),

    ###   format 2) : huggingface id
    #'Llama-2-7B-Chat': load_llm_model('Llama2','meta-llama/Llama-2-7b-chat-hf'),
    #'Gemma-7B-IT': load_llm_model('Gemma','google/gemma-7b-it'),
    #'Mistral-7B-Instruct-v0.2': load_llm_model('Mistral','mistralai/Mistral-7B-Instruct-v0.2'),
    'Qwen1.5-7B-Chat': load_llm_model('Qwen','/home/ubuntu/ChatGPT/Models/Qwen/Qwen1.5-7B-Chat'),

    ###   format 3) : llm_prepare.py special configured model, sometimes we need special model params
    #'ChatGLM3-6B': load_llm_model('chatglm3')
}

########  VECTOR_STORE Types  #######################################
VECTOR_STORE_DICT = [
    'faiss',
    'oracle',
    #'opensearch',
    'heatwave'
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
