from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from modelPrepare import chatgptAPI
from llm_prepare import load_llm_model
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms import OCIGenAI
import torch
from modelPrepare import sparkAPI
from modelPrepare import glm4API

device = "cuda" if torch.cuda.is_available() else "cpu"

#######  download fqdn    #####################
http_prefix = 'https://dev.oracle.k8scloud.site.key/'

#######  the knowledge base root directory    #####################

KB_ROOT_PATH = '/home/ubuntu/kbroot'

#######  OCI genAI Settings    #####################
compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"
GenAIEndpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

#######  Embedding model setting    #######################################
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
#e5_large_v2 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/e5-large-v2", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-large-zh-v1.5", model_kwargs={'device': device})
# e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
bge_m3 = HuggingFaceEmbeddings(model_name="/u20/kbot/bge", model_kwargs={'device': device})
# bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': device})

genaiEmbedding = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL'
)

#######  reranker model setting    #######################################
#rerankerModel = 'bgeReranker'
#BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-large"  #BAAI/bge-reranker-large
BGE_RERANK_PATH="/u20/kbot/bgereranker"
# cohere_api_key='f2tdOlbKMadK2UwfcTlAI8BjTBqQSRwvwLcoSsYG'
rerankerModel = 'bgeReranker'
reranker_topk= 2

#######  Vector Store setting    #######################################
score_threshold =  0.6
vector_store_limit= 10

#######  chat with history    #######################################
history_k =  3

## no need for a certain model, comment it
EMBEDDING_DICT = {
    # 'm3e-base': m3eEmbedding,
    'bge_m3': bge_m3,
    # 'bge_large_zh_v15': bge_large_zh_v15,
    # 'oci_genai_embed': genaiEmbedding,
    # 'e5_large_v2': e5_large_v2
}

#######  llm model setting          #######################################

# use default authN method   INSTANCE_PRINCIPAL
ociGenAICohere = OCIGenAI(
    model_id="cohere.command",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL',
    model_kwargs={'max_tokens': 4096,
                  'temperature': 0,
                  }
)

ociGenAILlama2 =  OCIGenAI(
    model_id="meta.llama-2-70b-chat",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL',
    model_kwargs = {  
      'max_tokens': 4096,
     'temperature'   : 0.10,
     }
)

MODEL_DICT = {
    ######################      API models        #############################################
    # 'OCIGenAICohere': ociGenAICohere,
    # 'OCIGenAILlama2': ociGenAILlama2,
    # #'Chatglm3Remote': load_llm_model('chatglm3Remote'),
    # 'XingHuo': sparkAPI.SparkLLM(),
    # 'ChatGLM4':  glm4API.GLM4(),
    # 'ChatGPT' : chatgptAPI.gpt3,
    # 'Qwen-plus': load_llm_model('qwen-plus'),
    # ######################      local models      ###########################################
    # ###   format 1) : local path
    # # e.g.  'llama-2-7b-chat':   load_llm_model("/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf"),
    # #'Llama-2-7B-Chat': load_llm_model('/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf'),
    # #'Gemma-7B-IT': load_llm_model('/home/ubuntu/ChatGPT/Models/google/gemma-7b-it'),
    # #'Mistral-7B-Instruct-v0.2': load_llm_model('/home/ubuntu/ChatGPT/Models/mistral/Mistral-7B-Instruct-v0.2'),
    # #'Qwen1.5-7B-Chat': load_llm_model('/home/ubuntu/ChatGPT/Models/Qwen/Qwen1.5-7B-Chat'),
    #
    # ###   format 2) : huggingface id
    # 'Llama-2-7B-Chat': load_llm_model('meta-llama/Llama-2-7b-chat-hf'),
    # 'Gemma-7B-IT': load_llm_model('google/gemma-7b-it'),
    # 'Mistral-7B-Instruct-v0.2': load_llm_model('mistralai/Mistral-7B-Instruct-v0.2'),
    'Qwen1.5-7B-Chat': load_llm_model('/u20/kbot/qwen'),

    ###   format 3) : llm_prepare.py special configured model, sometimes we need special model params
    #'ChatGLM3-6B': load_llm_model('chatglm3')
}


########  VECTOR_STORE Types  #######################################
VECTOR_STORE_DICT = [
    'faiss',
    'oracle'
]
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector_js/welcome1@146.235.233.91:1521/pdb1.sub08030309530.justinvnc1.oraclevcn.com"
ORACLE_AI_VECTOR_CONNECTION_STRING="vector/vector@129.159.40.144:1521/orclpdb1"

#######  sqlite parent directory    #######################################
sqlite_path = KB_ROOT_PATH
