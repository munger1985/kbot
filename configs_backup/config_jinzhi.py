from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llm_prepare import load_llm_model

from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms import OCIGenAI
import torch
import logging
from modelPrepare import glm4API

device = "cuda" if torch.cuda.is_available() else "cpu"

#######  download fqdn    #####################
http_prefix = 'https://poc1.hub.sehub.tech/'

#######  the knowledge base root directory    #####################

KB_ROOT_PATH = '/home/ubuntu/kbroot'

#######  OCI genAI Settings    #####################
compartment_id = "ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"
GenAIEndpoint = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"

#######  Embedding model setting    #######################################
CHUNK_SIZE = 555
CHUNK_OVERLAP = 55
m3eEmbedding = HuggingFaceEmbeddings(model_name='moka-ai/m3e-base', model_kwargs={'device': device})
e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': device})
genaiEmbedding = OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL'
)

#######  Vector Store setting    #######################################
score_threshold =  0.6
reranker_topk= 1
vector_store_limit= 10

## no need for a certain model, comment it
EMBEDDING_DICT = {
    # 'm3e-base': m3eEmbedding,
    'bge_m3': bge_m3,
    'bge_large_zh_v15': bge_large_zh_v15,
    # 'oci_genai_embed': genaiEmbedding,
    'e5_large_v2': e5_large_v2
}

#######  llm model setting          #######################################

# use default authN method   INSTANCE_PRINCIPAL
ociGenAILLM = OCIGenAI(
    model_id="cohere.command",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL',
    model_kwargs={'max_tokens': 4096,
                  'temperature': 0.8,
                  'top_p': 0.77,
                  'frequency_penalty': 0.4
                  }

)

genai_llama =  OCIGenAI(
    model_id="meta.llama-2-70b-chat",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type='INSTANCE_PRINCIPAL',
    model_kwargs = {  'max_tokens': 600,
     'temperature'   : 0.10,
     'top_p':0.9}
)
MODEL_DICT = {
    ######################      API models        #############################################

    'glm4': glm4API.GLM4(),

    ######################      local models      ###########################################

    ###   format 1) : local path
    # e.g.  'llama-2-7b-chat':   load_llm_model("/home/ubuntu/ChatGPT/Models/meta/llama2/Llama-2-7b-chat-hf"),

    ###   format 2) : huggingface id


    ###   format 3) : llm_prepare.py special configured model, sometimes we need special model params
    'chatglm3': load_llm_model('chatglm3'),
}


########  VECTOR_STORE Types  #######################################
VECTOR_STORE_DICT = [
     'oracle' , 'faiss',

]
ORACLE_AI_VECTOR_CONNECTION_STRING="vector/vector@10.0.0.192:1521/orclpdb1"


########  sqlite parent directory    #######################################
sqlite_path = KB_ROOT_PATH



########  init at boot up      #######################################
logger.info(MODEL_DICT.get('chatglm3'))
