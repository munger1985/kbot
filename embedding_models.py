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
device = "cuda" if torch.cuda.is_available() else "cpu"




EMBEDDING_DICT = {
    'bge_m3': load_embedding_model("BAAI/bge-m3",device),
    'bge_large_zh_v15': load_embedding_model("BAAI/bge-large-zh-v1.5",device),
    'e5_large_v2': load_embedding_model("intfloat/e5-large-v2",device),
    'OCI-cohere.embed-multilingual-v3.0': ociGenAIAPI.genaiEmbedding,
    'cohere_embed':cohereAPI.cohereEmbedding,
    'text-embedding-3-large':chatgptAPI.openaiEmbeddings,
    'cohereRemoteVM': remoteEmbedding(model_name='OCI-cohere.embed-multilingual-v3.0',
                                      openai_api_key='xxxx',
                                      openai_api_base='http://localhost:8093/v1')
}


