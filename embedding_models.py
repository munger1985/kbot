import torch
from modelPrepare import cohereAPI
from lazy_load import lazy_func

from modelPrepare import chatgptAPI
from modelPrepare  import  ociGenAIAPI     
from langchain_community.embeddings import HuggingFaceEmbeddings

from modelPrepare.remoteAPI import *
device = "cuda" if torch.cuda.is_available() else "cpu"
@lazy_func
def load_embedding_model( model_name_or_path,device)  :

    # e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
    # bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
    # bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh-v1.5", model_kwargs={'device': device})

    return HuggingFaceEmbeddings(model_name=model_name_or_path, model_kwargs={'device': device})


EMBEDDING_DICT = {
    'bge_m3': load_embedding_model("BAAI/bge-m3",device),
    'bge_large_zh_v15': load_embedding_model("BAAI/bge-large-zh-v1.5",device),
    'e5-multi': load_embedding_model("intfloat/multilingual-e5-large-instruct",device),
    'e5_large_v2': load_embedding_model("intfloat/e5-large-v2",device),
    'OCI-cohere.embed-multilingual-v3.0': ociGenAIAPI.genaiEmbedding,
    'cohere_embed':cohereAPI.cohereEmbedding,
    'text-embedding-3-large':chatgptAPI.openaiEmbeddings,
    'cohereRemoteVM': remoteEmbedding(model_name='OCI-cohere.embed-multilingual-v3.0',
                                      openai_api_key='xxxx',
                                      openai_api_base='http://localhost:8093/v1')
}


