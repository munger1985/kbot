from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

def remoteEmbedding(model_name,openai_api_base,openai_api_key, dimensions=1024,tiktoken_enabled=False):
    '''
    openai_api_base : "http://138.2.237.212:8093/v1"
    '''
    embeddings = OpenAIEmbeddings(
                model=model_name,
                openai_api_key=openai_api_key,
                openai_api_base=openai_api_base
    )
    return embeddings
def remoteModel(model_name,openai_api_base,openai_api_key, max_tokens=256,temperature=0.1,default_headers={}):
    '''
    openai_api_base : "http://138.2.237.212:8000/v1"
    '''
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        max_tokens= max_tokens,
        temperature=temperature,
        default_headers=default_headers,
        stop=["<|eot_id|>"]
    )
    return llm