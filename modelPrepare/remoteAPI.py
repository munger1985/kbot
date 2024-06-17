from langchain_openai import ChatOpenAI

def remoteModel(model_name,openai_api_base,openai_api_key, max_tokens=256,temperature=0.2):
    '''
    openai_api_base : "http://138.2.237.212:8000/v1"
    '''
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        max_tokens= max_tokens,
        temperature=temperature
    )
    return llm