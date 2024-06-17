import llm_keys
from langchain_community.llms import Cohere






def commandRPlus(model="command-r-plus",temperature=0, max_tokens=102400):
    llm = Cohere(model=model,
                 cohere_api_key=llm_keys.cohere_api_key,
                 temperature=temperature,
                 max_tokens=max_tokens)
    return llm