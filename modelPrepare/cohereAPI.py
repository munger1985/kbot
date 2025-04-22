from config import config
from langchain_community.llms import Cohere


# from langchain_community.embeddings import CohereEmbeddings

from langchain_cohere import CohereEmbeddings


def commandRPlus(model="command-r-plus",temperature=0, max_tokens=102400):
    llm = Cohere(model=model,
                 cohere_api_key=config.cohere_api_key,
                 temperature=temperature,
                 max_tokens=max_tokens)
    return llm

cohereEmbedding = CohereEmbeddings(model="embed-multilingual-v3.0", cohere_api_key=config.cohere_api_key)
