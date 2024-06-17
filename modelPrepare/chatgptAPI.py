from langchain_openai import OpenAI
from langchain_openai import ChatOpenAI
import llm_keys

#gpt3 = OpenAI(model_name="gpt-3.5-turbo-instruct",
#      openai_api_key="sess-xxx" ,
#      temperature   =0.10,
#      max_tokens = 100,
#      top_p=0.9
#)

#https://platform.openai.com/docs/models
#gpt-4-0125-preview     128,000 tokens  Up to Dec 2023
#gpt-4(gpt-4-0613)      8,192 tokens    Up to Sep 2021
#gpt-3.5-turbo          16,385 tokens   Up to Sep 2021
#gpt-3.5-turbo-instruct 4,096 tokens    Up to Sep 2021

gpt3 = ChatOpenAI(temperature=0, 
            openai_api_key=llm_keys.openai_key,
            model_name="gpt-3.5-turbo")
