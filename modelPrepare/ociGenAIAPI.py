
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from llm_keys import GenAIEndpoint, auth_type, compartment_id


class KbotOCIGenAIEmbeddings(OCIGenAIEmbeddings):
    def __init__(self, **kwargs):
        if kwargs['auth_type']=='none':
            return
        super().__init__(**kwargs)
class KbotOCIGenAI(OCIGenAI):
    def __init__(self, **kwargs):
        if kwargs['auth_type']=='none':
            return
        super().__init__(**kwargs)
class KbotChatOCIGenAI(ChatOCIGenAI):
    def __init__(self, **kwargs):
        if kwargs['auth_type']=='none':
            return
        super().__init__(**kwargs)


genaiEmbedding=KbotOCIGenAIEmbeddings(model_id="cohere.embed-multilingual-v3.0",
                                      service_endpoint=GenAIEndpoint,
                                      compartment_id=compartment_id,
                                      auth_type=auth_type)
ociGenAICohere = KbotOCIGenAI(
    model_id="cohere.command",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
                  }
)

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

ociGenAILlama2 =  KbotOCIGenAI(
    model_id="meta.llama-2-70b-chat",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama3_1_405B =  KbotChatOCIGenAI(
    model_id="meta.llama-3.1-405b-instruct",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)

ociGenAILlama3_1_70B =  KbotChatOCIGenAI(
    model_id="meta.llama-3.1-70b-instruct",
    service_endpoint=GenAIEndpoint,
    compartment_id=compartment_id,
    auth_type=auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
