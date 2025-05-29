
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from config import config

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
                                      service_endpoint=config.GenAIEndpoint,
                                      compartment_id=config.compartment_id,
                                      auth_type=config.auth_type)
# ociGenAICohere = KbotOCIGenAI(
#     model_id="cohere.command",
#     service_endpoint=config.GenAIEndpoint,
#     compartment_id=config.compartment_id,
#     auth_type=config.auth_type,
#     model_kwargs={'max_tokens': 4000,
#                   'temperature': 0,
#                   }
# )

# ociCMDR = KbotChatOCIGenAI(
#     model_id="cohere.command-r-16k",
#     service_endpoint=config.GenAIEndpoint,
#     compartment_id=config.compartment_id,
#     auth_type=config.auth_type,
#     model_kwargs={'max_tokens': 4000,
#                   'temperature': 0,
#     }
# )

# ociCMDRPlus = KbotChatOCIGenAI(
#     model_id="cohere.command-r-plus",
#     service_endpoint=config.GenAIEndpoint,
#     compartment_id=config.compartment_id,
#     auth_type=config.auth_type,
#     model_kwargs={'max_tokens': 4000,
#                   'temperature': 0,
#     }
# )
ociCMDRPlus082024 = KbotChatOCIGenAI(
    model_id="cohere.command-r-plus-08-2024",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
    }
)
ociCMDA032025 = KbotChatOCIGenAI(
    model_id="cohere.command-a-03-2025",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
    }
)
ociCMDR082024 = KbotChatOCIGenAI(
    model_id="cohere.command-r-08-2024",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs={'max_tokens': 4000,
                  'temperature': 0,
    }
)

ociGenAILlama2 =  KbotOCIGenAI(
    model_id="meta.llama-2-70b-chat",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama3_1_405B =  KbotChatOCIGenAI(
    model_id="meta.llama-3.1-405b-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama3_2_90b =  KbotChatOCIGenAI(
    model_id="meta.llama-3.2-90b-vision-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama4_maverick_17b =  KbotChatOCIGenAI(
    model_id="meta.llama-4-maverick-17b-128e-instruct-fp8",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama4_scout_17b =  KbotChatOCIGenAI(
    model_id="meta.llama-4-scout-17b-16e-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)
ociGenAILlama3_3_70b =  KbotChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)

ociGenAILlama3_1_70B =  KbotChatOCIGenAI(
    model_id="meta.llama-3.1-70b-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)

ociGenAILlama3_3_70B =  KbotChatOCIGenAI(
    model_id="meta.llama-3.3-70b-instruct",
    service_endpoint=config.GenAIEndpoint,
    compartment_id=config.compartment_id,
    auth_type=config.auth_type,
    model_kwargs = {
      'max_tokens': 1024,
     'temperature'   : 0.10,
     }
)



def init_oci_auth(auth_type):
    finalConfig = {}
    if auth_type == 'API_KEY':
        ociconfig = oci.config.from_file()
        finalConfig = {'config': ociconfig}

    if auth_type == 'INSTANCE_PRINCIPAL':
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        finalConfig = {'config': {}, 'signer': signer}
    # if region:
    #     finalConfig.update({"region":region})
    return finalConfig


from pathlib import Path
import sys
# 获取当前模块的父目录的父目录（即与util同级的根目录）
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

import oci
region = 'ap-osaka-1'
model_id = 'cohere.rerank-multilingual-v3.1'
if config.auth_type == 'none':
    generative_ai_inference_client = None

else:
    generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
        **init_oci_auth(config.auth_type),
        service_endpoint=f"https://inference.generativeai.{region}.oci.oraclecloud.com",
        retry_strategy=oci.retry.NoneRetryStrategy(),
        timeout=(10, 240))

