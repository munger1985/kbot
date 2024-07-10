
from langchain_community.embeddings import OCIGenAIEmbeddings
from langchain_community.llms.oci_generative_ai import OCIGenAI
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI


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
