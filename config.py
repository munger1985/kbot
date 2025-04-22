

from dynaconf import Dynaconf

# 初始化配置
config = Dynaconf(
    # 配置文件路径，按顺序加载（后面的会覆盖前面的）
    settings_files=['settings.toml', '.secrets.toml'],

    # 启用环境变量支持
    environments=True,
    # 环境变量前缀
    env_prefix="MYAPP_",
    # 加载 .env 文件
    load_dotenv=True,
    # 配置文件中的默认环境
    env="default",
    # 环境变量切换器
    env_switcher="MYAPP_ENV",
    # 配置验证（可选）
    # validators=[
    #     # 确保必要的配置项存在
    #     "verify_required_settings",
    # ],
    # 必要的配置项
    # required_settings=["database.host", "database.name"],
)




#######  DEV #################
#http_prefix  is the url prefix for downloading the docs, is kbot server itself
# http_prefix = 'http://150.230.37.250:8093/'
# DOC_VIEWER_FLAG='N' ##Y|N,If config `Y`, it will use the http_doc_viewer to view the document, otherwise it will http_prefix to download the file.
# http_doc_viewer = "http://150.230.37.250/HysunDocuViewer/?src=http://150.230.37.250:8080/"
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector/vector@129.159.40.144:1521/orclpdb1"
# ORACLE_AI_VECTOR_CONNECTION_STRING="vector_dev/VEctor#_123@165.1.65.228:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"

# # ADW AI Vector Search
# ADW_VECTOR_SEARCH_USER = "vector_dev"
# ADW_VECTOR_SEARCH_PASSWORD = "BotWelcome123##"
# ADW_VECTOR_SEARCH_DSN = "kbotadw23ai_medium"
# ADW_VECTOR_SEARCH_WALLET_LOCATION = "/home/ubuntu/kbot/keys/adwvectordb"  # Wallet zip文件解压缩后的目录
# ADW_VECTOR_SEARCH_WALLET_PASSWORD = "BotWelcome123##"

# OCI_OPEN_SEARCH_URL="https://amaaaaaaak7gbrialufa2y2ozyzfflp5ox2g5roy5aw5b6f7h3j2ee5z2zva.opensearch.ap-melbourne-1.oci.oraclecloud.com:9200"
# OCI_OPEN_SEARCH_USER='opc'
# OCI_OPEN_SEARCH_PASSWD='Qartrz!66'

####### opensearch docker in ai-dev
# OCI_OPEN_SEARCH_URL="https://localhost:9200"
# OCI_OPEN_SEARCH_USER='admin'
# OCI_OPEN_SEARCH_PASSWD='admin'


# HeatWave VectorStore
# HEATWAVE_CONNECTION_PARAMS = {
#     "user": "admin",
#     "password": "BotWelcome123##",
#     "host": "192.9.158.173",
#     "database": "kbot_dev",
# }
# HEATWAVE_VECTOR_STORE_POOL_NAME = "heatwave_vectorstore_pool"
# HEATWAVE_VECTOR_STORE_POOL_SIZE = 6

#######  PRD #################
#http_prefix = 'https://prd.oracle.k8scloud.site/'
#ORACLE_AI_VECTOR_CONNECTION_STRING="vector_prd/VEctor#_123@165.1.65.228:1521/kbpdb1.sub08030309530.justinvnc1.oraclevcn.com"


#######  API auth method   #####################
# auth_type= 'none'
# auth_type= 'API_KEY'
# auth_type= 'INSTANCE_PRINCIPAL'


#######  the knowledge base root directory    #####################
# KB_ROOT_PATH = '/home/ubuntu/kbroot'
#######  if use auto, the kbroot will be automatically set  in the same directory where kbot/ locates   ######################

#######  sqlite parent directory    #######################################
# config.sqlite_path = config.KB_ROOT_PATH



# #######  Vector Store setting    #######################################
# score_threshold =  0.6
# vector_store_limit= 10

#######  Reranker model setting    #######################################
#rerankerModel = 'bgeReranker'
#BGE_RERANK_PATH="/home/ubuntu/ChatGPT/Models/Embeddings/bge-reranker-large"  #BAAI/bge-reranker-large
# BGE_RERANKER="BAAI/bge-reranker-v2-m3"
# # disableReranker, ociCohereReranker,cohereReranker
# rerankerModel = 'ociCohereReranker'
# reranker_topk= 2

# #######  the memory window for chat history   #####################
# history_k = 5

# #######  Embedding model setting    #######################################
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 50
#e5_large_v2 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/e5-large-v2", model_kwargs={'device': device})
#bge_large_zh_v15 = HuggingFaceEmbeddings(model_name="/home/ubuntu/ChatGPT/Models/Embeddings/bge-large-zh-v1.5", model_kwargs={'device': device})
# e5_large_v2 = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={'device': device})
# bge_m3 = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={'device': device})
# bge_large_zh_v15 = HuggingFaceEmbeddings(model_name=, model_kwargs={'device': device})



#######  llm model setting          #######################################
# use default authN method   INSTANCE_PRINCIPAL

########  VECTOR_STORE Types  #######################################
# VECTOR_STORE_DICT = [
#     'faiss',
#     'oracle',
#     'adb',
#     'opensearch',
#     'heatwave'
# ]
########## summary model #################################
# SUMMARY_MODEL ="OCI-cohere.command-r-plus082024"

######## Select AI ########
# selectai_pool = None
#selectai_pool = oracledb.create_pool(
#    user="WKSP_XH",
#    password="Cntech!123456#",
#    dsn="cntech_medium",
#    wallet_location="/home/ubuntu/qq/Keys",
#    config_dir="/home/ubuntu/qq/Keys",
#    wallet_password="admin1234",
#)

