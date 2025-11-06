#!/bin/bash

# 刷新微服务核心包
cp core/dictionary.py microservices/embedding/ms_core/dictionary.py 
cp core/dictionary.py microservices/llm/ms_core/dictionary.py 
cp core/dictionary.py microservices/vlm/ms_core/dictionary.py 
cp core/dictionary.py microservices/reranker/ms_core/dictionary.py

cp configuration/config_type.py microservices/embedding/ms_core/config_type.py 
cp configuration/config_type.py microservices/llm/ms_core/config_type.py 
cp configuration/config_type.py microservices/vlm/ms_core/config_type.py 
cp configuration/config_type.py microservices/reranker/ms_core/config_type.py 

cp configuration/config_manager.py microservices/embedding/ms_core/config_manager.py 
cp configuration/config_manager.py microservices/llm/ms_core/config_manager.py 
cp configuration/config_manager.py microservices/vlm/ms_core/config_manager.py 
cp configuration/config_manager.py microservices/reranker/ms_core/config_manager.py 

sed -i 's/from core.nacos_manager import nacos_manager/from .nacos_manager import nacos_manager/g' microservices/embedding/ms_core/config_manager.py
sed -i 's/from core.nacos_manager import nacos_manager/from .nacos_manager import nacos_manager/g' microservices/llm/ms_core/config_manager.py
sed -i 's/from core.nacos_manager import nacos_manager/from .nacos_manager import nacos_manager/g' microservices/vlm/ms_core/config_manager.py
sed -i 's/from core.nacos_manager import nacos_manager/from .nacos_manager import nacos_manager/g' microservices/reranker/ms_core/config_manager.py

cp core/logger_manager/logger_manager.py microservices/embedding/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/llm/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/vlm/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/reranker/ms_core/logger_manager.py 


cp core/nacos_manager/nacos_encryptor.py microservices/embedding/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/llm/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/vlm/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/reranker/ms_core/nacos_encryptor.py 


cp core/nacos_manager/nacos_manager.py microservices/embedding/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/llm/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/vlm/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/reranker/ms_core/nacos_manager.py 


cp core/database/meta_redis.py microservices/embedding/ms_core/meta_redis.py 
cp core/database/meta_redis.py microservices/llm/ms_core/meta_redis.py 
cp core/database/meta_redis.py microservices/vlm/ms_core/meta_redis.py 
cp core/database/meta_redis.py microservices/reranker/ms_core/meta_redis.py 

sed -i 's/from configuration import ConfigManager/from .config_manager import ConfigManager/g' microservices/embedding/ms_core/meta_redis.py
sed -i 's/from configuration import ConfigManager/from .config_manager import ConfigManager/g' microservices/llm/ms_core/meta_redis.py
sed -i 's/from configuration import ConfigManager/from .config_manager import ConfigManager/g' microservices/vlm/ms_core/meta_redis.py
sed -i 's/from configuration import ConfigManager/from .config_manager import ConfigManager/g' microservices/reranker/ms_core/meta_redis.py

# cp core/security/auth.py microservices/embedding/ms_core/auth.py 
# cp core/security/auth.py microservices/vlm/ms_core/auth.py 
# cp core/security/auth.py microservices/llm/ms_core/auth.py 
# cp core/security/auth.py microservices/reranker/ms_core/auth.py 

# sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' microservices/embedding/ms_core/auth.py
# sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' microservices/llm/ms_core/auth.py
# sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' microservices/vlm/ms_core/auth.py
# sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' microservices/reranker/ms_core/auth.py



# 初始化微服务.env
cp .env.example microservices/embedding/.env
cp .env.example microservices/llm/.env
cp .env.example microservices/vlm/.env
cp .env.example microservices/reranker/.env

echo "初始化微服务完成"