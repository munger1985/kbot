#!/bin/bash

# 刷新微服务核心包
cp core/dictionary.py microservices/embedding/ms_core/dictionary.py 
cp core/dictionary.py microservices/llm/ms_core/dictionary.py 
cp core/dictionary.py microservices/vlm/ms_core/dictionary.py 
cp core/dictionary.py microservices/reranker/ms_core/dictionary.py
cp core/dictionary.py microservices/synonym/ms_core/dictionary.py

cp configuration/config_type.py microservices/embedding/ms_core/config_type.py 
cp configuration/config_type.py microservices/llm/ms_core/config_type.py 
cp configuration/config_type.py microservices/vlm/ms_core/config_type.py 
cp configuration/config_type.py microservices/reranker/ms_core/config_type.py 
cp configuration/config_type.py microservices/synonym/ms_core/config_type.py 

cp core/logger_manager/logger_manager.py microservices/embedding/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/llm/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/vlm/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/reranker/ms_core/logger_manager.py 
cp core/logger_manager/logger_manager.py microservices/synonym/ms_core/logger_manager.py 

cp core/nacos_manager/nacos_encryptor.py microservices/embedding/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/llm/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/vlm/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/reranker/ms_core/nacos_encryptor.py 
cp core/nacos_manager/nacos_encryptor.py microservices/synonym/ms_core/nacos_encryptor.py 

cp core/nacos_manager/nacos_manager.py microservices/embedding/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/llm/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/vlm/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/reranker/ms_core/nacos_manager.py 
cp core/nacos_manager/nacos_manager.py microservices/synonym/ms_core/nacos_manager.py 

cp core/security/auth.py microservices/embedding/ms_core/auth.py 
cp core/security/auth.py microservices/vlm/ms_core/auth.py 
cp core/security/auth.py microservices/llm/ms_core/auth.py 
cp core/security/auth.py microservices/reranker/ms_core/auth.py 
cp core/security/auth.py microservices/synonym/ms_core/auth.py 

sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' /home/chris/kbot3/microservices/embedding/ms_core/auth.py
sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' /home/chris/kbot3/microservices/llm/ms_core/auth.py
sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' /home/chris/kbot3/microservices/vlm/ms_core/auth.py
sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' /home/chris/kbot3/microservices/reranker/ms_core/auth.py
sed -i 's/from core.dictionary import AccessorType/from .dictionary import AccessorType/g' /home/chris/kbot3/microservices/synonym/ms_core/auth.py

sed -i 's/from configuration.config_type import AppConfig, DBConfig, ModelConfig/from .config_type import AppConfig, DBConfig, ModelConfig/g' /home/chris/kbot3/microservices/embedding/ms_core/nacos_manager.py
sed -i 's/from configuration.config_type import AppConfig, DBConfig, ModelConfig/from .config_type import AppConfig, DBConfig, ModelConfig/g' /home/chris/kbot3/microservices/llm/ms_core/nacos_manager.py
sed -i 's/from configuration.config_type import AppConfig, DBConfig, ModelConfig/from .config_type import AppConfig, DBConfig, ModelConfig/g' /home/chris/kbot3/microservices/vlm/ms_core/nacos_manager.py
sed -i 's/from configuration.config_type import AppConfig, DBConfig, ModelConfig/from .config_type import AppConfig, DBConfig, ModelConfig/g' /home/chris/kbot3/microservices/reranker/ms_core/nacos_manager.py
sed -i 's/from configuration.config_type import AppConfig, DBConfig, ModelConfig/from .config_type import AppConfig, DBConfig, ModelConfig/g' /home/chris/kbot3/microservices/synonym/ms_core/nacos_manager.py



