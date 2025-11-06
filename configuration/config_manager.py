from .config_type import *
from core.nacos_manager import nacos_manager

    
class ConfigManager:
    """
    封装 Nacos 配置读取逻辑，避免代码中重复读取
    """

    @staticmethod
    def get_app_config() -> AppConfig:
        """
        获取应用配置

        :return: AppConfig 实例，如果失败则返回默认配置
        """
        try:
            config_str = nacos_manager.get_config("app_config")
            if not config_str:
                return AppConfig(
                    kbot=KBotConfig(
                        title="KBot 3.0.0",
                        description="KBot backend API services for KBot",
                        version="3.0.0",
                        debug=True,
                        file_storage="kbot_files/",
                        upload_workers=4,
                        parser=ParserConfig(max_workers=4, check_interval=10),
                        log=LogConfig(level="DEBUG", dir="logs/", rotation="100 MB", retention="10 days")
                    ),
                    libre=LibreConfig(host="localhost", port=9309)
                )

            return AppConfig.model_validate_json(config_str)
        except Exception as e:
            print(f"Failed to get app config from Nacos: {e}")
            return AppConfig(
                    kbot=KBotConfig(
                        title="KBot 3.0.0",
                        description="KBot backend API services for KBot",
                        version="3.0.0",
                        debug=True,
                        file_storage="kbot_files/",
                        upload_workers=4,
                        parser=ParserConfig(max_workers=4, check_interval=10),
                        log=LogConfig(level="DEBUG", dir="logs/", rotation="100 MB", retention="10 days")
                    ),
                    libre=LibreConfig(host="localhost", port=9309)
                )


    @staticmethod
    def get_db_config() -> DBConfig:
        """
        获取数据库配置

        :return: DBConfig 实例，如果失败则返回默认配置
        """
        try:
            config_str = nacos_manager.get_config("db_config")
            if not config_str:
                return DBConfig(
                    oracle=OracleConfig(
                        host="localhost",
                        port=1521,
                        username="kbot",
                        password="welcome1",
                        service_name="ORCL"
                    ),
                    redis=RedisConfig(
                        host="localhost",
                        port=6379,
                        password="welcome1",
                        max_connections=10,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                        retry_on_timeout=True,
                        health_check_interval=30
                    ),
                    sqlalchemy=SQLAlchemyConfig(
                        echo=False,
                        pool_size=5,
                        pool_timeout=30,
                        max_overflow=10,
                        pool_pre_ping=True,
                        pool_recycle=3600,
                        pool_use_lifo=False
                    ),
                    eslog=EslogConfig(
                        hosts=["https://localhost:9201"],
                        username="elastic",
                        password="<PASSWORD>",
                        index="kbot_logs"
                    )
                )

            return DBConfig.model_validate_json(config_str)
        except Exception as e:
            print(f"Failed to get db config from Nacos: {e}")
            return DBConfig(
                oracle=OracleConfig(
                    host="localhost",
                    port=1521,
                    username="default",
                    password="welcome1",
                    service_name="ORCL"
                ),
                redis=RedisConfig(
                    host="localhost",
                    port=6379,
                    password="welcome1",
                    max_connections=10,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30
                ),
                sqlalchemy=SQLAlchemyConfig(
                    echo=False,
                    pool_size=5,
                    pool_timeout=30,
                    max_overflow=10,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    pool_use_lifo=False
                ),
                eslog=EslogConfig(
                    hosts=["https://localhost:9201"],
                    username="elastic",
                    password="<PASSWORD>",
                    index="kbot_logs"
                )
            )

    @staticmethod
    def get_model_config() -> ModelConfig:
        """
        获取模型配置

        :return: ModelConfig 实例，如果失败则返回默认配置
        """
        try:
            config_str = nacos_manager.get_config("model_config")
            if not config_str:
                return ModelConfig(
                    embed=EmbedConfig(
                        service_name="embedding",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9301,
                        max_tokens=8192,
                        timeout=30,
                        max_retries=3,
                        cache_dir="/tmp"
                    ),
                    llm=LLMConfig(
                        service_name="llm",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9302,
                        max_tokens=8192,
                        temperature=0.7,
                        top_p=1.0,
                        top_k=5,
                        timeout=30,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    ),
                    reranker=RerankerConfig(
                        service_name="reranker",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9303,
                        cache_dir="/tmp",
                        timeout=30
                    ),
                    vlm=VLMConfig(
                        service_name="vlm",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9304,
                        timeout=30
                    ),
                    tokenizer=TokenizerConfig(
                        custom_dict_path="configuration/custom_dict.txt",
                        stop_words_path="configuration/stop_words.txt"
                    ),
                    prompt=PromptConfig(
                        image2text="SYSTEM/image2text",
                        summary="SYSTEM/summary"
                    )
                )

            return ModelConfig.model_validate_json(config_str)
        except Exception as e:
            print(f"Failed to get model config from Nacos: {e}")
            return ModelConfig(
                    embed=EmbedConfig(
                        service_name="embedding",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9301,
                        max_tokens=8192,
                        timeout=30,
                        max_retries=3,
                        cache_dir="/tmp"
                    ),
                    llm=LLMConfig(
                        service_name="llm",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9302,
                        max_tokens=8192,
                        temperature=0.7,
                        top_p=1.0,
                        top_k=5,
                        timeout=30,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    ),
                    reranker=RerankerConfig(
                        service_name="reranker",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9303,
                        cache_dir="/tmp",
                        timeout=30
                    ),
                    vlm=VLMConfig(
                        service_name="vlm",
                        service_version="1.0.0",
                        service_host="localhost",
                        service_port=9304,
                        timeout=30
                    ),
                    tokenizer=TokenizerConfig(
                        custom_dict_path="configuration/custom_dict.txt",
                        stop_words_path="configuration/stop_words.txt"
                    ),
                    prompt=PromptConfig(
                        image2text="SYSTEM/image2text",
                        summary="SYSTEM/summary"
                    )
                )
