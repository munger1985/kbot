# core/config/settings.py
from functools import lru_cache
from pathlib import Path
from typing import Any
import os
import tomli
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

class LogConfig(BaseModel):
    """Log configuration settings.
    
    Configuration parameters for logging system, including log level, storage path,
    rotation policy, and retention rules.
    """
    level: str = Field(default="INFO", description="Log severity level (DEBUG/INFO/WARN/ERROR/FATAL)")
    dir: str = Field(default="./logs", description="Directory to store log files")
    rotation: str = Field(default="100 MB", description="Log file rotation threshold (size/time)")
    retention: str = Field(default="10 days", description="Log file retention period")
    api_log_enabled: bool = Field(default=True, description="Whether to enable API request logging")

class AppConfig(BaseModel):
    """Main application configuration.
    
    Core configuration parameters for the KBOT application, including service metadata,
    network settings, and file storage configuration.
    """
    app_id: int = Field(default=1, description="Unique application identifier")
    service_name: str = Field(default="main_service", description="Name of the service instance")
    service_version: str = Field(default="3.0.0", description="Service version number")
    service_host: str = Field(default="0.0.0.0", description="Service binding host address")
    service_port: int = Field(default=18099, ge=1, le=65535, description="Service listening port (1-65535)")
    mcp_port: int = Field(default=18098, ge=1, le=65535, description="MCP Service listening port (1-65535)")
    title: str = Field(default="KBOT", description="Application display title")
    description: str = Field(default="KBot API Service", description="Application description")
    debug: bool = Field(default=False, description="Debug mode flag (True/False)")
    file_storage: str = Field(default="./knowledge_base", description="Root directory for knowledge base file storage")
    upload_workers: int = Field(default=5, ge=1, le=50, description="Number of worker threads for file uploads (1-50)")
    log: LogConfig = LogConfig()

class OracleConfig(BaseModel):
    """Oracle database configuration.
    
    Connection parameters for Oracle database, including authentication and network settings.
    """
    username: str = Field(default="kbot", description="Oracle database username")
    password: str = Field(default="", description="Oracle database password")
    host: str = Field(default="localhost", description="Oracle database host address")
    port: int = Field(default=1521, ge=1, le=65535, description="Oracle database port (1-65535)")
    service_name: str = Field(default="kbotdev", description="Oracle service name/SID")
    
    @property
    def dsn(self) -> str:
        """Generate Oracle DSN (Data Source Name) string.
        
        Returns:
            str: Formatted Oracle DSN string for database connection.
        """
        return f"{self.username}/{self.password}@{self.host}:{self.port}/{self.service_name}"

class SQLAlchemyConfig(BaseModel):
    """SQLAlchemy ORM configuration.
    
    Connection pool and runtime settings for SQLAlchemy database ORM.
    """
    echo: bool = Field(default=False, description="Enable SQL statement logging (echo mode)")
    pool_size: int = Field(default=10, ge=1, le=50, description="Database connection pool size (1-50)")
    pool_timeout: int = Field(default=60, ge=10, le=300, description="Connection pool timeout in seconds (10-300)")
    max_overflow: int = Field(default=20, ge=0, le=50, description="Maximum overflow connections (0-50)")
    pool_pre_ping: bool = Field(default=True, description="Enable connection health check before use")
    pool_use_lifo: bool = Field(default=True, description="Use LIFO strategy for connection pool")
    pool_recycle: int = Field(default=1800, ge=60, le=3600, description="Connection recycle time in seconds (60-3600)")

class EmbedConfig(BaseModel):
    """Embedding service configuration.
    
    Configuration parameters for the text embedding service, including network settings,
    token limits, and retry policies.
    """
    service_name: str = Field(default="embedding-service", description="Name of the embedding service")
    service_version: str = Field(default="1.0.0", description="Embedding service version")
    service_host: str = Field(default="0.0.0.0", description="Embedding service host address")
    service_port: int = Field(default=18091, ge=1, le=65535, description="Embedding service port (1-65535)")
    dimensions: int | None = Field(default=None, ge=512, le=32768, description="Embedding dimensions (512-32768)")
    max_tokens: int = Field(default=1024, ge=512, le=32768, description="Maximum tokens per embedding request (512-32768)")
    timeout: int = Field(default=300, ge=10, le=65535, description="Request timeout in seconds (10-65535)")
    health_check_timeout: int = Field(default=10, ge=5, le=60, description="Health check timeout in seconds (5-60)")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts for failed requests (0-10)")
    cache_dir: str = Field(default="./cached_models", description="Directory for cached embedding models")
    
    @property
    def service_url(self) -> str:
        """Generate full service URL for embedding service.
        
        Returns:
            str: Complete URL for accessing the embedding service.
        """
        return f"http://{self.service_host}:{self.service_port}"

class LLMConfig(BaseModel):
    """LLM (Large Language Model) service configuration.
    
    Configuration parameters for the LLM service, including network settings,
    generation parameters, and timeout settings.
    """
    service_name: str = Field(default="llm-service", description="Name of the LLM service")
    service_version: str = Field(default="1.0.0", description="LLM service version")
    service_host: str = Field(default="0.0.0.0", description="LLM service host address")
    service_port: int = Field(default=18092, ge=1, le=65535, description="LLM service port (1-65535)")
    max_tokens: int = Field(default=8192, ge=512, le=32768, description="Maximum output tokens (512-32768)")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature (0.0-2.0)")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter (0.0-1.0)")
    top_k: int = Field(default=0, ge=0, le=100, description="Top-k sampling parameter (0-100)")
    timeout: int = Field(default=300, ge=10, le=65535, description="Request timeout in seconds (10-65535)")
    health_check_timeout: int = Field(default=10, ge=5, le=60, description="Health check timeout in seconds (5-60)")
    frequency_penalty: float = Field(default=0.0, ge=0.0, le=2.0, description="Frequency penalty (0.0-2.0)")
    presence_penalty: float = Field(default=0.0, ge=0.0, le=2.0, description="Presence penalty (0.0-2.0)")
    
    @property
    def service_url(self) -> str:
        """Generate full service URL for LLM service.
        
        Returns:
            str: Complete URL for accessing the LLM service.
        """
        return f"http://{self.service_host}:{self.service_port}"

class RerankerConfig(BaseModel):
    """Reranker service configuration.
    
    Configuration parameters for the reranking service (reordering search results),
    including network settings and model caching.
    """
    service_name: str = Field(default="reranker-service", description="Name of the reranker service")
    service_version: str = Field(default="1.0.0", description="Reranker service version")
    service_host: str = Field(default="0.0.0.0", description="Reranker service host address")
    service_port: int = Field(default=18093, ge=1, le=65535, description="Reranker service port (1-65535)")
    cache_dir: str = Field(default="./cached_models", description="Directory for cached reranker models")
    timeout: int = Field(default=300, ge=10, le=65535, description="Request timeout in seconds (10-65535)")
    health_check_timeout: int = Field(default=10, ge=5, le=60, description="Health check timeout in seconds (5-60)")
    
    @property
    def service_url(self) -> str:
        """Generate full service URL for reranker service.
        
        Returns:
            str: Complete URL for accessing the reranker service.
        """
        return f"http://{self.service_host}:{self.service_port}"

class VLMConfig(BaseModel):
    """VLM (Vision-Language Model) service configuration.
    
    Configuration parameters for the vision-language model service, including
    network settings and timeout configurations.
    """
    service_name: str = Field(default="vlm-service", description="Name of the VLM service")
    service_version: str = Field(default="1.0.0", description="VLM service version")
    service_host: str = Field(default="0.0.0.0", description="VLM service host address")
    service_port: int = Field(default=18094, ge=1, le=65535, description="VLM service port (1-65535)")
    timeout: int = Field(default=300, ge=10, le=65535, description="Request timeout in seconds (10-65535)")
    health_check_timeout: int = Field(default=10, ge=5, le=60, description="Health check timeout in seconds (5-60)")
    
    @property
    def service_url(self) -> str:
        """Generate full service URL for VLM service.
        
        Returns:
            str: Complete URL for accessing the VLM service.
        """
        return f"http://{self.service_host}:{self.service_port}"

class ParserConfig(BaseModel):
    """Document parser service configuration.
    
    Configuration parameters for the document parsing service, including network settings,
    parallel processing limits, and artifact storage.
    """
    service_name: str = Field(default="parser-service", description="Name of the parser service")
    service_version: str = Field(default="1.0.0", description="Parser service version")
    service_host: str = Field(default="0.0.0.0", description="Parser service host address")
    service_port: int = Field(default=18095, ge=1, le=65535, description="Parser service port (1-65535)")
    timeout: int = Field(default=300, ge=10, le=65535, description="Request timeout in seconds (10-65535)")
    local_artifacts_path: str = Field(default="./cached_models", description="Path for local parser artifacts/models")
    queue_workers: int = Field(default=2, ge=1, le=20, description="Number of queue worker threads (1-20)")
    parser_parallel: int = Field(default=4, ge=1, le=100, description="Number of parallel parser processes (1-100)")
    db_check_interval: int = Field(default=60, ge=10, le=3600, description="Database check interval in seconds (10-3600)")

class ExecutorConfig(BaseModel):
    """SQL 执行器配置"""
    service_name: str = Field(default="sql-executor-service")
    service_version: str = Field(default="1.0.0")
    service_host: str = Field(default="0.0.0.0")
    service_port: int = Field(default=18096, ge=1, le=65535)
    timeout: int = Field(default=300, ge=10, le=3600)

class PromptConfig(BaseModel):
    """Prompt template configuration.
    
    Configuration for system prompt templates used in various NLP tasks.
    """
    image2text: str = Field(default="SYSTEM/image2text", description="Prompt template for image-to-text conversion")
    rewrite_question: str = Field(default="SYSTEM/rewrite_question", description="Prompt template for rewriting questions")
    refresh_summary: str = Field(default="SYSTEM/refresh_summary", description="Prompt template for refreshing summaries")
    rag_final_render: str = Field(default="SYSTEM/rag_final_render", description="Prompt template for RAG final render")
    user_profile: str = Field(default="SYSTEM/user_profile", description="Prompt template for user profile")
    sql_gen: str = Field(default="SYSTEM/sql_gen", description="SQL generation prompt template")
    sql_repair: str = Field(default="SYSTEM/sql_repair", description="SQL repair prompt template")
    task_planner: str = Field(default="SYSTEM/task_planner", description="Task planner prompt template")
    intent_router: str = Field(default="SYSTEM/intent_router", description="Intent router prompt template")
    reasoning: str = Field(default="SYSTEM/reasoning", description="Model reasoning prompt template")
    generate_chart: str = Field(default="SYSTEM/generate_chart", description="Generate chart prompt template")
    db_router: str = Field(default="SYSTEM/db_router", description="Database router prompt template")
    graph_vertex_fusion: str = Field(default="SYSTEM/graph_vertex_fusion", description="Graph vertex fusion prompt template")
    graph_extractor: str = Field(default="SYSTEM/graph_extractor", description="Graph extractor prompt template")


class Settings(BaseSettings):
    """Global application settings.
    
    Centralized configuration management for the entire KBOT application, supporting
    environment-specific configuration files (TOML) and environment variable overrides.
    """
    
    # Environment configuration - override via environment variables
    environment: str = "development"
    config_dir: str = "../configuration"
    
    # Module-specific configurations
    app: AppConfig = AppConfig()
    oracle: OracleConfig = OracleConfig()
    sqlalchemy: SQLAlchemyConfig = SQLAlchemyConfig()
    embed: EmbedConfig = EmbedConfig()
    llm: LLMConfig = LLMConfig()
    reranker: RerankerConfig = RerankerConfig()
    vlm: VLMConfig = VLMConfig()
    parser: ParserConfig = ParserConfig()
    executor: ExecutorConfig = ExecutorConfig()
    prompt: PromptConfig = PromptConfig()
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8", 
        "case_sensitive": False,
        "extra": "ignore",
        "env_prefix": "",  # No prefix for environment variables
    }
    
    @classmethod
    def create(cls, toml_path: Path | None = None) -> "Settings":
        """Create configuration instance with environment-specific overrides.
        
        Loads base configuration and merges with environment-specific configuration,
        with environment variables taking highest priority.
        
        Args:
            toml_path: Optional path to custom TOML config file
            
        Returns:
            Settings: Fully initialized configuration instance
        """
        # Check environment variables first
        env_from_env = os.getenv("ENVIRONMENT")
        config_dir_from_env = os.getenv("CONFIG_DIR")
        
        # Create temporary instance to get base config values
        temp_settings = cls()
        
        # Determine environment: env var first, then config file
        environment = env_from_env or temp_settings.environment
        config_dir = Path(config_dir_from_env or temp_settings.config_dir)
        
        print(f"Loading configuration for environment: {environment}")
        print(f"Config directory: {config_dir}")
        
        if toml_path is None:
            toml_path = config_dir / f"{environment}.toml"
            print(f"Loading TOML from: {toml_path}")
        
        # Ensure config directory exists
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load base configuration
        base_config_path = config_dir / "base.toml"
        base_config = cls._load_toml(base_config_path)
        
        # Load environment-specific configuration
        env_config = cls._load_toml(toml_path)
        
        # Merge configurations (env config overrides base config)
        merged_config = cls._deep_merge(base_config, env_config)
        
        # Create final configuration instance
        final_settings = cls(**merged_config)
        
        # Ensure environment settings are correct (env vars may override config files)
        if env_from_env:
            final_settings.environment = env_from_env
        if config_dir_from_env:
            final_settings.config_dir = config_dir_from_env
            
        return final_settings
    
    @staticmethod
    def _load_toml(file_path: Path) -> dict[str, Any]:
        """Load TOML configuration file, return empty dict if file not found.
        
        Args:
            file_path: Path to TOML configuration file
            
        Returns:
            dict[str, Any]: Parsed configuration dictionary (empty if file not found/error)
        """
        if not file_path.exists():
            print(f"Warning: Config file {file_path} not found, using defaults")
            return {}
        
        try:
            with open(file_path, "rb") as f:
                config = tomli.load(f)
                print(f"Loaded TOML config from: {file_path}")
                return config
        except Exception as e:
            print(f"Error loading TOML config {file_path}: {e}, using defaults")
            return {}
    
    @staticmethod
    def _deep_merge(base: dict, update: dict) -> dict:
        """Deep merge two dictionaries (update overrides base).
        
        Recursively merges nested dictionaries, with update dict values taking
        precedence over base dict values.
        
        Args:
            base: Base configuration dictionary
            update: Update configuration dictionary (higher priority)
            
        Returns:
            dict: Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = Settings._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result

    def is_development(self) -> bool:
        """Check if current environment is development.
        
        Returns:
            bool: True if development environment, False otherwise
        """
        return self.environment.lower() in ["dev", "development", "debug"]
    
    def is_production(self) -> bool:
        """Check if current environment is production.
        
        Returns:
            bool: True if production environment, False otherwise
        """
        return self.environment.lower() in ["prod", "production", "live"]
    
    def is_testing(self) -> bool:
        """Check if current environment is testing/staging.
        
        Returns:
            bool: True if testing/staging environment, False otherwise
        """
        return self.environment.lower() in ["test", "testing", "staging"]


# Global configuration instance
@lru_cache()
def get_settings() -> Settings:
    """Get cached global configuration instance.
    
    Uses LRU cache to ensure singleton configuration instance.
    
    Returns:
        Settings: Cached global configuration instance
    """
    return Settings.create()

# Convenience accessor functions
def get_app_config() -> AppConfig:
    """Get main application configuration.
    
    Returns:
        AppConfig: Application configuration object
    """
    return get_settings().app

def get_log_config() -> LogConfig:
    """Get logging configuration.
    
    Returns:
        LogConfig: Logging configuration object
    """
    return get_settings().app.log

def get_embed_config() -> EmbedConfig:
    """Get embedding service configuration.
    
    Returns:
        EmbedConfig: Embedding service configuration object
    """
    return get_settings().embed

def get_llm_config() -> LLMConfig:
    """Get LLM service configuration.
    
    Returns:
        LLMConfig: LLM service configuration object
    """
    return get_settings().llm

def get_oracle_config() -> OracleConfig:
    """Get Oracle database configuration.
    
    Returns:
        OracleConfig: Oracle database configuration object
    """
    return get_settings().oracle

def get_sqlalchemy_config() -> SQLAlchemyConfig:
    """Get SQLAlchemy configuration.
    
    Returns:
        SQLAlchemyConfig: SQLAlchemy ORM configuration object
    """
    return get_settings().sqlalchemy

def get_prompt_config() -> PromptConfig:
    """Get prompt template configuration.
    
    Returns:
        PromptConfig: Prompt template configuration object
    """
    return get_settings().prompt

def get_reranker_config() -> RerankerConfig:
    """Get reranker service configuration.
    
    Returns:
        RerankerConfig: Reranker service configuration object
    """
    return get_settings().reranker

def get_vlm_config() -> VLMConfig:
    """Get VLM service configuration.
    
    Returns:
        VLMConfig: VLM service configuration object
    """
    return get_settings().vlm

def get_parser_config() -> ParserConfig:
    """Get document parser configuration.
    
    Returns:
        ParserConfig: Document parser service configuration object
    """
    return get_settings().parser

def get_executor_config() -> ExecutorConfig:
    """Get executor configuration."""
    return get_settings().executor