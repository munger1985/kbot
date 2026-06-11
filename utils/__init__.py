# utils/__init__.py — 工具模块统一导出

# 客户端: 外部服务调用
from .clients import AIModelClient, OpsDBExecutor

# 监控: Prometheus / Zabbix + 指标注册中心
from .monitor import (
    BaseMonitorProvider,
    MetricResult,
    PrometheusClient,
    ZabbixProvider,
    UnifiedMetricRegistry,
)

# 编码与序列化
from .codec import (
    DecimalEncoder,
    ImageEncoder,
    SerializerUtils,
    serialize_value,
    safe_serialize,
    OracleVecHandler,
)

# 线程工具
from .thread import (
    run_in_thread_pool,
    safe_read_content,
    model_to_dict,
    detect_language,
    get_embedding_dimension,
    detect_file_encoding,
)

# 文本清理
from .sanitize import sanitize_text_for_oracle_json

__all__ = [
    # clients
    "AIModelClient",
    "OpsDBExecutor",
    # monitor
    "BaseMonitorProvider",
    "MetricResult",
    "PrometheusClient",
    "ZabbixProvider",
    "UnifiedMetricRegistry",
    # codec
    "DecimalEncoder",
    "ImageEncoder",
    "SerializerUtils",
    "serialize_value",
    "safe_serialize",
    "OracleVecHandler",
    # thread
    "run_in_thread_pool",
    "safe_read_content",
    "model_to_dict",
    "detect_language",
    "get_embedding_dimension",
    "detect_file_encoding",
    # sanitize
    "sanitize_text_for_oracle_json",
]
