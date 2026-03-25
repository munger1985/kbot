from typing import Dict, Any, List
from loguru import logger

class IntentRouter:
    """
    意图路由器：根据 LLM 识别的意图分类，决定检索路径和召回权重。
    """
    
    # 定义不同意图的检索配置
    STRATEGIES = {
        "technical_inquiry": {
            "use_kb": True,         # 是否搜索技术文档库
            "use_memory": True,     # 是否搜索历史对话记忆
            "top_k": 5,             # 检索数量
            "rerank": True          # 是否执行重排
        },
        "troubleshooting": {
            "use_kb": True,
            "use_memory": True,
            "top_k": 8,             # 故障排查需要更多上下文
            "rerank": True
        },
        "general_chat": {
            "use_kb": False,
            "use_memory": True,     # 仅搜索记忆以维持对话连续性
            "top_k": 3,
            "rerank": False
        },
        "system_action": {
            "use_kb": False,
            "use_memory": False,    # 插件执行类通常不需要检索
            "top_k": 0,
            "rerank": False
        }
    }

    @classmethod
    async def get_search_strategy(cls, intent: str) -> Dict[str, Any]:
        """
        获取检索策略，如果意图未知，回退到默认的技术咨询模式
        """
        strategy = cls.STRATEGIES.get(intent, cls.STRATEGIES["technical_inquiry"])
        logger.info(f"Router selected strategy for intent '{intent}': {strategy}")
        return strategy

    @staticmethod
    def should_trigger_plugin(intent: str, entities: Dict[str, Any]) -> bool:
        """
        判断是否需要触发 OCI 插件（例如：意图是 system_action 且包含 instance_id）
        """
        if intent == "system_action" and ("instance_id" in entities or "vm_name" in entities):
            return True
        return False