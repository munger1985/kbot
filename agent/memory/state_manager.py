from typing import Any
from loguru import logger

class SessionStateManager:
    """
    负责维护和持久化会话状态 (Session State)
    """

    @staticmethod
    def merge_state(
        old_state: dict[str, Any] | None, 
        turn_entities: dict[str, Any] | None,
        relevance_score: float = 1.0  # 新增：从 ContextManager 获取的相关性得分
    ) -> dict[str, Any]:
        """
        逻辑：将本轮提取的实体合并到全局状态中。
        规则：
        1. 如果话题发生剧烈切换 (relevance_score 低)，考虑清理过期状态。
        2. 支持 "null" 语义：如果提取到某个 Key 为空/None，视为用户想重置该状态。
        """
        # 1. 深度话题切换处理
        # 如果相关性极低（例如 < 0.3），说明用户开启了全新的话题
        # 此时我们不应该继承旧的所有实体，否则会造成“实体污染”
        if relevance_score < 0.3:
            logger.info(f"Topic shift detected ({relevance_score}). Resetting transient states.")
            # 仅保留极其核心的全局状态（如 user_id），清空业务状态
            # 或者直接返回本轮新实体，不合并旧实体
            return turn_entities or {}

        base_state = (old_state or {}).copy()
        updates = turn_entities or {}
        
        for key, value in updates.items():
            # 2. 状态重置逻辑
            # 如果新提取的值是 None 或 明确的空字符串，且旧状态里有，则删除它
            # 场景：用户说“不用看 2025 年的数据了，看全部”，此时需要清空 year 标签
            if value is None or value == "":
                if key in base_state:
                    logger.info(f"State Reset [{key}]")
                    base_state.pop(key)
                continue

            # 3. 记录变更并更新
            if key in base_state and base_state[key] != value:
                logger.info(f"State Update [{key}]: {base_state[key]} -> {value}")
            
            base_state[key] = value
                
        return base_state