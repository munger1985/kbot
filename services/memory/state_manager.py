from typing import Any
from loguru import logger

class SessionStateManager:
    """
    负责维护和持久化会话状态 (Session State)
    """

    @staticmethod
    def merge_state(old_state: dict[str, Any] | None, turn_entities: dict[str, Any] | None) -> dict[str, Any]:
        """
        逻辑：将本轮提取的实体合并到全局状态中。
        规则：新实体覆盖旧实体；如果 turn_entities 为空，保持原样。
        """
        base_state = (old_state or {}).copy()
        updates = turn_entities or {}
        
        for key, value in updates.items():
            if value is not None and value != "":
                # 记录状态变更，便于调试
                if key in base_state and base_state[key] != value:
                    logger.info(f"State Update [{key}]: {base_state[key]} -> {value}")
                base_state[key] = value
                
        return base_state