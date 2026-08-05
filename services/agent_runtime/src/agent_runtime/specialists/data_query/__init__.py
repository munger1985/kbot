"""统一问数 Skill 与 Provider Executor。"""

from .contracts import QueryResult
from .skill import (
    DataQuerySkill,
    MCPDataQueryExecutor,
    SemanticDataQueryExecutor,
)

__all__ = [
    "DataQuerySkill",
    "MCPDataQueryExecutor",
    "QueryResult",
    "SemanticDataQueryExecutor",
]
