"""MCP 问数与 ECharts Skill。"""

from .contracts import EChartsResult, QueryResult
from .skill import EChartsSkill, MCPDataQuerySkill

__all__ = [
    "EChartsResult",
    "EChartsSkill",
    "MCPDataQuerySkill",
    "QueryResult",
]
