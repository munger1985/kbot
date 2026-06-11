from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator
from dataclasses import dataclass
from agent.common import ContextMemory


class SkillDomain(str, Enum):
    """Skill domain enumeration for sandbox-level isolation."""
    BUSINESS = "business"   # 通用业务域（普通文档、业务报表 Text-SQL 等）
    OPS = "ops"             # 自动化运维域（DBMetric, DBAction 等专有能力）
    SECURITY = "security"   # 安全与审计域


class SkillRunMode(str, Enum):
    """Skill execution mode — controls safety gate behavior."""
    READ_ONLY = "read_only" # 只读探测（无副作用，可交给 Planner 自由编排）
    MUTATION = "mutation"   # 变更/高危操作（需要挂载人工审批组件或二次确认）


@dataclass
class SkillMeta:
    """Skill metadata declared as a class attribute on each Skill subclass."""
    name: str
    description: str
    domain: SkillDomain = SkillDomain.BUSINESS
    run_mode: SkillRunMode = SkillRunMode.READ_ONLY


class BaseSkill(ABC):
    # 类属性元数据定义：每个子类通过复写此属性来进行自我声明
    meta: SkillMeta = SkillMeta(
        name="base_skill",
        description="系统基础技能基类"
    )

    @abstractmethod
    async def run_stream(self, context: ContextMemory, **kwargs) -> AsyncGenerator[Any, None]:
        """
        所有 Skill 必须实现的流式入口。
        kwargs 将包含：question, task_input, model_name, context (dict)
        """
        pass

    async def run(self, context: ContextMemory, **kwargs) -> Any:
        """非流式兼容层"""
        results = []
        async for packet in self.run_stream(context, **kwargs): # type: ignore
            results.append(packet)
        return results

    def _get_final_answer_from_stream(self, packets: list[dict[str, Any]]) -> Any:
        """通用工具：从流式包列表中提取最终答案或数据"""
        final_ans = ""
        last_data = None
        for p in packets:
            if p.get("type") in ["doc_results", "sql_results", "data"]:
                last_data = p.get("content")
            elif p.get("type") == "answer":
                final_ans += p.get("content", "")
        return last_data if last_data is not None else final_ans
