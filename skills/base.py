from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator
from agent.common import ContextMemory


class BaseSkill(ABC):
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