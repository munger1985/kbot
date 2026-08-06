import json
import importlib.util
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch


def _load_context_manager_module() -> ModuleType:
    modules = {
        "services": ModuleType("services"),
        "services.search": ModuleType("services.search"),
        "services.search.result": ModuleType("services.search.result"),
        "agent": ModuleType("agent"),
        "agent.prompt": ModuleType("agent.prompt"),
        "utils": ModuleType("utils"),
        "utils.clients": ModuleType("utils.clients"),
        "core": ModuleType("core"),
        "core.config": ModuleType("core.config"),
    }
    modules["services.search.result"].TxtBaseSearchResult = object
    modules["agent.prompt"].default_prompt = SimpleNamespace()
    modules["utils.clients"].AIModelClient = object
    modules["core.config"].get_prompt_config = lambda: SimpleNamespace()

    with patch.dict(sys.modules, modules):
        source_path = Path(__file__).parents[2] / "agent/memory/context_manager.py"
        spec = importlib.util.spec_from_file_location("context_manager_under_test", source_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


context_manager_module = _load_context_manager_module()
ContextManager = context_manager_module.ContextManager
default_prompt = context_manager_module.default_prompt


class TestContextManager(IsolatedAsyncioTestCase):
    async def test_process_query_serializes_decimal_session_state_for_prompt(self):
        manager = ContextManager.__new__(ContextManager)
        manager.rewrite_prompt = "重写提示词"
        manager.llm_client = type(
            "FakeLlmClient",
            (),
            {
                "get_llm_json": AsyncMock(
                    return_value={
                        "standalone_query": "查询",
                        "turn_entities": {},
                    }
                )
            },
        )()

        generate_prompt = AsyncMock(return_value="提示词")
        with patch.object(default_prompt, "generate", generate_prompt, create=True):
            await manager.process_query_with_memory(
                query="查询",
                chat_history="",
                context_summary=None,
                session_state={"AIDP": Decimal("1")},
                model_name="test-model",
            )

        serialized_state = generate_prompt.await_args.kwargs["session_state"]
        self.assertEqual(json.loads(serialized_state), {"AIDP": "1"})
