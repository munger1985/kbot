"""KBot 4.0 自动测试、静态验收和实库 Smoke 工具。"""

from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOTS = (
    _ROOT / "packages" / "platform_core" / "src",
    _ROOT / "packages" / "platform_clients" / "src",
    _ROOT / "services" / "main_api" / "src",
    _ROOT / "services" / "agent_runtime" / "src",
    _ROOT / "services" / "knowledge_core" / "src",
    _ROOT / "services" / "aiops_agent" / "src",
    _ROOT / "services" / "model_serving" / "src",
)
for _source_root in reversed(_SOURCE_ROOTS):
    _source = str(_source_root)
    if _source not in sys.path:
        sys.path.insert(0, _source)
