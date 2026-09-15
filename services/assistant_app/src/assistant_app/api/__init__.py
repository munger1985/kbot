"""智能工作台内部路由。"""

from .agents import router as agent_router
from .bindings import router as binding_router
from .image_generation import router as image_generation_router
from .media import router as media_router
from .research import router as research_router
from .runs import router as run_router

__all__ = [
    "agent_router",
    "binding_router",
    "image_generation_router",
    "media_router",
    "research_router",
    "run_router",
]
