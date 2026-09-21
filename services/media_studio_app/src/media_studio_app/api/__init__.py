"""多媒体创作工作台内部路由。"""

from .bindings import router as binding_router
from .image_generation import router as image_generation_router
from .media import router as media_router
from .runs import router as run_router

__all__ = ["binding_router", "image_generation_router", "media_router", "run_router"]
