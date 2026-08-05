"""Data Query Internal API routers."""

from .management import router as management_router
from .runtime import router as runtime_router
from .model_references import router as model_reference_router

__all__ = ["management_router", "model_reference_router", "runtime_router"]
