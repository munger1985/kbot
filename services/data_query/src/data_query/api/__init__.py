"""Data Query Internal API routers."""

from .management import router as management_router
from .managed_datasets import router as managed_dataset_router
from .runtime import router as runtime_router
from .model_references import router as model_reference_router

__all__ = ["managed_dataset_router", "management_router", "model_reference_router", "runtime_router"]
