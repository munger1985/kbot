"""Model Serving 持久化入口。"""

from .uow import ModelServingUnitOfWork, create_model_serving_uow_factory

__all__ = ["ModelServingUnitOfWork", "create_model_serving_uow_factory"]
