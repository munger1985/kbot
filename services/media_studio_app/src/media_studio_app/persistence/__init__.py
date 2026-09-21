"""多媒体创作工作台持久化边界。"""

from .uow import MediaStudioAppUnitOfWork, create_media_studio_app_uow

__all__ = ["MediaStudioAppUnitOfWork", "create_media_studio_app_uow"]
