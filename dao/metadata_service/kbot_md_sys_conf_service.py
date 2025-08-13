import grpc
from dao.metadata_service.kbot_md_sys_conf_pb2 import (
    KbotMdSysConf as GrpcKbotMdSysConf,
    KbotMdSysConfList,
    DeleteResponse,
    Empty
)
from dao.metadata_service.kbot_md_sys_conf_pb2_grpc import KbotMdSysConfServiceServicer
from dao.entities.kbot_md_sys_conf import KbotMdSysConf
from dao.repositories.kbot_md_sys_conf_repo import KbotMdSysConfRepository

class KbotMdSysConfService(KbotMdSysConfServiceServicer):
    """gRPC service implementation for KBOT_MD_SYS_CONF operations."""
    
    async def _to_grpc_config(self, config: KbotMdSysConf) -> GrpcKbotMdSysConf:
        """Convert SQLAlchemy model to gRPC message."""
        return GrpcKbotMdSysConf(
            conf_id=int(config.conf_id), # type: ignore
            # 添加其他字段转换...
        )
    
    async def _from_grpc_config(self, grpc_config: GrpcKbotMdSysConf) -> KbotMdSysConf:
        """Convert gRPC message to SQLAlchemy model."""
        return KbotMdSysConf(
            conf_id=grpc_config.conf_id,
            # 添加其他字段转换...
        )
    
    async def Create(self, request: GrpcKbotMdSysConf, context) -> GrpcKbotMdSysConf:
        """Create a new system configuration record."""
        config = await self._from_grpc_config(request)
        created_config = await KbotMdSysConfRepository().create(config)
        return await self._to_grpc_config(created_config)
    
    async def GetById(self, request, context) -> GrpcKbotMdSysConf:
        """Get system configuration by ID."""
        config = await KbotMdSysConfRepository().get_by_id(request.conf_id)
        if config is None:
            await context.abort(grpc.StatusCode.NOT_FOUND, "Configuration not found")
        return await self._to_grpc_config(config) # type: ignore
    
    async def GetAll(self, request: Empty, context) -> KbotMdSysConfList:
        """Get all system configuration records."""
        configs = await KbotMdSysConfRepository().get_all()
        return KbotMdSysConfList(
            configs=[await self._to_grpc_config(config) for config in configs]
        )
    
    async def Update(self, request: GrpcKbotMdSysConf, context) -> GrpcKbotMdSysConf:
        """Update a system configuration record."""
        config = await self._from_grpc_config(request)
        updated_config = await KbotMdSysConfRepository().update(config)
        return await self._to_grpc_config(updated_config)
    
    async def Delete(self, request, context) -> DeleteResponse:
        """Delete a system configuration record by ID."""
        success = await KbotMdSysConfRepository().delete(request.conf_id)
        return DeleteResponse(success=success)