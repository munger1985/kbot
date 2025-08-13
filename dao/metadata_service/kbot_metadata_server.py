from concurrent import futures
import asyncio
import grpc
import os
import platform
import signal
import sys
from typing import Optional, Sequence
from grpc import aio

from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health as grpc_health

from loguru import logger

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from dao.metadata_service.kbot_md_sys_conf_service import KbotMdSysConfService
from dao.metadata_service.kbot_md_sys_conf_pb2_grpc import add_KbotMdSysConfServiceServicer_to_server



class GracefulServer:
    """gRPC服务器封装，实现跨平台优雅关闭"""
    
    def __init__(self, max_workers: int = 10, max_message_length: int = 100 * 1024 * 1024):
        """
        初始化服务器
        
        :param max_workers: 线程池最大工作线程数
        :param max_message_length: 最大消息长度(字节)
        """
        self.server: Optional[aio.Server] = None
        self.shutdown_event = asyncio.Event()
        self.max_workers = max_workers
        self.max_message_length = max_message_length
        self.health_servicer: Optional[grpc_health.aio.HealthServicer] = None # type: ignore
    
    async def serve(self, host: str = '0.0.0.0', port: int = 50051, enable_health_check: bool = True):
        """
        启动gRPC服务器并处理优雅关闭
        
        :param host: 监听主机
        :param port: 监听端口
        :param enable_health_check: 是否启用健康检查
        """
        # 创建gRPC服务器
        self.server = aio.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[
                ('grpc.max_send_message_length', self.max_message_length),
                ('grpc.max_receive_message_length', self.max_message_length),
            ]
        )
        
        # 添加健康检查服务
        if enable_health_check:
            self.health_servicer = grpc_health.aio.HealthServicer() # type: ignore
            health_pb2_grpc.add_HealthServicer_to_server(self.health_servicer, self.server)
            logger.debug("Health check service enabled")
        
        # 添加业务服务
        add_KbotMdSysConfServiceServicer_to_server(KbotMdSysConfService(), self.server)
        
        # 绑定端口
        self.server.add_insecure_port(f'{host}:{port}')
        
        # 设置信号处理器
        self._setup_signal_handlers()
        
        # 启动服务器
        await self.server.start()
        logger.info(f"Server started on {host}:{port}")
        
        # 设置健康检查状态
        if self.health_servicer:
            await self._set_serving_status()
        
        # 等待关闭信号
        await self.shutdown_event.wait()
        
        # 执行优雅关闭
        await self._graceful_shutdown()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        if platform.system() != 'Windows':
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self._shutdown)
                logger.debug("Signal handlers installed for SIGINT and SIGTERM")
            except RuntimeError as e:
                logger.warning(f"Failed to install signal handlers: {e}")
    
    async def _set_serving_status(self):
        """设置健康检查状态"""
        if self.health_servicer:
            # 设置服务状态为健康
            await self.health_servicer.set(
                "kbot_md_sys_conf.KbotMdSysConfService",
                health_pb2.HealthCheckResponse.SERVING
            )
            # 设置整体状态
            await self.health_servicer.set(
                "",
                health_pb2.HealthCheckResponse.SERVING
            )
            # 更新日志
            logger.debug("Health status set to SERVING")
    
    async def _graceful_shutdown(self):
        """执行优雅关闭"""
        logger.info("Starting graceful shutdown...")
        
        # 更新健康状态
        if self.health_servicer:
            await self.health_servicer.set(
                "kbot_md_sys_conf.KbotMdSysConfService",
                health_pb2.HealthCheckResponse.NOT_SERVING
            )
            await self.health_servicer.enter_graceful_shutdown()
            logger.debug("Health status updated to NOT_SERVING")
        
        try:
            # 5秒宽限期，10秒超时
            await asyncio.wait_for(self.server.stop(grace=5), timeout=10) # type: ignore
            logger.info("Server shutdown completed gracefully")
        except asyncio.TimeoutError:
            logger.warning("Force shutdown after timeout")
            self.server.close() # type: ignore
            await self.server.wait_for_termination() # type: ignore
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise
    
    def _shutdown(self):
        """触发关闭事件"""
        if not self.shutdown_event.is_set():
            logger.info("Shutdown signal received")
            self.shutdown_event.set()





async def serve():
    """Start the gRPC server."""
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    add_KbotMdSysConfServiceServicer_to_server(KbotMdSysConfService(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    await server.wait_for_termination()


async def main():
    """主入口函数，封装服务器启动逻辑"""
    # 配置日志
    
    
    # 从环境变量获取配置
    host = os.getenv('GRPC_HOST', '0.0.0.0')
    port = int(os.getenv('GRPC_PORT', '50051'))
    max_workers = int(os.getenv('GRPC_MAX_WORKERS', '10'))
    
    # 创建并启动服务器
    server = GracefulServer(max_workers=max_workers)
    logger.info(f"Starting server on {host}:{port} with {max_workers} workers")
    
    try:
        await server.serve(host=host, port=port)
    except asyncio.CancelledError:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.critical(f"Server crashed: {e}")
        raise
    finally:
        logger.info("Server process completed")

def run():
    """跨平台启动入口"""
    if platform.system() == 'Windows':
        # Windows需要特殊处理
        asyncio.run(main())
    else:
        # Unix系统使用更灵活的事件循环管理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(main())
        except KeyboardInterrupt:
            logger.info("Server shutdown by user")
        except Exception as e:
            logger.exception(f"Server error: {e}")
            raise
        finally:
            loop.close()
            logger.info("Event loop closed")

if __name__ == '__main__':
    run()