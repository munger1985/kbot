# es_client_manager.py
import asyncio
from typing import Optional, Dict
from elasticsearch import AsyncElasticsearch
from loguru import logger

class ESClientManager:
    """ES连接管理单例类"""
    
    _instance: Optional['ESClientManager'] = None
    _clients: Dict[str, AsyncElasticsearch] = {}
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def get_client(self, connstr: dict) -> Optional[AsyncElasticsearch]:
        """获取ES客户端（单例）"""
        # 根据连接参数生成唯一key
        conn_key = self._generate_conn_key(connstr)
        
        async with self._lock:
            if conn_key in self._clients:
                client = self._clients[conn_key]
                # 检查连接是否仍然有效
                try:
                    if await client.ping():
                        logger.debug(f"使用现有的ES连接: {conn_key}")
                        return client
                    else:
                        logger.warning(f"ES连接已失效，重新创建: {conn_key}")
                        await client.close()
                        del self._clients[conn_key]
                except Exception:
                    logger.warning(f"ES连接检查失败，重新创建: {conn_key}")
                    if conn_key in self._clients:
                        try:
                            await self._clients[conn_key].close()
                        except Exception:
                            pass
                        del self._clients[conn_key]
            
            # 创建新连接
            try:
                client = await self._create_client(connstr)
                if client and await client.ping():
                    self._clients[conn_key] = client
                    logger.info(f"创建新的ES连接: {conn_key}")
                    return client
                else:
                    logger.error(f"创建ES连接失败: {conn_key}")
                    return None
            except Exception as e:
                logger.error(f"创建ES客户端异常: {e}")
                return None
    
    def _generate_conn_key(self, connstr: dict) -> str:
        """生成连接唯一标识"""
        hosts = connstr.get("hosts", [])
        if not isinstance(hosts, list):
            hosts = [hosts]
        
        key_parts = [
            "|".join(sorted(hosts)),
            connstr.get("user", ""),
            # 不包含密码在key中
        ]
        return hash("_".join(key_parts)) # type: ignore
    
    async def _create_client(self, connstr: dict) -> Optional[AsyncElasticsearch]:
        """创建ES客户端"""
        try:
            hosts = connstr.get("hosts")
            if not isinstance(hosts, list):
                hosts = [hosts]
            
            http_auth = None
            if connstr.get("user") and connstr.get("password"):
                http_auth = (connstr.get("user"), connstr.get("password"))
            
            es_params = {
                'hosts': hosts,
                'http_auth': http_auth,
                'verify_certs': connstr.get("verify_certs", True),
                'ca_certs': connstr.get("ca_certs")
            }
            
            # 移除None值参数
            es_params = {k: v for k, v in es_params.items() if v is not None}
            
            return AsyncElasticsearch(**es_params)
            
        except Exception as e:
            logger.error(f"创建ES客户端失败: {e}")
            return None
    
    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            for conn_key, client in self._clients.items():
                try:
                    await client.close()
                    logger.info(f"关闭ES连接: {conn_key}")
                except Exception as e:
                    logger.error(f"关闭ES连接失败 {conn_key}: {e}")
            self._clients.clear()
    
    async def close_client(self, connstr: dict):
        """关闭特定连接"""
        conn_key = self._generate_conn_key(connstr)
        async with self._lock:
            if conn_key in self._clients:
                try:
                    await self._clients[conn_key].close()
                    del self._clients[conn_key]
                    logger.info(f"关闭ES连接: {conn_key}")
                except Exception as e:
                    logger.error(f"关闭ES连接失败 {conn_key}: {e}")

# 全局单例实例
es_client_manager = ESClientManager()