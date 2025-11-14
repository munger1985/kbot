import re
import json
from elasticsearch import Elasticsearch
from typing import Any
from datetime import datetime
from loguru import logger
from core.config.settings import get_eslog_config


class EslogService:
    def __init__(self):
        es_config = get_eslog_config()
        self.es = Elasticsearch(
            es_config.hosts,
            basic_auth=(es_config.username, es_config.password)
            # ca_certs=es_config.ca_certs  # 指定证书路径
        )
        self.es_index = es_config.index
    
    async def get_recent_logs(self, size: int = 100) -> list[dict[str, Any]]:
        """
        获取最新的日志记录，按时间升序排列
        """
        try:
            query = {
                "size": size,
                "sort": [{"@timestamp": {"order": "desc"}}],
                "query": {"match_all": {}}
            }
            
            response = self.es.search(
                index=self.es_index,
                body=query
            )
            
            logs = self._format_logs(response) # type: ignore
            logs.reverse()
            
            return logs
            
        except Exception as e:
            logger.exception(f"Error fetching recent logs: {e}")
            return []
    
    
    async def search_logs(self, 
                     start_time: datetime,
                     end_time: datetime,
                     host: str | None = None,
                     log_level: str | None = None,
                     message: str | None = None,
                     size: int = 100) -> dict[str, Any]:
        """
        根据条件搜索日志
        """
        try:
            # 使用query_string构建更灵活的查询
            query_parts = []
            
            # 时间范围
            query_parts.append(f'@timestamp:[{start_time.isoformat()} TO {end_time.isoformat()}]')
            
            # 主机名
            if host:
                query_parts.append(f'host.name:"{host}"')
            
            # 日志级别 - 使用通配符匹配不同的空格数量
            if log_level:
                # 匹配 "| INFO     |" 或 "| WARNING  |" 等格式
                query_parts.append(f'message:"| {log_level}*|"')
            
            # 消息内容
            if message:
                query_parts.append(f'message:"*{message}*"')
            
            # 构建完整的query_string
            query_string = " AND ".join(query_parts)
            
            query = {
                "size": size,
                "sort": [{"@timestamp": {"order": "desc"}}],
                "query": {
                    "query_string": {
                        "query": query_string,
                        "default_field": "message"
                    }
                }
            }
            
            logger.debug(f"Query String: {query_string}")
            logger.debug(f"ES查询: {json.dumps(query, indent=2, default=str)}")
            
            response = self.es.search(
                index=self.es_index,
                body=query
            )
            
            logs = self._format_logs(response) # type: ignore
            total = response["hits"]["total"]["value"]
            
            logger.debug(f"找到 {total} 条记录")
            
            return {
                "total": total,
                "logs": logs
            }
        except Exception as e:
            logger.exception(f"搜索日志时发生错误: {str(e)}")
            return {"total": 0, "logs": []}
    
    def _format_logs(self, es_response: dict[str, Any]) -> list[dict[str, Any]]:
        """
        格式化ES返回的日志数据
        """
        logs = []
        hits = es_response.get("hits", {}).get("hits", [])
        logger.debug(f"格式化 {len(hits)} 条日志记录")
        
        for i, hit in enumerate(hits):
            source = hit["_source"]
            message = source.get("message", "")
            
            log_entry = {
                "timestamp": source.get("@timestamp"),
                "host": source.get("host", {}).get("name", ""),
                "level": self._extract_log_level(message),
                "message": message,
                "logger": self._extract_logger(message),
                "thread": source.get("thread", {}).get("name", "")
            }
            logs.append(log_entry)
        
        return logs
    
    def _extract_log_level(self, message: str) -> str:
        """从日志消息中提取日志级别"""
        if not message:
            return ""
        
        level_patterns = [
            r'\|\s*(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\s*\|',
            r'\b(DEBUG|INFO|WARN|WARNING|ERROR|FATAL|CRITICAL)\b',
        ]
        
        for pattern in level_patterns:
            match = re.search(pattern, message.upper())
            if match:
                level = match.group(1)
                return "WARN" if level == "WARNING" else level
        
        return ""
    
    def _extract_logger(self, message: str) -> str:
        """从日志消息中提取logger名称"""
        if not message:
            return ""
        
        pattern = r'\|\s*[A-Z]+\s*\|\s*([a-zA-Z0-9_.]+):'
        match = re.search(pattern, message)
        return match.group(1) if match else ""