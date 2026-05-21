import asyncio
import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from mcp.server.models import InitializationOptions
import mcp.types as types
from starlette.responses import Response
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp_tools import KBAskTool
from core.config.settings import get_app_config
from starlette.middleware.cors import CORSMiddleware
from anyio import BrokenResourceError, EndOfStream

from core.config.settings import get_app_config
from core.logger import LogConfig, LogManager

# 1. 初始化 Server 实例
server = Server("kbot-mcp-server")
kb_ask_tool = KBAskTool()

# 2. 注册工具列表
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    ask_schema = kb_ask_tool.get_schema()
    return [
        types.Tool(
            name=kb_ask_tool.tool_name,
            description=kb_ask_tool.description,
            inputSchema=ask_schema
        )
    ]

# 3. 处理工具调用逻辑
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    print(f"DEBUG: MCP Tool Called -> Name: {name}")

    if not arguments:
            raise ValueError("Missing arguments")
    
    if name == kb_ask_tool.tool_name:
        results = await kb_ask_tool.execute(
            agent_id=int(arguments.get("agent_id")),  # type: ignore
            question=arguments.get("query", "")
        )
    else:
        raise ValueError(f"Unknown tool: {name}")
    
    import json
    return [
        types.TextContent(
            type="text",
            text=json.dumps(results, ensure_ascii=False, indent=2)
        )
    ]

# 4. 创建 SSE 传输实例
sse = SseServerTransport("/messages/")

# 5. 设置路由映射
async def handle_sse(request):
    """处理 SSE 连接"""
    try:
        async with sse.connect_sse(
            request.scope, 
            request.receive, 
            request._send
        ) as (read_stream, write_stream): # type: ignore
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="kbot-mcp-service",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )
    except (asyncio.CancelledError, BrokenResourceError, EndOfStream):
        # 捕获客户端断开或服务器停止导致的流中断
        pass
    except Exception as e:
        logger.error(f"SSE Error: {e}")
    
    return Response(content="")

# 6. 组装 Starlette 应用
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app_config = get_app_config()

def setup_mcp_logging():
    """初始化 MCP 专属日志配置"""
    try:
        # 1. 构造日志配置 (可以共用主服务的日志目录，或者在 settings 增加 mcp 专属配置)
        log_conf = LogConfig(
            service_name="kbot-mcp-service", # 也可以从 app_config 动态获取
            log_dir=app_config.log.dir,
            level=app_config.log.level,
            rotation=app_config.log.rotation,
            retention=app_config.log.retention,
        )
        
        # 2. 启动 LogManager
        LogManager(log_conf).setup()
        logger.info("MCP Logging system initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize logging: {e}") # 如果日志初始化失败，退回到 print

# 7. 启动并设置端口
if __name__ == "__main__":
    setup_mcp_logging()
    # 在这里设置 host 和 port
    host = app_config.service_host
    port = app_config.mcp_port
    uvicorn.run(app, host=host, port=port)