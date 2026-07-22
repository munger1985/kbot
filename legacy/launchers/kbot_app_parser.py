"""Compatibility launcher; deployable implementation lives in apps.knowledge_core_parser."""
from apps.knowledge_core_parser.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    from apps.knowledge_core_parser.main import SERVICE_HOST, SERVICE_PORT
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_config=None, loop="asyncio")
