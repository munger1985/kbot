"""Compatibility launcher; implementation lives in apps.ai_models_llm."""
from apps.ai_models_llm.main import app

__all__ = ["app"]

if __name__ == "__main__":
    import uvicorn
    from apps.ai_models_llm.main import SERVICE_HOST, SERVICE_PORT
    uvicorn.run(app, host=SERVICE_HOST, port=SERVICE_PORT, log_config=None)
