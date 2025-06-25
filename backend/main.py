import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
)

from .api.routers import router
from backend.core.log.logger import setup_logging, logger
from backend.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application starting up")
    yield
    logging.info("Application shutting down")



def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    try:
        # Initiate loguru configuration
        setup_logging()
        logger.debug("Starting application initialization")

        async def lifespan(app: FastAPI):
            # Startup logic
            logger.info("Application startup")
            yield
            # Shutdown logic
            logger.info("Application shutdown")

        app = FastAPI(
            title=settings.app.name,
            description=settings.app.description,
            version=settings.app.version,
            debug=settings.app.debug,
            lifespan=lifespan,
        )
        
        # Add middleware with safer defaults
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routers
        app.include_router(router)
        
        # Add health check endpoint
        @app.get("/health", tags=["health"])
        async def health_check() -> JSONResponse:
            return JSONResponse({"status": "ok"})

        # Add API documentation endpoint 
        @app.get("/docs", include_in_schema=False)
        async def custom_swagger_ui_html():
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=app.title + " - Swagger UI",
                swagger_js_url="/static/swagger-ui-bundle.js",
                swagger_css_url="/static/swagger-ui.css",
            )
        
        # Add API documentation endpoint
        @app.get("/redoc", include_in_schema=False)
        async def redoc_html():
            return get_redoc_html(
                openapi_url=app.openapi_url,
                title=app.title + " - ReDoc",
                redoc_js_url="/static/redoc.standalone.js",
            )

        return app
        
    except Exception as e:
        logger.critical(f"Failed to create application: {str(e)}")
        raise

if __name__ == "__main__":
    
    app = create_app()
    logger.info("Application created, starting Uvicorn server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    app = create_app()
    logger.info("Application created as module")