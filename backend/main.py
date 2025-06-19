import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
import toml
from typing import Dict, Any
import logging

from backend.core.log.logger import setup_logging

def load_config() -> Dict[str, Any]:
    """Load and validate configuration from TOML files.
    
    Returns:
        Dict[str, Any]: Merged configuration dictionary
        
    Raises:
        FileNotFoundError: If main config file not found
        toml.TomlDecodeError: If config file is invalid
        ValueError: If required config is missing
    """
    try:
        config_path = os.getenv("CONFIG_PATH", "settings.toml")
        env_config_path = os.getenv("ENV_CONFIG_PATH", "env/development.toml")
        
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        config = toml.load(config_path)
        
        # Validate required config
        if "app" not in config:
            raise ValueError("Missing required 'app' section in config")
            
        if Path(env_config_path).exists():
            env_config = toml.load(env_config_path)
            config.update(env_config)
        
        return config
        
    except (toml.TomlDecodeError, FileNotFoundError, ValueError) as e:
        logging.error(f"Failed to load config: {str(e)}")
        raise

def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    try:
        config = load_config()
        
        # Setup logging from config file
        logging_config_path = config.get("logging_config", "settings.toml")
        setup_logging(logging_config_path)
        
        app = FastAPI(
            title=config["app"]["name"],
            description=config["app"]["description"],
            version=config["app"]["version"],
            debug=config["app"].get("debug", False),
        )
        
        # Add middleware with safer defaults
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.get("cors", {}).get("allow_origins", []),
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["Content-Type", "Authorization"],
        )
        
        # Add routers
        from backend.api.routers import router
        app.include_router(router)
        
        # Add health check endpoint
        @app.get("/health", tags=["health"])
        async def health_check() -> JSONResponse:
            return JSONResponse({"status": "ok"})
            
        # Add startup/shutdown events
        @app.on_event("startup")
        async def startup():
            logging.info("Application startup")
            
        @app.on_event("shutdown") 
        async def shutdown():
            logging.info("Application shutdown")
        
        return app
        
    except Exception as e:
        logging.critical(f"Failed to create application: {str(e)}")
        raise

app = create_app()