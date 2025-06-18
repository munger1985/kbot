import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import toml

from backend.core.log.logger import setup_logging

def load_config() -> dict:
    """Load configuration from TOML files."""
    config_path = os.getenv("CONFIG_PATH", "core/config/settings.toml")
    env_config_path = os.getenv("ENV_CONFIG_PATH", "core/config/env/development.toml")
    
    config = toml.load(config_path)
    
    if Path(env_config_path).exists():
        env_config = toml.load(env_config_path)
        config.update(env_config)
    
    return config

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = load_config()
    
    # Setup logging
    setup_logging()
    
    app = FastAPI(
        title=config["app"]["name"],
        description=config["app"]["description"],
        version=config["app"]["version"],
        debug=config["app"]["debug"],
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routers
    from backend.api.routers import router
    app.include_router(router)
    
    return app

app = create_app()