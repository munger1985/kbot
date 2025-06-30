import os
from dynaconf import Dynaconf
from pathlib import Path
from typing import Dict, Any

# Initialize configuration with simplified setup
current_env = os.getenv("KBOT_ENV", "default")

# Initialize the complete configuration
settings = Dynaconf(
    envvar_prefix="KBOT",
    settings_files=[
        Path(__file__).parent.parent.parent / "settings.toml",  # base configuration
        Path(__file__).parent / "env" / f"{current_env}.toml",  # environment specific configuration
        Path(__file__).parent.parent.parent / ".secret.toml",  # secret configuration
    ],
    environments=True,
    env=current_env,
    load_dotenv=True,
    env_switcher="KBOT_ENV",
    lowercase_read=True,  # Support lowercase access
    merge_enabled=True,   # Allow merging of configurations
)

def load_config() -> Dict[str, Any]:
    """Load and merge configuration files."""
    return dict(settings)