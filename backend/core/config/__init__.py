import os
from pathlib import Path
from typing import Dict, Any

import toml

def load_config() -> Dict[str, Any]:
    """Load and merge configuration files."""
    # Load base config
    default_path = Path(__file__).parent.parent / "settings.toml"
    config_path = os.getenv("CONFIG_PATH", str(default_path))
    base_config = toml.load(Path(config_path))
    
    # Load environment specific config
    env_config_path = os.getenv("ENV_CONFIG_PATH")
    if env_config_path and Path(env_config_path).exists():
        env_config = toml.load(Path(env_config_path))
        # Merge configs with environment overriding base
        for section, values in env_config.items():
            if section in base_config:
                base_config[section].update(values)
            else:
                base_config[section] = values
    
    return base_config