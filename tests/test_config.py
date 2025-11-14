# test_environment_switch.py
import os
import sys
from pathlib import Path

# Add both project root and backend directory to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config.settings import get_settings

# 加载环境变量
from dotenv import load_dotenv
env_path = project_root / ".env"
load_dotenv(env_path)

def test_environment_switch():
    """测试环境切换"""
    
    # 测试开发环境
    dev_settings = get_settings()
    print(f"Dev Environment: {dev_settings.environment}")
    print(f"Dev Debug: {dev_settings.app.debug}")
    print(f"Dev LLM Temp: {dev_settings.llm.temperature}")

if __name__ == "__main__":
    test_environment_switch()