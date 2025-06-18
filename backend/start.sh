#!/bin/bash

export PYTHONPATH=/home/chris/kbot
export CONFIG_PATH=/home/chris/kbot/backend/core/config/settings.toml
export ENV_CONFIG_PATH=/home/chris/kbot/backend/core/config/env/development.toml
export LOGGING_CONFIG=/home/chris/kbot/backend/core/config/settings.toml

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload