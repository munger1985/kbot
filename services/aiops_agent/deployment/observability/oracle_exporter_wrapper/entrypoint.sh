#!/bin/sh
set -eu

# Exporter仅支持环境变量连接串；在容器启动后从只读 Secret加载，避免写入Compose。
secret_path=/run/secrets/oracle_exporter_dsn
if [ ! -r "${secret_path}" ]; then
  echo "错误：Oracle Exporter连接Secret不可读" >&2
  exit 1
fi
DATA_SOURCE_NAME=$(cat "${secret_path}")
export DATA_SOURCE_NAME
exec /oracledb_exporter "$@"
