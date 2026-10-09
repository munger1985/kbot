#!/bin/sh
set -eu

read_secret() {
    variable_name="$1"
    secret_path="$2"
    eval "current_value=\${$variable_name:-}"
    if [ -z "$current_value" ] && [ -r "$secret_path" ]; then
        current_value="$(cat "$secret_path")"
        export "$variable_name=$current_value"
    fi
}

read_secret KBOT_ORACLE_PASSWORD /run/secrets/kbot_oracle_password
read_secret KBOT_MASTER_KEY /run/secrets/kbot_master_key

if [ -z "${KBOT_ORACLE_PASSWORD:-}" ]; then
    echo "缺少 Oracle 密码 Secret：/run/secrets/kbot_oracle_password" >&2
    exit 1
fi
if [ -z "${KBOT_MASTER_KEY:-}" ] || [ "${#KBOT_MASTER_KEY}" -lt 32 ]; then
    echo "KBot 主密钥 Secret 必须至少为 32 字节" >&2
    exit 1
fi

exec "$@"
