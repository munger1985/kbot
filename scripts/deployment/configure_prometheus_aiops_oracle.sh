#!/usr/bin/env bash
set -Eeuo pipefail

# 为同机 Oracle、Oracle Exporter、Prometheus 和 KBot AIOps 安装指标映射与告警规则。
# 所有路径和标识均可通过同名环境变量覆盖。
PROMETHEUS_CONFIG="${PROMETHEUS_CONFIG:-/etc/prometheus/prometheus.yml}"
PROMETHEUS_RULE_DIR="${PROMETHEUS_RULE_DIR:-/etc/prometheus/rules}"
PROMETHEUS_SERVICE="${PROMETHEUS_SERVICE:-prometheus}"
PROMETHEUS_URL="${PROMETHEUS_URL:-http://127.0.0.1:9090}"
ORACLE_EXPORTER_URL="${ORACLE_EXPORTER_URL:-http://127.0.0.1:9161}"
ORACLE_JOB="${ORACLE_JOB:-oracle_db_monitor}"
ORACLE_INSTANCE="${ORACLE_INSTANCE:-oracle-dev-01}"

RULE_FILE="${PROMETHEUS_RULE_DIR}/kbot-aiops-oracle.yml"
OVERRIDE_FILE="${PROMETHEUS_RULE_DIR}/kbot-aiops-query-overrides.json"

fail() {
  echo "错误：$*" >&2
  exit 1
}

if [[ "${EUID}" -ne 0 ]]; then
  fail "请使用 sudo 运行此脚本"
fi

for command_name in curl grep awk install mktemp promtool systemctl; do
  command -v "${command_name}" >/dev/null 2>&1 \
    || fail "缺少命令：${command_name}"
done

[[ -f "${PROMETHEUS_CONFIG}" ]] \
  || fail "Prometheus 配置不存在：${PROMETHEUS_CONFIG}"
[[ "${ORACLE_JOB}" =~ ^[A-Za-z0-9_.:-]+$ ]] \
  || fail "ORACLE_JOB 包含非法字符"
[[ "${ORACLE_INSTANCE}" =~ ^[A-Za-z0-9_.:-]+$ ]] \
  || fail "ORACLE_INSTANCE 包含非法字符"

temporary_dir="$(mktemp -d)"
cleanup() {
  rm -rf -- "${temporary_dir}"
}
trap cleanup EXIT

metrics_file="${temporary_dir}/oracle.metrics"
curl -fsS --max-time 15 "${ORACLE_EXPORTER_URL}/metrics" >"${metrics_file}" \
  || fail "无法读取 Oracle Exporter 指标"
curl -fsS --max-time 15 "${PROMETHEUS_URL}/-/ready" >/dev/null \
  || fail "Prometheus 未就绪"

required_metrics=(
  oracledb_up
  oracledb_sessions_value
  oracledb_activity_user_commits
  oracledb_activity_user_rollbacks
  oracledb_tablespace_used_percent
  oracledb_tablespace_free_bytes
  oracledb_tablespace_max_bytes
  oracledb_exporter_last_scrape_error
  oracledb_exporter_last_scrape_duration_seconds
  oracledb_resource_current_utilization
  oracledb_resource_limit_value
  oracledb_kbot_cpu_utilization_percent
  oracledb_kbot_errors_total
)
for metric_name in "${required_metrics[@]}"; do
  grep -Eq "^${metric_name}(\{|[[:space:]])" "${metrics_file}" \
    || fail "Oracle Exporter 缺少必要指标：${metric_name}"
done

rule_candidate="${temporary_dir}/kbot-aiops-oracle.yml"
cat >"${rule_candidate}" <<EOF
groups:
  - name: kbot-aiops-oracle-recording
    interval: 15s
    rules:
      - record: kbot_db_active_connections
        expr: sum by (instance, target_key) (oracledb_sessions_value{job="${ORACLE_JOB}", status="ACTIVE", type="USER"})

      - record: kbot_db_cpu_utilization_percent
        expr: oracledb_kbot_cpu_utilization_percent{job="${ORACLE_JOB}"}

      - record: kbot_db_connection_utilization_percent
        expr: 100 * oracledb_resource_current_utilization{job="${ORACLE_JOB}", resource_name="sessions"} / oracledb_resource_limit_value{job="${ORACLE_JOB}", resource_name="sessions"} > 0

      - record: kbot_db_transactions_total
        expr: sum by (instance, target_key) (oracledb_activity_user_commits{job="${ORACLE_JOB}"}) + sum by (instance, target_key) (oracledb_activity_user_rollbacks{job="${ORACLE_JOB}"})

      - record: kbot_db_response_latency_milliseconds
        expr: oracledb_exporter_last_scrape_duration_seconds{job="${ORACLE_JOB}"} * 1000

      - record: kbot_db_storage_utilization_percent
        expr: oracledb_tablespace_used_percent{job="${ORACLE_JOB}"}

      - record: kbot_db_storage_free_bytes
        expr: oracledb_tablespace_free_bytes{job="${ORACLE_JOB}"}

      - record: kbot_db_storage_max_bytes
        expr: oracledb_tablespace_max_bytes{job="${ORACLE_JOB}"}

      - record: kbot_db_errors_total
        expr: oracledb_kbot_errors_total{job="${ORACLE_JOB}"}

  - name: kbot-aiops-oracle-alerts
    interval: 15s
    rules:
      - alert: OracleExporterTargetMissing
        expr: absent(up{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"})
        for: 2m
        labels:
          severity: critical
          instance: "${ORACLE_INSTANCE}"
        annotations:
          summary: "Oracle Exporter Target 已从 Prometheus 消失"
          description: "Prometheus 连续 2 分钟未发现 ${ORACLE_INSTANCE} 的 Oracle Exporter Target。"

      - alert: OracleExporterDown
        expr: up{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Oracle Exporter 不可用"
          description: "Prometheus 连续 2 分钟无法抓取 ${ORACLE_INSTANCE}。"

      - alert: OracleDatabaseUnavailable
        expr: oracledb_up{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Oracle 数据库不可用"
          description: "Exporter 存活，但连续 1 分钟无法连接 ${ORACLE_INSTANCE} 的 Oracle 数据库。"

      - alert: OracleExporterScrapeFailed
        expr: oracledb_exporter_last_scrape_error{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"} > 0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Oracle Exporter 数据库指标抓取失败"
          description: "Exporter 仍可访问，但数据库指标查询持续失败。"

      - alert: OracleTablespaceUsageHigh
        expr: oracledb_tablespace_used_percent{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"} > 85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Oracle 表空间使用率超过 85%"
          description: "表空间 {{ \$labels.tablespace }} 当前使用率为 {{ \$value | printf \"%.2f\" }}%。"

      - alert: OracleTablespaceUsageCritical
        expr: oracledb_tablespace_used_percent{job="${ORACLE_JOB}", instance="${ORACLE_INSTANCE}"} > 95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Oracle 表空间使用率超过 95%"
          description: "表空间 {{ \$labels.tablespace }} 当前使用率为 {{ \$value | printf \"%.2f\" }}%。"

      - alert: OracleHostCpuHigh
        expr: 100 * (1 - avg(rate(node_cpu_seconds_total{job="node", mode="idle"}[5m]))) > 90
        for: 10m
        labels:
          severity: warning
          instance: "${ORACLE_INSTANCE}"
        annotations:
          summary: "Oracle 主机 CPU 持续超过 90%"
          description: "数据库同机主机 CPU 已连续 10 分钟处于高负载。"

      - alert: OracleHostMemoryLow
        expr: 100 * node_memory_MemAvailable_bytes{job="node"} / node_memory_MemTotal_bytes{job="node"} < 10
        for: 10m
        labels:
          severity: warning
          instance: "${ORACLE_INSTANCE}"
        annotations:
          summary: "Oracle 主机可用内存低于 10%"
          description: "数据库同机主机可用内存已连续 10 分钟低于 10%。"

      - alert: OracleHostFilesystemLow
        expr: 100 * node_filesystem_avail_bytes{job="node", fstype!~"tmpfs|overlay|squashfs"} / node_filesystem_size_bytes{job="node", fstype!~"tmpfs|overlay|squashfs"} < 15
        for: 10m
        labels:
          severity: warning
          instance: "${ORACLE_INSTANCE}"
        annotations:
          summary: "Oracle 主机文件系统可用空间低于 15%"
          description: "挂载点 {{ \$labels.mountpoint }} 的可用空间为 {{ \$value | printf \"%.2f\" }}%。"
EOF

override_candidate="${temporary_dir}/kbot-aiops-query-overrides.json"
cat >"${override_candidate}" <<EOF
{
  "prometheus_queries": {
    "db.availability": "oracledb_up{instance=\"\${external_target}\"}",
    "db.cpu.utilization": "kbot_db_cpu_utilization_percent{instance=\"\${external_target}\"}",
    "db.connection.active": "kbot_db_active_connections{instance=\"\${external_target}\"}",
    "db.connection.utilization": "kbot_db_connection_utilization_percent{instance=\"\${external_target}\"}",
    "db.transaction.throughput": "rate(kbot_db_transactions_total{instance=\"\${external_target}\"}[5m])",
    "db.response.latency": "kbot_db_response_latency_milliseconds{instance=\"\${external_target}\"}",
    "db.storage.utilization": "kbot_db_storage_utilization_percent{instance=\"\${external_target}\"}",
    "db.storage.free_bytes": "kbot_db_storage_free_bytes{instance=\"\${external_target}\"}",
    "db.storage.max_bytes": "kbot_db_storage_max_bytes{instance=\"\${external_target}\"}",
    "db.error.rate": "rate(kbot_db_errors_total{instance=\"\${external_target}\"}[5m])"
  }
}
EOF

promtool check rules "${rule_candidate}"

config_candidate="${temporary_dir}/prometheus.yml"
if grep -Fq "${RULE_FILE}" "${PROMETHEUS_CONFIG}"; then
  cp "${PROMETHEUS_CONFIG}" "${config_candidate}"
else
  awk -v rule_file="${RULE_FILE}" '
    /^rule_files:[[:space:]]*$/ && !inserted {
      print
      print "  - \"" rule_file "\""
      inserted = 1
      next
    }
    { print }
    END {
      if (!inserted) {
        print ""
        print "rule_files:"
        print "  - \"" rule_file "\""
      }
    }
  ' "${PROMETHEUS_CONFIG}" >"${config_candidate}"
fi

mkdir -p "${PROMETHEUS_RULE_DIR}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
cp -a "${PROMETHEUS_CONFIG}" "${PROMETHEUS_CONFIG}.${timestamp}.bak"
if [[ -f "${RULE_FILE}" ]]; then
  cp -a "${RULE_FILE}" "${RULE_FILE}.${timestamp}.bak"
fi

install -m 0644 "${rule_candidate}" "${RULE_FILE}"
install -m 0644 "${override_candidate}" "${OVERRIDE_FILE}"
promtool check config "${config_candidate}"
cp "${config_candidate}" "${PROMETHEUS_CONFIG}"

if ! systemctl reload "${PROMETHEUS_SERVICE}"; then
  systemctl kill -s HUP "${PROMETHEUS_SERVICE}"
fi

loaded=false
rules_response="${temporary_dir}/rules.json"
for _attempt in 1 2 3 4 5; do
  if curl -fsS --max-time 10 "${PROMETHEUS_URL}/api/v1/rules" \
    >"${rules_response}" \
    && grep -Fq 'kbot-aiops-oracle-recording' "${rules_response}"; then
    loaded=true
    break
  fi
  sleep 1
done
[[ "${loaded}" == "true" ]] || fail "Prometheus 重载后未发现 KBot AIOps 规则"

echo "Prometheus AIOps Oracle 规则已安装：${RULE_FILE}"
echo "KBot Target Monitor mapping_overrides 已生成：${OVERRIDE_FILE}"
echo "已覆盖的AIOps基线：可用性、CPU、活动连接、连接利用率、事务吞吐、响应延迟、表空间和错误率"
