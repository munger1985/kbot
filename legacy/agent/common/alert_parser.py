"""告警解析器 — 将异构监控告警统一为 AIOps 标准 alert_context。

支持格式:
  - Prometheus AlertManager v4 webhook
  - Zabbix Action webhook (JSON)
"""
from typing import Any


class AlertParser:
    """将异构告警统一为 OpsContextMemory.alert_context 字典。"""

    @staticmethod
    def parse(payload: dict[str, Any]) -> dict[str, Any]:
        """自动检测告警来源并解析。

        Returns:
            {"source": "prometheus"|"zabbix", "alerts": [...], "summary": "..."}
        """
        if AlertParser._is_prometheus(payload):
            return AlertParser.parse_prometheus_alertmanager(payload)
        if AlertParser._is_zabbix(payload):
            return AlertParser.parse_zabbix_action(payload)
        return AlertParser._parse_generic(payload)

    @staticmethod
    def _is_prometheus(payload: dict) -> bool:
        return "alerts" in payload and isinstance(payload.get("alerts"), list)

    @staticmethod
    def _is_zabbix(payload: dict) -> bool:
        return "jsonrpc" not in payload and any(
            k in payload for k in ("trigger_name", "host_name", "trigger_status")
        )

    @staticmethod
    def parse_prometheus_alertmanager(payload: dict) -> dict[str, Any]:
        """AlertManager v4 webhook 格式。"""
        alerts: list[dict] = []
        for alert in payload.get("alerts", []):
            annotations = alert.get("annotations", {})
            labels = alert.get("labels", {})
            alerts.append({
                "status": alert.get("status", "firing"),
                "alertname": labels.get("alertname", ""),
                "severity": labels.get("severity", ""),
                "instance": labels.get("instance", ""),
                "job": labels.get("job", ""),
                "summary": annotations.get("summary", ""),
                "description": annotations.get("description", ""),
                "starts_at": alert.get("startsAt", ""),
            })

        summary_parts = []
        for a in alerts:
            summary_parts.append(
                f"[{a['severity'].upper()}] {a['alertname']}: {a['summary']} "
                f"(instance={a['instance']})"
            )

        return {
            "source": "prometheus",
            "alerts": alerts,
            "summary": "\n".join(summary_parts),
            "highest_severity": AlertParser._highest_severity(a["severity"] for a in alerts),
        }

    @staticmethod
    def parse_zabbix_action(payload: dict) -> dict[str, Any]:
        """Zabbix Action webhook JSON 格式。"""
        alerts = [{
            "status": payload.get("trigger_status", "PROBLEM"),
            "alertname": payload.get("trigger_name", ""),
            "severity": payload.get("trigger_severity", ""),
            "host": payload.get("host_name", ""),
            "item": payload.get("item_name", ""),
            "value": payload.get("item_value", ""),
            "description": payload.get("trigger_description", ""),
        }]

        return {
            "source": "zabbix",
            "alerts": alerts,
            "summary": (
                f"[{alerts[0]['severity']}] {alerts[0]['alertname']}: "
                f"{alerts[0]['description']} (host={alerts[0]['host']})"
            ),
            "highest_severity": alerts[0]["severity"],
        }

    @staticmethod
    def _parse_generic(payload: dict) -> dict[str, Any]:
        return {
            "source": "generic",
            "alerts": [payload],
            "summary": str(payload.get("message", payload.get("summary", str(payload)))),
            "highest_severity": "warning",
        }

    @staticmethod
    def _highest_severity(severities: list[str]) -> str:
        order = {"info": 0, "warning": 1, "critical": 2, "page": 3, "disaster": 4}
        max_s = "info"
        max_v = 0
        for s in severities:
            v = order.get(s.lower(), 0)
            if v > max_v:
                max_v = v
                max_s = s.lower()
        return max_s
