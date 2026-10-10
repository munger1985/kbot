"""Grafana 固定 UID 与受控 LINK/DISABLED 安全门禁。"""

from __future__ import annotations

from dataclasses import dataclass

from platform_core.contracts.aiops.monitoring import (
    GrafanaDashboardUid,
    MonitoringDashboardIntegration,
)


@dataclass(frozen=True)
class GrafanaLinkSecurity:
    non_admin_identity: bool = False
    target_locked: bool = False
    domain_isolated: bool = False
    audit_enabled: bool = False
    https_upstream: bool = False

    @property
    def ready(self) -> bool:
        return all(
            (
                self.non_admin_identity,
                self.target_locked,
                self.domain_isolated,
                self.audit_enabled,
                self.https_upstream,
            )
        )


def resolve_grafana_integration(
    *,
    source_type: str,
    profile_dashboard_uid: GrafanaDashboardUid | None,
    instance_count: int,
    launch_url: str | None = None,
    security: GrafanaLinkSecurity | None = None,
) -> MonitoringDashboardIntegration:
    dashboard_uid: GrafanaDashboardUid | None = (
        "kbot-database-fleet"
        if instance_count > 1
        else profile_dashboard_uid
    )
    if source_type != "PROMETHEUS":
        return MonitoringDashboardIntegration(
            integration_mode="DISABLED",
            dashboard_uid=dashboard_uid,
            reason_code="GRAFANA_SOURCE_UNSUPPORTED",
        )
    if dashboard_uid is None:
        return MonitoringDashboardIntegration(
            integration_mode="DISABLED",
            reason_code="GRAFANA_DASHBOARD_UNAVAILABLE",
        )
    if launch_url is None or security is None:
        return MonitoringDashboardIntegration(
            integration_mode="DISABLED",
            dashboard_uid=dashboard_uid,
            reason_code="GRAFANA_SECURE_GATEWAY_UNAVAILABLE",
        )
    if not security.ready or not _safe_launch_url(launch_url):
        return MonitoringDashboardIntegration(
            integration_mode="DISABLED",
            dashboard_uid=dashboard_uid,
            reason_code="GRAFANA_SECURITY_GATE_FAILED",
        )
    return MonitoringDashboardIntegration(
        integration_mode="LINK",
        dashboard_uid=dashboard_uid,
        launch_url=launch_url,
    )


def _safe_launch_url(value: str) -> bool:
    lowered = value.lower()
    return (
        value.startswith("/api/v1/apps/aiops/")
        and "?" not in value
        and "#" not in value
        and "://" not in value
        and ".." not in value
        and "token" not in lowered
        and "var-" not in lowered
        and "locator" not in lowered
    )


__all__ = ["GrafanaLinkSecurity", "resolve_grafana_integration"]
