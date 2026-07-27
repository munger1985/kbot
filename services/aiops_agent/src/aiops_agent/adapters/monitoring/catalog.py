"""Metric Catalog 部署资产加载器。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aiops_agent.contracts.monitoring import (
    MetricCatalogDocument,
    MetricDefinition,
)


class MetricCatalog:
    def __init__(self, document: MetricCatalogDocument):
        self.version = document.catalog_version
        self._metrics = {
            item.metric_code: item for item in document.metrics
        }
        canonical = json.dumps(
            document.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.manifest_hash = hashlib.sha256(canonical).hexdigest()

    def get(self, metric_code: str) -> MetricDefinition:
        try:
            return self._metrics[metric_code]
        except KeyError as exc:
            raise KeyError(f"未知监控指标：{metric_code}") from exc

    def select(
        self, metric_codes: tuple[str, ...], *, db_type: str
    ) -> tuple[MetricDefinition, ...]:
        return tuple(
            item
            for code in metric_codes
            if db_type.upper()
            in (item := self.get(code)).supported_db_types
        )


def load_metric_catalog(path: Path | None = None) -> MetricCatalog:
    resolved = path or (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "metrics"
        / "baseline.v1.json"
    )
    document = MetricCatalogDocument.model_validate_json(
        resolved.read_text(encoding="utf-8")
    )
    return MetricCatalog(document)
