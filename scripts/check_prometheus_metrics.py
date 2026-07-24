"""验证 Prometheus/OpenMetrics 抓取端点的基本结构与数据库指标。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import urlopen


DEFAULT_PREFIXES = ("kbot_", "oracle_", "oracledb_")


@dataclass(frozen=True)
class MetricsSummary:
    help_count: int
    type_count: int
    sample_count: int
    family_count: int
    database_family_count: int


def validate_metrics_url(url: str) -> None:
    parsed = urlparse(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("指标地址必须是无凭据、无 Query 的 HTTP(S) URL")


def summarize_metrics(
    payload: str,
    *,
    required_prefixes: tuple[str, ...] = DEFAULT_PREFIXES,
) -> MetricsSummary:
    lines = [line for line in payload.splitlines() if line.strip()]
    help_count = sum(line.startswith("# HELP ") for line in lines)
    type_count = sum(line.startswith("# TYPE ") for line in lines)
    samples = [line for line in lines if not line.startswith("#")]
    families = {
        line.split("{", 1)[0].split(" ", 1)[0] for line in samples
    }
    database_families = {
        name
        for name in families
        if name.startswith(required_prefixes)
    }
    summary = MetricsSummary(
        help_count=help_count,
        type_count=type_count,
        sample_count=len(samples),
        family_count=len(families),
        database_family_count=len(database_families),
    )
    if min(help_count, type_count, len(samples), len(families)) <= 0:
        raise ValueError("指标端点缺少 HELP、TYPE 或 Sample")
    if not database_families:
        raise ValueError(
            f"指标端点缺少数据库指标前缀：{required_prefixes}"
        )
    return summary


def check_endpoint(url: str, *, timeout_seconds: float) -> MetricsSummary:
    validate_metrics_url(url)
    with urlopen(url, timeout=timeout_seconds) as response:
        if response.status != 200:
            raise RuntimeError(f"指标端点返回 HTTP {response.status}")
        payload = response.read(10 * 1024 * 1024 + 1)
    if len(payload) > 10 * 1024 * 1024:
        raise RuntimeError("指标正文超过 10 MiB 上限")
    return summarize_metrics(payload.decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5,
    )
    args = parser.parse_args()
    try:
        summary = check_endpoint(
            args.url,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"Prometheus 指标校验失败：{exc}")
        return 1
    print(
        "Prometheus 指标校验通过："
        f"HELP={summary.help_count} TYPE={summary.type_count} "
        f"Samples={summary.sample_count} "
        f"Families={summary.family_count} "
        f"DatabaseFamilies={summary.database_family_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
