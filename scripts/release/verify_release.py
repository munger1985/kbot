"""运行 KBot 4.0 发布前确定性检查并生成机器可读证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_PACKAGES = (
    "packages/platform_core/src/platform_core",
    "packages/platform_clients/src/platform_clients",
    "services/model_serving/src/model_serving",
    "services/knowledge_core/src/knowledge_core",
    "services/knowledge_retrieval_app/src/knowledge_retrieval_app",
    "services/agent_runtime/src/agent_runtime",
    "services/aiops_agent/src/aiops_agent",
    "services/data_query/src/data_query",
    "services/main_api/src/main_api",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tracked_inputs() -> list[Path]:
    roots = (
        ROOT / "database" / "oracle",
        ROOT / "configuration",
        ROOT / "docs" / "openapi",
        ROOT / "resources",
        ROOT / "packages" / "platform_core" / "src"
        / "platform_core" / "resources",
        ROOT / "services" / "aiops_agent" / "src"
        / "aiops_agent" / "resources",
    )
    files = [
        path
        for base in roots
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path != ROOT / "configuration" / "kbot.toml"
    ]
    for name in (
        "pyproject.toml",
        "requirements.txt",
        "resources/topology.toml",
        "packages/platform_core/pyproject.toml",
        "packages/platform_clients/pyproject.toml",
        "services/main_api/pyproject.toml",
        "services/agent_runtime/pyproject.toml",
        "services/knowledge_core/pyproject.toml",
        "services/knowledge_retrieval_app/pyproject.toml",
        "services/aiops_agent/pyproject.toml",
        "services/data_query/pyproject.toml",
        "services/model_serving/pyproject.toml",
    ):
        path = ROOT / name
        if path.is_file():
            files.append(path)
    return sorted(set(files))


def build_input_manifest() -> dict[str, str]:
    """冻结 DDL、配置样例、OpenAPI 和依赖声明的内容 Hash。"""
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in _tracked_inputs()
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def _checks(
    *,
    include_oracle: bool,
    include_external_databases: bool = False,
    prometheus_url: str | None = None,
) -> list[tuple[str, list[str], int]]:
    python = sys.executable
    checks = [
        (
            "compile_active_packages",
            [python, "-m", "compileall", "-q", *ACTIVE_PACKAGES],
            120,
        ),
        (
            "architecture_boundaries",
            [python, "tests/acceptance/check_4_0_boundaries.py"],
            120,
        ),
        (
            "oracle_ddl_contract",
            [python, "tests/acceptance/check_oracle_schema.py"],
            120,
        ),
        (
            "entity_table_ownership",
            [python, "tests/acceptance/check_entity_ownership.py"],
            120,
        ),
        (
            "process_topology",
            [python, "tests/acceptance/check_process_topology.py"],
            120,
        ),
        (
            "configuration_contract",
            [python, "tests/acceptance/check_configuration_contract.py"],
            120,
        ),
        (
            "openapi_contracts",
            [python, "tests/acceptance/check_openapi_contracts.py"],
            120,
        ),
        (
            "supply_chain_baseline",
            [python, "tests/acceptance/check_supply_chain.py"],
            120,
        ),
        (
            "aiops_diagnostic_catalog",
            [python, "tests/acceptance/check_aiops_diagnostic_catalog.py"],
            120,
        ),
        (
            "unit_component_contract",
            [
                python,
                "-m",
                "pytest",
                "-q",
                "tests/unit",
                "tests/contract",
            ],
            300,
        ),
        (
            "development_logs_http",
            [python, "tests/smoke/smoke_development_logs_http.py"],
            30,
        ),
    ]
    if include_oracle:
        checks.extend(
            (
                (
                    "oracle_object_catalog",
                    [python, "tests/acceptance/check_oracle_catalog.py"],
                    30,
                ),
                (
                    "oracle_all_entity_catalog",
                    [python, "tests/acceptance/check_oracle_entity_schema.py"],
                    30,
                ),
                (
                    "oracle_aiops_entity_catalog",
                    [python, "tests/acceptance/check_aiops_entity_schema.py"],
                    30,
                ),
                (
                    "oracle_cross_service_uow",
                    [python, "tests/smoke/smoke_oracle_service_uow.py"],
                    120,
                ),
                (
                    "oracle_aiops_persistence",
                    [python, "tests/smoke/smoke_aiops_persistence.py"],
                    120,
                ),
                (
                    "oracle_aiops_runtime",
                    [python, "tests/smoke/smoke_aiops_runtime.py"],
                    180,
                ),
                (
                    "oracle_data_query_runtime",
                    [python, "tests/smoke/smoke_data_query_oracle.py"],
                    120,
                ),
                (
                    "oracle_agent_memory",
                    [python, "tests/smoke/smoke_agent_memory.py"],
                    120,
                ),
                (
                    "oracle_knowledge_core_s3",
                    [python, "tests/smoke/smoke_knowledge_core_s3.py"],
                    120,
                ),
                (
                    "oracle_model_serving_s4",
                    [python, "tests/smoke/smoke_model_serving_oracle.py"],
                    120,
                ),
                (
                    "oracle_notifications_s6",
                    [python, "tests/smoke/smoke_notifications_oracle.py"],
                    120,
                ),
            )
        )
    if include_external_databases:
        checks.append(
            (
                "data_query_external_databases",
                [
                    python,
                    "tests/smoke/smoke_data_query_external_databases.py",
                ],
                120,
            )
        )
    if prometheus_url:
        checks.append(
            (
                "prometheus_metrics",
                [
                    python,
                    "tests/acceptance/check_prometheus_metrics.py",
                    "--url",
                    prometheus_url,
                ],
                30,
            )
        )
    return checks


def _run_check(
    name: str, command: list[str], timeout_seconds: int
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        exit_code = result.returncode
        output = (result.stdout + result.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (
            exc.stdout or ""
        )
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else (
            exc.stderr or ""
        )
        output = (
            f"{stdout}{stderr}\n检查超过 {timeout_seconds} 秒，已终止"
        ).strip()
    return {
        "name": name,
        "command": command,
        "status": "PASSED" if exit_code == 0 else "FAILED",
        "exit_code": exit_code,
        "timeout_seconds": timeout_seconds,
        "duration_seconds": round(time.monotonic() - started, 3),
        "output_sha256": hashlib.sha256(
            output.encode("utf-8")
        ).hexdigest(),
        "output_tail": output[-4000:],
    }


def verify(
    *,
    profile: str,
    include_oracle: bool,
    include_external_databases: bool,
    prometheus_url: str | None,
    require_clean: bool,
) -> dict[str, Any]:
    """执行发布检查；返回值可直接序列化为 JSON。"""
    dirty_paths = tuple(
        line[3:] for line in _git("status", "--short").splitlines()
    )
    results = [
        _run_check(name, command, timeout_seconds)
        for name, command, timeout_seconds in _checks(
            include_oracle=include_oracle,
            include_external_databases=include_external_databases,
            prometheus_url=prometheus_url,
        )
    ]
    failures = [
        item["name"]
        for item in results
        if item["status"] != "PASSED"
    ]
    if require_clean and dirty_paths:
        failures.append("clean_worktree")
    return {
        "schema_version": "KBotReleaseEvidence.v1",
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat(),
        "commit_sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "require_clean": require_clean,
        "oracle_checked": include_oracle,
        "prometheus_checked": prometheus_url is not None,
        "external_databases_checked": include_external_databases,
        "dirty_paths": dirty_paths,
        "input_manifest": build_input_manifest(),
        "checks": results,
        "status": "PASSED" if not failures else "FAILED",
        "failures": failures,
        "limitations": (
            "未包含真实模型质量、负载、Chaos、安全扫描和生产等价 E2E",
        ),
    }


def resolve_profile_options(
    *,
    profile: str,
    include_oracle: bool,
    include_external_databases: bool,
    require_clean: bool,
) -> tuple[bool, bool, bool]:
    """发布候选档位强制启用实库检查和干净工作树门禁。"""
    if profile == "rc":
        return True, True, True
    return include_oracle, include_external_databases, require_clean


def main() -> int:
    parser = argparse.ArgumentParser(
        description="执行 KBot 4.0 发布验证并输出 JSON 证据"
    )
    parser.add_argument(
        "--profile",
        choices=("developer", "rc"),
        default="developer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "release-evidence.json",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="额外执行真实 Oracle Entity/Catalog 校验",
    )
    parser.add_argument(
        "--prometheus-url",
        help="额外校验无凭据的 Prometheus/OpenMetrics 抓取地址",
    )
    parser.add_argument(
        "--external-databases",
        action="store_true",
        help=(
            "执行真实 PostgreSQL/MySQL Data Query Smoke；"
            "连接信息由 KBOT_DQ_SMOKE_* 环境变量提供"
        ),
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="工作树有任何改动时令验证失败",
    )
    args = parser.parse_args()
    include_oracle, include_external_databases, require_clean = (
        resolve_profile_options(
            profile=args.profile,
            include_oracle=args.oracle,
            include_external_databases=args.external_databases,
            require_clean=args.require_clean,
        )
    )
    evidence = verify(
        profile=args.profile,
        include_oracle=include_oracle,
        include_external_databases=include_external_databases,
        prometheus_url=args.prometheus_url,
        require_clean=require_clean,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"发布验证 {evidence['status']}，证据已写入：{output}")
    return 0 if evidence["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
