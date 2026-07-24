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


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PACKAGES = (
    "platform_core",
    "platform_clients",
    "model_serving",
    "knowledge_core",
    "agent_runtime",
    "aiops_agent",
    "main_api",
    "apps",
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
        ROOT / "configuration" / "example",
        ROOT / "docs" / "openapi",
    )
    files = [
        path
        for base in roots
        for path in base.rglob("*")
        if path.is_file()
    ]
    for name in (
        "requirements.txt",
        "configuration/process_topology.toml",
        "release/sbom/python-direct.cdx.json",
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
    *, include_oracle: bool
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
            [python, "scripts/check_4_0_boundaries.py"],
            120,
        ),
        (
            "oracle_ddl_contract",
            [python, "scripts/check_oracle_schema.py"],
            120,
        ),
        (
            "entity_table_ownership",
            [python, "scripts/check_entity_ownership.py"],
            120,
        ),
        (
            "process_topology",
            [python, "scripts/check_process_topology.py"],
            120,
        ),
        (
            "configuration_contract",
            [python, "scripts/check_configuration_contract.py"],
            120,
        ),
        (
            "openapi_contracts",
            [python, "scripts/check_openapi_contracts.py"],
            120,
        ),
        (
            "supply_chain_baseline",
            [python, "scripts/check_supply_chain.py"],
            120,
        ),
        (
            "aiops_diagnostic_catalog",
            [python, "scripts/check_aiops_diagnostic_catalog.py"],
            120,
        ),
        (
            "unit_component_contract",
            [python, "-m", "unittest", "discover", "-s", "tests"],
            300,
        ),
    ]
    if include_oracle:
        checks.extend(
            (
                (
                    "oracle_object_catalog",
                    [python, "scripts/check_oracle_catalog.py"],
                    30,
                ),
                (
                    "oracle_aiops_entity_catalog",
                    [python, "scripts/check_aiops_entity_schema.py"],
                    30,
                ),
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
    require_clean: bool,
) -> dict[str, Any]:
    """执行发布检查；返回值可直接序列化为 JSON。"""
    dirty_paths = tuple(
        line[3:] for line in _git("status", "--short").splitlines()
    )
    results = [
        _run_check(name, command, timeout_seconds)
        for name, command, timeout_seconds in _checks(
            include_oracle=include_oracle
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
        "dirty_paths": dirty_paths,
        "input_manifest": build_input_manifest(),
        "checks": results,
        "status": "PASSED" if not failures else "FAILED",
        "failures": failures,
        "limitations": (
            "未包含真实模型质量、负载、Chaos、安全扫描和生产等价 E2E",
        ),
    }


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
        "--require-clean",
        action="store_true",
        help="工作树有任何改动时令验证失败",
    )
    args = parser.parse_args()
    evidence = verify(
        profile=args.profile,
        include_oracle=args.oracle,
        require_clean=args.require_clean,
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
