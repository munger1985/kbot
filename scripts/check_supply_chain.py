"""校验直接依赖锁、直接依赖 SBOM 与受 Git 跟踪文件中的 Secret。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from uuid import NAMESPACE_URL, uuid5


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
SBOM_PATH = ROOT / "release" / "sbom" / "python-direct.cdx.json"
REQUIREMENT_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]*)(?P<extras>\[[^]]+])?"
    r"==(?P<version>[A-Za-z0-9][A-Za-z0-9_.+!-]*)$"
)
SECRET_PATTERNS = {
    "private_key": re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        rb"[\s\S]{32,}?"
        rb"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "aws_access_key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "openai_api_key": re.compile(rb"\bsk-[A-Za-z0-9_-]{32,}\b"),
    "google_api_key": re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
}
SENSITIVE_FILENAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
}
SENSITIVE_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_requirements() -> tuple[tuple[str, str], ...]:
    dependencies: list[tuple[str, str]] = []
    for line_number, raw in enumerate(
        REQUIREMENTS.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        match = REQUIREMENT_PATTERN.fullmatch(value)
        if not match:
            raise ValueError(
                f"requirements.txt:{line_number} 必须使用 name==version 精确锁定"
            )
        dependencies.append(
            (
                _normalize_name(match.group("name")),
                match.group("version"),
            )
        )
    names = [name for name, _ in dependencies]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"requirements.txt 存在重复依赖：{duplicates}")
    return tuple(sorted(dependencies))


def build_sbom() -> dict[str, Any]:
    dependencies = load_requirements()
    requirements_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    identity = "\n".join(f"{name}=={version}" for name, version in dependencies)
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid5(NAMESPACE_URL, identity)}",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "bom-ref": "pkg:generic/kbot@4.0.0",
                "name": "kbot",
                "version": "4.0.0",
            },
            "properties": [
                {
                    "name": "kbot:requirements:sha256",
                    "value": requirements_hash,
                },
                {
                    "name": "kbot:sbom:scope",
                    "value": "python-direct-dependencies",
                },
            ],
        },
        "components": [
            {
                "type": "library",
                "bom-ref": f"pkg:pypi/{name}@{version}",
                "name": name,
                "version": version,
                "purl": f"pkg:pypi/{name}@{version}",
            }
            for name, version in dependencies
        ],
    }


def _tracked_files() -> tuple[Path, ...]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    )
    return tuple(
        ROOT / value.decode("utf-8")
        for value in output.split(b"\0")
        if value and (ROOT / value.decode("utf-8")).is_file()
    )


def scan_tracked_secrets() -> list[str]:
    findings: list[str] = []
    for path in _tracked_files():
        relative = path.relative_to(ROOT)
        if (
            path.name in SENSITIVE_FILENAMES
            or path.suffix.lower() in SENSITIVE_SUFFIXES
        ):
            findings.append(f"{relative}: 禁止跟踪敏感文件类型")
            continue
        try:
            payload = path.read_bytes()
        except OSError as exc:
            findings.append(f"{relative}: 无法读取：{exc}")
            continue
        if b"\0" in payload:
            continue
        for secret_type, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                findings.append(f"{relative}: 检测到 {secret_type}")
    return findings


def check_supply_chain() -> list[str]:
    errors: list[str] = []
    try:
        expected_sbom = build_sbom()
    except ValueError as exc:
        return [str(exc)]
    if not SBOM_PATH.is_file():
        errors.append(f"缺少直接依赖 SBOM：{SBOM_PATH.relative_to(ROOT)}")
    else:
        try:
            actual_sbom = json.loads(SBOM_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"直接依赖 SBOM 无法读取：{exc}")
        else:
            if actual_sbom != expected_sbom:
                errors.append("直接依赖 SBOM 与 requirements.txt 不一致")
    errors.extend(scan_tracked_secrets())
    return errors


def _write_sbom() -> None:
    SBOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    SBOM_PATH.write_text(
        json.dumps(build_sbom(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"直接依赖 SBOM 已写入：{SBOM_PATH.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-sbom",
        action="store_true",
        help="根据 requirements.txt 重新生成直接依赖 SBOM",
    )
    args = parser.parse_args()
    if args.write_sbom:
        _write_sbom()
    errors = check_supply_chain()
    if errors:
        print("KBot 供应链基线校验失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "KBot 供应链基线校验通过："
        f"{len(load_requirements())} 个精确锁定的直接依赖，"
        f"{len(_tracked_files())} 个受跟踪文件已扫描"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
