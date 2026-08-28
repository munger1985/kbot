"""校验AI DBA聊天专业评测集的覆盖面与安全断言。"""

from __future__ import annotations

import json
from pathlib import Path


DATASET = Path(__file__).with_name("aiops_dba_chat_cases.json")
REQUIRED_BEHAVIORS = {
    "PERSIST_USER_EVIDENCE",
    "USE_OFFLINE_SOURCES",
    "QUERY_DATABASE",
    "QUERY_PROMETHEUS",
    "CITE_EVIDENCE",
    "DISTINGUISH_CUMULATIVE_FROM_WINDOW",
    "SEPARATE_DIAGNOSIS_FROM_CHANGE",
    "IDENTIFY_MEASUREMENT_CONFLICT",
    "RECORD_EXACT_PRIVILEGE_GAP",
    "ATTEMPT_ENABLED_SOURCE_ONCE",
    "ENFORCE_READONLY_POLICY",
    "USE_AGENT_TARGET_ONLY",
    "INHERIT_SOURCE_RUN_EVIDENCE",
    "STOP_UNSAFE",
}


def validate_dataset() -> list[str]:
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("schema_version") != "aiops.dba-chat-evaluation.v1":
        errors.append("评测集Schema版本无效")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) < 12:
        return [*errors, "专业DBA评测场景不能少于12个"]
    identifiers = [case.get("case_id") for case in cases]
    if len(identifiers) != len(set(identifiers)):
        errors.append("评测场景ID重复")
    covered = {
        behavior
        for case in cases
        for behavior in case.get("expected_behaviors", ())
    }
    missing = sorted(REQUIRED_BEHAVIORS - covered)
    if missing:
        errors.append("评测能力覆盖不足：" + ", ".join(missing))
    for case in cases:
        if not case.get("question"):
            errors.append(f"{case.get('case_id')}缺少问题")
        if not case.get("expected_behaviors"):
            errors.append(f"{case.get('case_id')}缺少正向断言")
        if not case.get("forbidden_behaviors"):
            errors.append(f"{case.get('case_id')}缺少安全反向断言")
    return errors


def main() -> int:
    errors = validate_dataset()
    if errors:
        for error in errors:
            print(f"- {error}")
        return 1
    print("AI DBA聊天评测集校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
