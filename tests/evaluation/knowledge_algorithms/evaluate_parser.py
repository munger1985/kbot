"""按显式 Profile 和黄金语料评测新解析事件流；仅在手工调用时处理指定文件。"""

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [
    str(path) for path in ROOT.glob("services/*/src")]
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.contracts import ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseEventType, ParseRunIdentity, ParseStreamValidator
from knowledge_core.adapters.bounded_parse_executor import BoundedParseExecutor
from knowledge_core.domain.parse_jobs import ComputeAccess, ParseJob


async def evaluate(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = decode(ParseProfile, manifest["profile"])
    owner = manifest["owner_app_id"]
    executor = BoundedParseExecutor(**manifest["executor"])
    results = []
    for case in manifest["cases"]:
        path = (manifest_path.parent / case["path"]).resolve()
        failures, text, headings, kinds, count = [], [], [], set(), 0
        try:
            size = path.stat().st_size
            if not 0 < size <= profile.max_input_bytes:
                raise ValueError("输入超出显式 Profile 预算")
            digest = sha256()
            with path.open("rb") as source_file:
                while block := source_file.read(65536):
                    digest.update(block)
            source = SourceDescriptor(owner, str(uuid4()), digest.hexdigest(),
                case["mime_type"], size, "evaluation-source")
            run = ParseRunIdentity(owner, str(uuid4()), str(uuid4()), 1, 1, source.sha256, profile.profile_hash)
            job = ParseJob(ComputeAccess(owner, "evaluation", str(uuid4())), run, source, profile,
                           int(time.time()) + int(profile.timeout_seconds) + 60)
            class Reader:
                descriptor = source
                async def chunks(self):
                    with path.open("rb") as file:
                        while block := file.read(65536):
                            yield block
            validator = ParseStreamValidator(run, max_events=profile.max_objects * 4 + 100,
                max_objects=profile.max_objects, max_payload_bytes=profile.max_event_bytes)
            with tempfile.TemporaryDirectory(prefix="knowledge-parser-evaluation-") as directory:
                async for event in executor.events(job, Reader(), Path(directory)):
                    validator.accept(event)
                    if event.event_type == ParseEventType.NODE_BATCH:
                        for node in json.loads(event.payload_json)["objects"]:
                            count += 1
                            text.append(node["text"])
                            kinds.add(node["kind"])
                            if node.get("heading_level"):
                                headings.append(node["text"])
                    if event.event_type == ParseEventType.RUN_FAILED:
                        failures.append("PARSE_FAILED")
            if not validator.completed:
                failures.append("STREAM_NOT_COMPLETED")
            content = "\n".join(text)
            for expected in case.get("expected_content", []):
                if expected not in content:
                    failures.append("MISSING_CONTENT:" + expected)
            for expected in case.get("expected_headings", []):
                if expected not in headings:
                    failures.append("MISSING_HEADING:" + expected)
            for expected in case.get("expected_kinds", []):
                if expected not in kinds:
                    failures.append("MISSING_KIND:" + expected)
        except Exception as exc:
            failures.append(type(exc).__name__)
        results.append({"case_id": case["case_id"], "passed": not failures,
                        "failures": failures, "node_count": count, "kinds": sorted(kinds)})
    return {"owner_app_id": owner, "profile_hash": profile.profile_hash,
        "passed": bool(results) and all(row["passed"] for row in results),
        "case_count": len(results), "results": results,
        "retrieval_quality_proven": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="包含 owner_app_id、完整 profile、executor 与 cases")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.manifest))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
