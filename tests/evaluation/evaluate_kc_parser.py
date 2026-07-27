"""使用带标注的黄金语料清单评测 KC 解析器。"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from knowledge_core.parsing.converter import KcDoclingConverter
from knowledge_core.parsing.pipeline import KcParsingPipeline


async def evaluate(args) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    converter = KcDoclingConverter(artifacts_path=str(args.artifacts_path))
    pipeline = KcParsingPipeline(parser_version=args.parser_version)
    results = []
    for index, case in enumerate(manifest["cases"], start=1):
        document = await converter.convert(
            source_path=Path(case["path"]), do_ocr=bool(case.get("do_ocr", True)),
        )
        output = pipeline.parse(
            document_version_id=index, parse_view_id=index, document=document,
        )
        headings = [
            node["heading"]["text"] for node in output.artifacts["structure_ir"]["nodes"]
            if node.get("heading")
        ]
        content = "\n".join(evidence.content_text for evidence in output.evidences)
        locator_schemas = {evidence.locator_schema_version for evidence in output.evidences}
        failures = []
        for expected in case.get("expected_headings", []):
            if expected not in headings:
                failures.append(f"missing heading: {expected}")
        for expected in case.get("expected_content", []):
            if expected not in content:
                failures.append(f"missing content: {expected}")
        for expected in case.get("expected_locator_schemas", []):
            if expected not in locator_schemas:
                failures.append(f"missing locator schema: {expected}")
        if not output.quality_report.passed:
            failures.extend(output.quality_report.hard_failures)
        results.append({
            "case_id": case["case_id"], "passed": not failures,
            "failures": failures, "quality": output.quality_report.as_dict(),
            "evidence_count": len(output.evidences),
        })
    return {
        "passed": all(result["passed"] for result in results),
        "case_count": len(results), "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--artifacts-path", type=Path, required=True)
    parser.add_argument("--parser-version", default="benchmark")
    args = parser.parse_args()
    report = asyncio.run(evaluate(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
