"""用户上传诊断材料的受控字面量检索 Handler。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .handlers import TaskExecutionContext


class AttachmentEvidenceMatch(BaseModel):
    """一段带行号的附件检索证据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=524288)
    match_terms: tuple[str, ...]


class AttachmentEvidenceSet(BaseModel):
    """由受控附件检索形成的不可变证据集。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["ATTACHMENT_EVIDENCE_SET.v1"] = (
        "ATTACHMENT_EVIDENCE_SET.v1"
    )
    upload_id: str
    file_name: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_terms: tuple[str, ...]
    matches: tuple[AttachmentEvidenceMatch, ...] = ()
    truncated: bool = False
    query_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")


class AttachmentSearchHandler:
    """以固定 rg 参数检索当前 Turn 已冻结的规范文本。"""

    _MAX_OUTPUT_BYTES = 524288
    _MAX_MATCHES_PER_TERM = 6

    def __init__(self, *, upload_store) -> None:
        self._upload_store = upload_store

    async def execute(
        self, context: TaskExecutionContext
    ) -> AttachmentEvidenceSet:
        action_id = context.task_key.removeprefix("attachment:").split(":", 1)[0]
        searches = context.plan_snapshot.get("attachment_search", ())
        search = next(
            (
                item
                for item in searches
                if str(item.get("action_id")) == action_id
            ),
            None,
        )
        if not isinstance(search, dict):
            raise ValueError("附件检索冻结快照不存在")
        terms = tuple(str(item) for item in search["terms"])
        context_lines = int(search["context_lines"])
        max_result_bytes = min(
            int(search.get("max_result_bytes", self._MAX_OUTPUT_BYTES)),
            self._MAX_OUTPUT_BYTES,
        )
        self._upload_store.read_artifact(
            payload_uri=str(search["payload_uri"]),
            content_hash=str(search["content_hash"]),
            byte_size=int(search["byte_size"]),
        )
        # 通过受控 URI 在 store 内定位文件，模型输入绝不参与命令或路径构造。
        payload_path = self._upload_store.artifact_path(str(search["payload_uri"]))
        lines: dict[int, str] = {}
        line_terms: dict[int, set[str]] = defaultdict(set)
        truncated = False
        for term in terms:
            events, output_truncated = await self._rg(
                path=payload_path,
                term=term,
                context_lines=context_lines,
                max_output_bytes=max_result_bytes,
            )
            truncated = truncated or output_truncated
            for line_number, line_text, is_match in events:
                if not line_number or not line_text:
                    continue
                lines[line_number] = line_text
                if is_match:
                    line_terms[line_number].add(term)
        matches = []
        consumed = 0
        groups: list[list[int]] = []
        for line_number in sorted(lines):
            if not groups or line_number > groups[-1][-1] + 1:
                groups.append([line_number])
            else:
                groups[-1].append(line_number)
        for group in groups:
            rendered = "\n".join(
                f"{number}: {lines[number]}" for number in group
            )
            encoded = rendered.encode("utf-8")
            if consumed + len(encoded) > max_result_bytes:
                truncated = True
                break
            consumed += len(encoded)
            matches.append(
                AttachmentEvidenceMatch(
                    line_start=group[0],
                    line_end=group[-1],
                    text=rendered,
                    match_terms=tuple(
                        sorted(
                            {
                                term
                                for line_number in group
                                for term in line_terms[line_number]
                            }
                        )
                    ),
                )
            )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "upload_id": search["upload_id"],
                    "content_hash": search["content_hash"],
                    "terms": terms,
                    "context_lines": context_lines,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return AttachmentEvidenceSet(
            upload_id=str(search["upload_id"]),
            file_name=str(search["file_name"]),
            content_hash=str(search["content_hash"]),
            query_terms=terms,
            matches=tuple(matches),
            truncated=truncated,
            query_fingerprint=fingerprint,
        )

    async def _rg(self, *, path, term: str, context_lines: int, max_output_bytes: int):
        process = await asyncio.create_subprocess_exec(
            "rg",
            "--json",
            "--fixed-strings",
            "--line-number",
            "--no-heading",
            "--color=never",
            f"--context={context_lines}",
            f"--max-count={self._MAX_MATCHES_PER_TERM}",
            "--max-columns=4096",
            "--",
            term,
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=12)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise ValueError("附件检索超时") from None
        truncated = len(stdout) > max_output_bytes
        if truncated:
            stdout = stdout[:max_output_bytes]
        events = []
        for raw_line in stdout.splitlines():
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                truncated = True
                continue
            data = event.get("data") or {}
            if event.get("type") not in {"match", "context"}:
                continue
            line_number = data.get("line_number")
            line = (data.get("lines") or {}).get("text")
            if not isinstance(line_number, int) or not isinstance(line, str):
                continue
            events.append((line_number, line.rstrip("\r\n"), event["type"] == "match"))
        return events, truncated
