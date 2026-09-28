"""实施文档脚本 Artifact 的确定性构造与导出。"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass

from aiops_agent.contracts.implementation import (
    ImplementationRunbook,
    RunbookArtifactDescriptor,
)


@dataclass(frozen=True)
class GeneratedRunbookArtifact:
    artifact_id: str
    relative_path: str
    content: str
    media_type: str
    file_mode: str
    run_as: str
    target_path: str
    description: str


def attach_generated_artifacts(
    runbook: ImplementationRunbook,
    artifacts: tuple[GeneratedRunbookArtifact, ...],
) -> ImplementationRunbook:
    """计算摘要并把正文仅挂到进程内私有字段，Block 不重复保存正文。"""
    descriptors: list[RunbookArtifactDescriptor] = []
    payloads: dict[str, str] = {}
    for artifact in sorted(artifacts, key=lambda item: item.relative_path):
        content = artifact.content.rstrip() + "\n"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        descriptors.append(
            RunbookArtifactDescriptor(
                artifact_id=artifact.artifact_id,
                file_name=artifact.relative_path.rsplit("/", 1)[-1],
                relative_path=artifact.relative_path,
                media_type=artifact.media_type,
                sha256=digest,
                file_mode=artifact.file_mode,
                run_as=artifact.run_as,
                target_path=artifact.target_path,
                description=artifact.description,
            )
        )
        payloads[artifact.artifact_id] = content
    runbook.artifacts = tuple(descriptors)
    runbook._artifact_payloads = payloads
    return runbook


def generated_artifact_payloads(
    runbook: ImplementationRunbook,
) -> tuple[dict[str, str], ...]:
    """返回运行时物化所需的临时正文。"""
    descriptors = {item.artifact_id: item for item in runbook.artifacts}
    return tuple(
        {
            "artifact_id": artifact_id,
            "relative_path": descriptors[artifact_id].relative_path,
            "media_type": descriptors[artifact_id].media_type,
            "sha256": descriptors[artifact_id].sha256,
            "content": content,
        }
        for artifact_id, content in sorted(runbook._artifact_payloads.items())
    )


def render_runbook_artifact_zip(
    payload: dict,
    artifact_contents: dict[str, str],
) -> bytes:
    """按固定顺序和时间戳生成可复现脚本包。"""
    descriptors = list(payload.get("artifacts") or [])
    if not descriptors:
        raise ValueError("当前实施文档没有可下载脚本")
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        manifest = {
            "schema_version": payload.get("schema_version"),
            "profile": payload.get("profile"),
            "title": payload.get("title"),
            "artifacts": descriptors,
        }
        entries = [("00-manifest.json", json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n", "0644")]
        for descriptor in sorted(
            descriptors, key=lambda item: str(item.get("relative_path") or "")
        ):
            artifact_id = str(descriptor.get("artifact_id") or "")
            content = artifact_contents.get(artifact_id)
            if content is None:
                raise ValueError(f"脚本正文缺失：{artifact_id}")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if digest != descriptor.get("sha256"):
                raise ValueError(f"脚本摘要不一致：{artifact_id}")
            entries.append((
                str(descriptor["relative_path"]),
                content,
                str(descriptor.get("file_mode") or "0644"),
            ))
        for path, content, file_mode in entries:
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100000 | int(file_mode, 8)) << 16
            archive.writestr(info, content.encode("utf-8"))
    return buffer.getvalue()
