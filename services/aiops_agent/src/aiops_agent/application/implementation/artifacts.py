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


def _markdown_text(value: object) -> str:
    """把制品元数据转换为可放入 Markdown 列表的单行文本。"""
    return str(value or "").replace("`", "\\`").replace("\n", " ").strip()


def _package_layout(descriptors: list[dict]) -> tuple[str, str]:
    """从相对路径和目标路径解析包目录与服务器暂存目录。"""
    package_roots: set[str] = set()
    target_roots: set[str] = set()
    for descriptor in descriptors:
        relative_path = str(descriptor.get("relative_path") or "").strip("/")
        if not relative_path:
            continue
        parts = relative_path.split("/", 1)
        package_roots.add(parts[0])
        target_path = str(descriptor.get("target_path") or "").rstrip("/")
        artifact_path = parts[1] if len(parts) == 2 else parts[0]
        suffix = "/" + artifact_path
        if target_path.endswith(suffix):
            target_roots.add(target_path[: -len(suffix)] or "/")
    package_root = next(iter(package_roots)) if len(package_roots) == 1 else ""
    target_root = next(iter(target_roots)) if len(target_roots) == 1 else ""
    return package_root, target_root


def _profile_usage(
    profile: str,
    *,
    package_root: str,
    target_root: str,
) -> str:
    """生成不同实施类型的最小、安全使用入口。"""
    if profile == "ORACLE_DATAPUMP_MIGRATION":
        stage_root = (
            target_root
            or "/var/tmp/kbot-runbooks/oracle-datapump-migration"
        )
        return f"""## Data Pump 包使用顺序

1. 将本 ZIP 分别部署到源数据库主机和目标数据库主机的 `{stage_root}`。
2. 按主文档参数附录核对导出模式、Schema 范围、Directory 路径和目标端路径审阅标记。
3. 先执行 `sql/assess-source.sql` 和 `sql/review-mapping.sql`，确认字符集、时区、对象及表空间差异。
4. 在源端和目标端分别创建文件系统目录，再执行 `sql/create-directory.sql`。
5. 源端使用 `par/expdp.par` 导出并生成 SHA-256 清单；通过批准通道传输转储后在目标端校验摘要。
6. 目标端先使用 `par/impdp-preview.par` 生成 SQLFILE，审批通过后再使用 `par/impdp.par` 正式导入。
7. 在源端和目标端分别执行 `sql/validate-objects.sql`，对比对象计数、无效对象和统计信息。

目标端命令使用本机 OS 认证，不需要在文件中保存 TNS、用户名或密码。
"""
    if profile == "ORACLE_DATABASE_MIGRATION":
        stage_root = (
            target_root
            or "/var/tmp/kbot-runbooks/oracle-database-migration"
        )
        return f"""## 数据库迁移包使用顺序

1. 将本 ZIP 分别部署到源数据库主机和目标数据库主机的 `{stage_root}`。
2. 阅读参数附录，确认默认方案为新目标主机、同平台同版本同拓扑同文件布局的离线 RMAN 迁移。
3. 在源端执行 `sql/assess-source.sql`，按结果准备目标 Oracle 软件、文件系统或 ASM Disk Group。
4. 在正式窗口前执行备份恢复链校验、全库逻辑读校验和目标暂存目录容量检查。
5. 获批停写后，将源库启动到 MOUNT，执行 `rman/backup-source.rman` 并生成 SHA-256 清单。
6. 通过批准通道传输备份、PFILE 和摘要清单；目标端校验后执行 `sql/start-target-nomount.sql`。
7. 在目标端执行 `rman/duplicate-target.rman`，完成后运行 `sql/validate-database.sql` 和业务验收。
8. 切换成功后执行 `rman/backup-post-cutover.rman`，观察期结束前保留关闭状态的源环境。

源端和目标端都使用本机 OS 认证，不需要把 TNS、用户名或密码写入脚本。
"""
    if profile == "ORACLE_RU_PATCH":
        stage_root = (
            target_root
            or "/var/tmp/kbot-runbooks/oracle-ru-patch"
        )
        return f"""## RU 补丁包使用顺序

1. 将本 ZIP 部署到数据库主机的 `{stage_root}`，先执行主文档中的数据库身份、拓扑和现有补丁核对。
2. 执行介质目录创建命令。将唯一已审批 RU ZIP 放入参数附录所列补丁暂存目录的 `media/ru/`；可选 OJVM ZIP 放入 `media/ojvm/`。
3. 将批准来源提供的官方 SHA-256 记录写入暂存目录的 `SHA256SUMS`，清单文件名使用相对暂存目录的路径。不要用下载后自行计算的摘要代替官方摘要。
4. 以 `root` 执行 `{stage_root}/bin/prepare-media.sh`，完成摘要校验、解压和唯一顶层补丁包识别。
5. 依次执行 `{stage_root}/bin/inventory-and-media-check.sh` 和 `{stage_root}/bin/analyze-patch.sh`；任何一项失败都必须停止。
6. 完成保护备份并取得维护窗口批准后，执行 `{stage_root}/bin/apply-patch.sh`。
7. 以 `oracle` 执行 `{stage_root}/bin/run-datapatch.sh`，随后执行二进制和数据库验证。
8. 回退决定获批后才可执行 `{stage_root}/bin/rollback-patch.sh`；二进制回退完成后还必须执行 `run-datapatch.sh` 和 `verify-patch.sh`。正常成功流程不得运行回退脚本。

脚本运行时按 `/etc/oratab`、PMON 和 `oraenv` 自动定位数据库 Home；GI 或 Oracle Restart 环境按 `/etc/oracle/olr.loc` 定位 GI Home。暂存目录中出现多套 RU/OJVM、摘要不一致、Inventory 异常或 Analyze 失败时，脚本会非零退出。
"""
    if profile != "ORACLE_RMAN_BACKUP_BUILD":
        return (
            "## 本方案的执行方式\n\n"
            "本包中的文件是实施文档命令所引用的配套制品。请严格按照配套 "
            "Markdown 或 PDF 文档的阶段顺序执行，不要脱离主文档单独批量运行。"
            "涉及停库、恢复、补丁、集群或角色切换的命令，必须在对应审批和停止条件"
            "满足后再执行。\n"
        )

    source_root = package_root or "oracle-rman-backup"
    stage_root = target_root or "/var/tmp/kbot-runbooks/oracle-rman-backup"
    return f"""## RMAN 备份包使用顺序

1. 解压后先核对 `{source_root}/env.conf` 中的实例、备份目录、日志目录、数据库角色和监控目录。
2. 按 `00-manifest.json` 核对文件摘要、权限、执行用户和目标路径。
3. 由管理员把 `{source_root}` 目录完整部署到 `{stage_root}`，并按清单设置属主和权限。
4. 先以 `oracle` 用户应用 RMAN 配置：

```bash
rman target / cmdfile={stage_root}/rman/configure.rman
```

5. 首次手工执行 Level 0 备份并检查结果：

```bash
{stage_root}/bin/run-rman-job.sh level0
{stage_root}/bin/check-last-backup.sh
```

6. 确认日志、状态文件和恢复校验均正常后，再由 `root` 安装 Systemd 调度：

```bash
{stage_root}/bin/install-schedule.sh
systemctl list-timers 'oracle-rman-*'
```

支持的手工任务为 `level0`、`level1`、`archivelog`、`controlfile`、`cleanup`、`validate` 和 `restore-validate`。监控目录存在时，还需在数据库主机安装指标采集文件，并在 Prometheus 主机检查和加载告警规则。
"""


def render_runbook_zip_readme(payload: dict) -> str:
    """为可执行 ZIP 生成随包交付的中文使用说明。"""
    descriptors = list(payload.get("artifacts") or [])
    profile = str(payload.get("profile") or "DATABASE_IMPLEMENTATION")
    title = str(payload.get("title") or "数据库实施操作文档")
    package_root, target_root = _package_layout(descriptors)
    file_lines = []
    for descriptor in sorted(
        descriptors, key=lambda item: str(item.get("relative_path") or "")
    ):
        path = _markdown_text(descriptor.get("relative_path"))
        description = (
            _markdown_text(descriptor.get("description")) or "配套实施文件"
        )
        run_as = _markdown_text(descriptor.get("run_as")) or "按主文档确认"
        mode = _markdown_text(descriptor.get("file_mode")) or "按清单确认"
        file_lines.append(
            f"- `{path}` — {description}（执行用户：`{run_as}`；权限：`{mode}`）"
        )
    files = "\n".join(file_lines)
    return f"""# {title}：ZIP 可执行包说明

本 README 随 ZIP 一起交付，用于说明包内文件和安全使用顺序。ZIP 是实施文档的配套可执行材料，不是可以跳过审核后一键运行的安装包。

## 下载格式的定位

- Markdown：便于阅读、评审和修改的实施文档。
- PDF：用于审批、签字和归档的固定版实施文档。
- ZIP：实际部署到目标服务器的脚本、配置、调度和监控文件。

结构化 Runbook JSON 仅供程序内部处理，不作为用户下载文件提供。

## 包内基础文件

- `README.md`：当前使用说明。
- `00-manifest.json`：Profile、文件清单、SHA-256、目标路径、权限和执行用户。
- `{package_root or '方案制品目录'}/`：本次 Runbook 生成的实际实施文件。

## 本次制品清单

Profile：`{profile}`

{files}

## 使用前检查

1. 先阅读配套 Markdown 或 PDF，确认实施范围、阶段顺序、停止条件、验证和回退要求。
2. 核对 `00-manifest.json` 中每个文件的 SHA-256；摘要不一致时不要执行。
3. 核对所有目标路径、挂载容量、数据库角色、Oracle 环境、执行用户和文件权限。
4. 先手工执行低风险检查和首次任务，验证日志及退出码后再启用自动调度。
5. 不要把测试环境生成的包直接用于其他数据库；实际事实变化后应重新生成 Runbook 和 ZIP。

{_profile_usage(profile, package_root=package_root, target_root=target_root)}
## 安全边界

涉及停库、归档模式切换、数据恢复、清理、补丁、集群资源或主备角色切换的步骤，必须遵守主文档中的风险等级、审批窗口和停止条件。执行结果应连同日志、状态文件和验证证据一起留存。
"""


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
    """按固定顺序和时间戳生成含 README 的可复现脚本包。"""
    descriptors = list(payload.get("artifacts") or [])
    if not descriptors:
        raise ValueError("当前实施文档没有可下载脚本")
    readme = render_runbook_zip_readme(payload).rstrip() + "\n"
    readme_digest = hashlib.sha256(readme.encode("utf-8")).hexdigest()
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        manifest = {
            "schema_version": payload.get("schema_version"),
            "profile": payload.get("profile"),
            "title": payload.get("title"),
            "package_files": [{
                "relative_path": "README.md",
                "media_type": "text/markdown",
                "sha256": readme_digest,
                "file_mode": "0644",
                "description": "ZIP 可执行包内容、安全检查和使用顺序说明。",
            }],
            "artifacts": descriptors,
        }
        entries = [("00-manifest.json", json.dumps(
            manifest, ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n", "0644"), ("README.md", readme, "0644")]
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
