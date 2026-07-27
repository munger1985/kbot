"""Deterministic Bundle/Document profile construction for Discovery."""
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from typing import Any

from knowledge_core.entities import KcDiscoveryObjectEntity, KcIngestionJobEntity
from knowledge_core.domain.parse_tasks import ParseLeaseError, verify_lease


PROFILE_SCHEMA_VERSION = "profile/v1"


@dataclass(frozen=True)
class MemberProfileInput:
    external_document_id: str
    declared_name: str | None
    document_role: str
    mime_type: str | None
    member_status: str
    evidence_count: int
    section_titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class BundleProfileInput:
    title: str
    source_system: str
    source_type: str
    source_id: str
    canonical_url: str | None
    facet: dict[str, Any] | None
    members: tuple[MemberProfileInput, ...]
    missing_members: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveryProfile:
    profile_key: str
    display_title: str
    profile_text: str
    facet: dict[str, Any] | None
    coverage: dict[str, Any]
    profile_hash: str
    profile_schema_version: str = PROFILE_SCHEMA_VERSION


def build_bundle_profile(value: BundleProfileInput) -> DiscoveryProfile:
    """Create stable text; never use an LLM or a vector provider here."""
    members = sorted(value.members, key=lambda item: item.external_document_id)
    lines = [
        f"标题: {value.title.strip()}",
        f"来源系统: {value.source_system}",
        f"来源类型: {value.source_type}",
        f"来源标识: {value.source_id}",
    ]
    if value.canonical_url:
        lines.append(f"规范地址: {value.canonical_url}")
    if value.facet:
        lines.append("属性: " + _stable_json(value.facet))
    lines.append("文件成员:")
    for member in members:
        name = member.declared_name or member.external_document_id
        mime = member.mime_type or "unknown"
        lines.append(
            f"- {name} | id={member.external_document_id} | role={member.document_role} "
            f"| mime={mime} | status={member.member_status} | evidence={member.evidence_count}"
        )
        if member.section_titles:
            lines.append("  章节: " + " / ".join(member.section_titles[:20]))
    if value.missing_members:
        lines.append("缺失成员: " + "、".join(sorted(value.missing_members)))
    text = "\n".join(line.strip() for line in lines if line.strip())
    coverage = {
        "member_count": len(members),
        "ready_member_count": sum(item.member_status == "READY" for item in members),
        "evidence_count": sum(item.evidence_count for item in members),
        "missing_members": list(sorted(value.missing_members)),
    }
    identity = {
        "profile_key": "bundle",
        "title": value.title,
        "source_system": value.source_system,
        "source_type": value.source_type,
        "source_id": value.source_id,
        "canonical_url": value.canonical_url,
        "facet": value.facet or {},
        "members": [member.__dict__ for member in members],
        "missing_members": sorted(value.missing_members),
        "profile_schema_version": PROFILE_SCHEMA_VERSION,
    }
    return DiscoveryProfile(
        profile_key="bundle", display_title=value.title.strip(), profile_text=text,
        facet=value.facet, coverage=coverage,
        profile_hash=sha256(_stable_json(identity).encode("utf-8")).hexdigest(),
    )


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


class KnowledgeCoreProfileService:
    """Materialize deterministic profiles after all Member INDEX jobs settle."""

    def __init__(self, *, uow_factory):
        self._uow_factory = uow_factory

    async def run_job(self, *, job_id: UUID, worker_id: str, input_fingerprint: str) -> int:
        async with self._uow_factory() as uow:
            if not all((uow.jobs, uow.revisions, uow.bundles, uow.members, uow.evidence, uow.discovery, uow.session)):
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "PROFILE" or job.bundle_revision_id is None:
                raise ValueError("invalid PROFILE job")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint)
            revision = await uow.revisions.get_by_id(bundle_revision_id=job.bundle_revision_id, lock=True)
            if revision is None:
                raise ValueError("PROFILE revision not found")
            bundle = await uow.bundles.get_by_id(bundle_id=revision.bundle_id)
            if bundle is None:
                raise ValueError("PROFILE bundle not found")
            members = await uow.members.list_by_revision(bundle_revision_id=revision.bundle_revision_id)
            manifest = next((item for item in members if item.document_role == "MANIFEST"), None)
            if manifest is not None and manifest.member_status != "READY":
                raise ValueError("PROFILE requires a ready manifest when one is declared")
            profile_members: list[MemberProfileInput] = []
            missing: list[str] = []
            for member in members:
                if member.member_status != "READY" or member.document_version_id is None:
                    missing.append(member.declared_name or member.external_document_id)
                    continue
                profile_members.append(MemberProfileInput(
                    external_document_id=member.external_document_id,
                    declared_name=member.declared_name,
                    document_role=member.document_role,
                    mime_type=member.declared_mime_type,
                    member_status=member.member_status,
                    evidence_count=await uow.evidence.count_active_for_version(
                        document_version_id=member.document_version_id,
                    ),
                    section_titles=tuple(await uow.evidence.list_section_titles(
                        document_version_id=member.document_version_id,
                    )),
                ))
            profile = build_bundle_profile(BundleProfileInput(
                title=revision.title, source_system=bundle.source_system,
                source_type=bundle.source_type, source_id=bundle.source_id,
                canonical_url=revision.canonical_url, facet=revision.facet_json,
                members=tuple(profile_members), missing_members=tuple(missing),
            ))
            existing = await uow.discovery.get_by_key(
                bundle_revision_id=revision.bundle_revision_id, profile_key="bundle", lock=True,
            )
            if existing is None:
                await uow.discovery.add(KcDiscoveryObjectEntity(
                    collection_id=revision.collection_id, bundle_id=revision.bundle_id,
                    bundle_revision_id=revision.bundle_revision_id, object_type="BUNDLE",
                    profile_key=profile.profile_key, display_title=profile.display_title,
                    profile_text=profile.profile_text, facet_json=profile.facet,
                    coverage_json=profile.coverage, profile_hash=profile.profile_hash,
                    profile_schema_version=profile.profile_schema_version,
                    security_level=revision.security_level, discovery_status="STAGED",
                    created_by=worker_id, updated_by=worker_id,
                ))
            else:
                existing.profile_text = profile.profile_text
                existing.facet_json = profile.facet
                existing.coverage_json = profile.coverage
                existing.profile_hash = profile.profile_hash
                existing.discovery_status = "STAGED"
                existing.updated_by = worker_id
            for member in profile_members:
                document = next((item for item in members if item.external_document_id == member.external_document_id), None)
                if document is None or document.document_version_id is None:
                    continue
                document_profile = _build_document_profile(revision, member, profile)
                existing_document = await uow.discovery.get_by_key(
                    bundle_revision_id=revision.bundle_revision_id,
                    profile_key=document_profile.profile_key, lock=True,
                )
                values = dict(
                    collection_id=revision.collection_id, bundle_id=revision.bundle_id,
                    bundle_revision_id=revision.bundle_revision_id,
                    bundle_revision_document_id=document.bundle_revision_document_id,
                    document_id=document.document_id, document_version_id=document.document_version_id,
                    object_type="DOCUMENT", profile_key=document_profile.profile_key,
                    display_title=document_profile.display_title, profile_text=document_profile.profile_text,
                    facet_json=document_profile.facet, coverage_json=document_profile.coverage,
                    profile_hash=document_profile.profile_hash,
                    profile_schema_version=document_profile.profile_schema_version,
                    security_level=revision.security_level, discovery_status="STAGED",
                    created_by=worker_id, updated_by=worker_id,
                )
                if existing_document is None:
                    await uow.discovery.add(KcDiscoveryObjectEntity(**values))
                else:
                    for key, value in values.items():
                        if key not in {"created_by"}:
                            setattr(existing_document, key, value)
            job.job_status, job.completed_at = "SUCCEEDED", datetime.now(timezone.utc)
            job.result_json = {"profile_count": 1 + len(profile_members), "status": "STAGED"}
            job.lease_owner = job.lease_until = None
            profile_fingerprint = sha256(
                f"{revision.bundle_revision_id}:{profile.profile_hash}".encode("utf-8")
            ).hexdigest()
            profile_index_key = f"INDEX:DISCOVERY:{revision.bundle_revision_id}:{profile_fingerprint}"
            existing_index = await uow.jobs.get_by_idempotency_key(
                collection_id=revision.collection_id,
                idempotency_key=profile_index_key,
                input_fingerprint=profile_fingerprint,
            )
            if existing_index is None:
                await uow.jobs.add(KcIngestionJobEntity(
                    collection_id=revision.collection_id,
                    bundle_revision_id=revision.bundle_revision_id,
                    job_type="INDEX", idempotency_key=profile_index_key,
                    input_fingerprint=profile_fingerprint,
                    payload_json={"target": "DISCOVERY", "profile_schema_version": PROFILE_SCHEMA_VERSION},
                    job_status="PENDING", priority=job.priority,
                    max_attempts=job.max_attempts, created_by=worker_id, updated_by=worker_id,
                ))
            await uow.session.flush()
            await uow.commit()
            return 1 + len(profile_members)

    async def heartbeat(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
        lease_seconds: int = 120,
    ) -> datetime:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "PROFILE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            job.heartbeat_at = now
            job.lease_until = now + timedelta(seconds=lease_seconds)
            job.row_version += 1
            await uow.session.flush()
            await uow.commit()
            return job.lease_until

    async def fail(
        self, *, job_id: UUID, worker_id: str, input_fingerprint: str,
        failure_class: str, failure_code: str, failure_message: str | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            if uow.jobs is None or uow.session is None:
                raise RuntimeError("Knowledge Core Unit of Work is not initialized")
            job = await uow.jobs.get_by_id(ingestion_job_id=job_id, lock=True)
            if job is None or job.job_type != "PROFILE":
                raise ParseLeaseError("JOB_LEASE_INVALID")
            verify_lease(job, worker_id=worker_id, input_fingerprint=input_fingerprint, now=now)
            retryable = failure_class == "TRANSIENT" and job.attempt_count < job.max_attempts
            job.failure_class, job.failure_code, job.failure_message = failure_class, failure_code, failure_message
            job.lease_owner = job.lease_until = None
            if retryable:
                job.job_status = "RETRY_WAIT"
                job.available_at = now + timedelta(seconds=min(300, 2 ** job.attempt_count * 5))
            else:
                job.job_status, job.completed_at = "FAILED", now
                if job.bundle_revision_id is not None and uow.revisions is not None:
                    revision = await uow.revisions.get_by_id(bundle_revision_id=job.bundle_revision_id, lock=True)
                    if revision is not None:
                        revision.failure_code, revision.failure_message = failure_code, failure_message
                        revision.status = "FAILED"
                        revision.completed_at = now
            await uow.session.flush()
            await uow.commit()
            return job.job_status


def _build_document_profile(revision, member: MemberProfileInput, bundle_profile: DiscoveryProfile) -> DiscoveryProfile:
    text = "\n".join([
        f"Bundle: {bundle_profile.display_title}",
        f"文件: {member.declared_name or member.external_document_id}",
        f"文件标识: {member.external_document_id}",
        f"角色: {member.document_role}",
        f"类型: {member.mime_type or 'unknown'}",
        f"Evidence 数量: {member.evidence_count}",
        *( ["章节: " + " / ".join(member.section_titles)] if member.section_titles else []),
    ])
    identity = {
        "profile_key": f"member:{member.external_document_id}",
        "revision": str(revision.bundle_revision_id), "member": member.__dict__,
        "schema": PROFILE_SCHEMA_VERSION,
    }
    return DiscoveryProfile(
        profile_key=f"member:{member.external_document_id}",
        display_title=member.declared_name or member.external_document_id,
        profile_text=text, facet=None,
        coverage={"evidence_count": member.evidence_count, "member_status": member.member_status},
        profile_hash=sha256(_stable_json(identity).encode("utf-8")).hexdigest(),
    )
