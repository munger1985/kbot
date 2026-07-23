"""Bundle 内部入库路由，不向调用方暴露 ORM 或数据库会话。"""
import json
import tempfile
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError
from starlette.datastructures import UploadFile

from platform_core.contracts import INTERNAL_API_V1
from knowledge_core.application.intake import IntakeCollectionError, IntakeConflictError
from knowledge_core.application.multipart import IntakeInProgressError, MultipartIntakeCommand
from knowledge_core.domain.intake import IntakeValidationError, KmAssetIntakeManifest


router = APIRouter(tags=["Knowledge Core"])


async def _copy_upload(upload: UploadFile, target: Path) -> None:
    with target.open("wb") as stream:
        while chunk := await upload.read(1024 * 1024):
            stream.write(chunk)


@router.post(
    f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}/collections/{{collection_key}}/ingestions/km-assets",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_km_asset(domain_id: int, collection_key: str, request: Request):
    """Receive one complete Asset bundle; acceptance never means parsing completed."""
    try:
        form = await request.form()
        bundle = json.loads(str(form["bundle"]))
        documents = json.loads(str(form["documents"]))
        failures = json.loads(str(form.get("document_failures", "[]")))
        manifest = KmAssetIntakeManifest.model_validate({
            "bundle": bundle, "documents": documents, "document_failures": failures,
        })
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_INTAKE_MANIFEST", "message": str(exc)}) from exc

    with tempfile.TemporaryDirectory(prefix="kc_http_") as directory:
        root = Path(directory)
        files: dict[str, Path] = {}
        try:
            for declaration in manifest.documents:
                if Path(declaration.part_name).name != declaration.part_name:
                    raise IntakeValidationError("file part name must not contain path separators")
                upload = form.get(declaration.part_name)
                if not isinstance(upload, UploadFile):
                    raise IntakeValidationError(f"file part is missing: {declaration.part_name}")
                target = root / declaration.part_name
                await _copy_upload(upload, target)
                files[declaration.part_name] = target
            orchestrator = request.app.state.kc_multipart_orchestrator
            accepted = await orchestrator.accept(MultipartIntakeCommand(
                domain_id=domain_id, collection_key=collection_key,
                actor_id=request.headers.get("X-KBot-Actor-Id", "svc:km-portal"),
                idempotency_key=request.headers.get("Idempotency-Key", ""),
                manifest=manifest, file_paths=files,
            ))
        except (IntakeValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail={"code": "INVALID_INTAKE_REQUEST", "message": str(exc)}) from exc
        except IntakeCollectionError as exc:
            raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
        except IntakeConflictError as exc:
            raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Intake conflicts with an immutable request or revision"}) from exc
        except IntakeInProgressError as exc:
            raise HTTPException(status_code=409, detail={"code": "INGESTION_IN_PROGRESS", "message": str(exc)}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "INGESTION_UNAVAILABLE", "message": type(exc).__name__}) from exc
    return {
        "bundle_id": accepted.bundle_id,
        "bundle_revision_id": accepted.bundle_revision_id,
        "source_revision": accepted.source_revision,
        "acceptance_status": accepted.acceptance_status,
        "status_url": (
            f"{INTERNAL_API_V1}/knowledge/domains/{domain_id}/bundles/{accepted.bundle_id}"
        ),
        "request_id": request.headers.get("X-Request-ID", str(uuid4())),
    }


class UserFileDeclaration(BaseModel):
    part_name: str = Field(min_length=1, max_length=128)
    client_file_id: str = Field(min_length=1, max_length=256)
    display_name: str = Field(min_length=1, max_length=512)
    declared_mime_type: str = Field(min_length=1, max_length=255)
    byte_size: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    ordinal: int = Field(default=0, ge=0)
    role: str = Field(default="CONTENT", pattern=r"^(CONTENT|SUPPLEMENT)$")
    required_flag: bool = False


class UserBundleDeclaration(BaseModel):
    client_bundle_id: str = Field(min_length=1, max_length=256)
    title: str = Field(min_length=1, max_length=512)
    source_revision: str | None = Field(default=None, max_length=256)
    security_level: int = Field(default=1, ge=0, le=999)
    facet: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


def _user_manifest(bundle: UserBundleDeclaration, files: list[UserFileDeclaration], *, source_revision: str):
    documents = []
    for item in files:
        documents.append({
            "external_document_id": item.client_file_id,
            "part_name": item.part_name,
            "role": item.role,
            "declared_name": item.display_name,
            "declared_mime_type": item.declared_mime_type,
            "ordinal": item.ordinal,
            "required_flag": item.required_flag,
            "byte_size": item.byte_size,
            "content_sha256": item.content_sha256,
        })
    return KmAssetIntakeManifest.model_validate({
        "bundle": {
            "source_id": bundle.client_bundle_id,
            "source_revision": source_revision,
            "title": bundle.title,
            "security_level": bundle.security_level,
            "facet": bundle.facet,
            "metadata": bundle.metadata,
        },
        "documents": documents,
        "document_failures": [],
    })


@router.post(
    f"{INTERNAL_API_V1}/knowledge/domains/{{domain_id}}/collections/{{collection_key}}/ingestions/user-files",
    status_code=status.HTTP_202_ACCEPTED,
)
async def ingest_user_files(domain_id: int, collection_key: str, request: Request):
    """Receive either independent files or one explicitly grouped Bundle."""
    try:
        form = await request.form()
        grouping_mode = str(form.get("grouping_mode", "EACH_FILE")).upper()
        if grouping_mode not in {"EACH_FILE", "SINGLE_BUNDLE"}:
            raise ValueError("grouping_mode must be EACH_FILE or SINGLE_BUNDLE")
        raw_files = json.loads(str(form["files"]))
        declarations = [UserFileDeclaration.model_validate(item) for item in raw_files]
        if not declarations:
            raise ValueError("files must not be empty")
        raw_bundle = form.get("bundle")
        bundle_data = UserBundleDeclaration.model_validate(json.loads(str(raw_bundle))) if raw_bundle else None
        if grouping_mode == "SINGLE_BUNDLE" and bundle_data is None:
            raise ValueError("bundle is required for SINGLE_BUNDLE")
        idempotency_key = request.headers.get("Idempotency-Key", "")
        if not idempotency_key:
            raise ValueError("Idempotency-Key is required")
    except (KeyError, TypeError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_USER_INTAKE", "message": str(exc)}) from exc

    with tempfile.TemporaryDirectory(prefix="kc_user_http_") as directory:
        root = Path(directory)
        file_paths: dict[str, Path] = {}
        try:
            for item in declarations:
                if Path(item.part_name).name != item.part_name:
                    raise IntakeValidationError("file part name must not contain path separators")
                upload = form.get(item.part_name)
                if not isinstance(upload, UploadFile):
                    raise IntakeValidationError(f"file part is missing: {item.part_name}")
                target = root / item.part_name
                await _copy_upload(upload, target)
                file_paths[item.part_name] = target
            actor = request.headers.get("X-KBot-Actor-Id", "user")
            orchestrator = request.app.state.kc_multipart_orchestrator
            items = []
            if grouping_mode == "SINGLE_BUNDLE":
                manifest = _user_manifest(
                    bundle_data, declarations,
                    source_revision=bundle_data.source_revision or idempotency_key,
                )
                accepted = await orchestrator.accept(MultipartIntakeCommand(
                    domain_id, collection_key, actor, idempotency_key, manifest, file_paths,
                    "kbot", "USER_UPLOAD", False, ("CONTENT", "SUPPLEMENT"),
                ))
                items.append({"status": "ACCEPTED", "bundle_id": accepted.bundle_id, "bundle_revision_id": accepted.bundle_revision_id})
            else:
                for item in declarations:
                    manifest = _user_manifest(
                        UserBundleDeclaration(client_bundle_id=item.client_file_id, title=item.display_name, security_level=1),
                        [item], source_revision=idempotency_key,
                    )
                    child_key = f"{idempotency_key}:{item.client_file_id}"
                    try:
                        accepted = await orchestrator.accept(MultipartIntakeCommand(
                            domain_id, collection_key, actor, child_key, manifest,
                            {item.part_name: file_paths[item.part_name]},
                            "kbot", "USER_UPLOAD", False, ("CONTENT", "SUPPLEMENT"),
                        ))
                        items.append({"status": "ACCEPTED", "client_file_id": item.client_file_id, "bundle_id": accepted.bundle_id, "bundle_revision_id": accepted.bundle_revision_id})
                    except IntakeCollectionError:
                        raise
                    except Exception as exc:
                        items.append({"status": "REJECTED", "client_file_id": item.client_file_id, "error": type(exc).__name__})
        except (IntakeValidationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail={"code": "INVALID_USER_INTAKE", "message": str(exc)}) from exc
        except IntakeCollectionError as exc:
            raise HTTPException(status_code=404, detail={"code": "COLLECTION_NOT_FOUND", "message": str(exc)}) from exc
        except IntakeConflictError as exc:
            raise HTTPException(status_code=409, detail={"code": str(exc), "message": "Intake conflicts with an immutable request or revision"}) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail={"code": "USER_INTAKE_UNAVAILABLE", "message": type(exc).__name__}) from exc
    return {"grouping_mode": grouping_mode, "items": items, "request_id": request.headers.get("X-Request-ID", str(uuid4()))}
