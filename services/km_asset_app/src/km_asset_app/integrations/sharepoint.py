"""Microsoft Graph SharePoint 下载 Client。"""

import base64
from dataclasses import dataclass
import re
from urllib.parse import quote, unquote, urlparse

import aiohttp


class SharePointDownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SharePointFile:
    external_document_id: str
    name: str
    mime_type: str
    content: bytes


class SharePointClient:
    GRAPH = "https://graph.microsoft.com/v1.0"

    def __init__(self, *, tenant_id: str, client_id: str, client_secret: str, site_path: str, timeout_seconds: int = 120):
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._site_path = self._normalize_site_path(site_path)
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def _token(self, session: aiohttp.ClientSession) -> str:
        url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        data = {"client_id": self._client_id, "client_secret": self._client_secret, "scope": "https://graph.microsoft.com/.default", "grant_type": "client_credentials"}
        async with session.post(url, data=data) as response:
            body = await response.json(content_type=None)
            if response.status >= 400 or not body.get("access_token"):
                raise SharePointDownloadError("SharePoint 身份验证失败")
            return str(body["access_token"])

    @staticmethod
    def _share_id(source_url: str) -> str:
        value = base64.urlsafe_b64encode(unquote(source_url).encode("utf-8")).decode("ascii").rstrip("=")
        return f"u!{value}"

    @staticmethod
    def _normalize_site_path(site_path: str) -> str:
        """把站点相对路径或完整 URL 收敛为 Graph 接受的站点路径。"""
        decoded = unquote(site_path.strip())
        parsed = urlparse(decoded)
        path = parsed.path if parsed.hostname else decoded
        normalized = "/" + path.strip("/") if path.strip("/") else ""
        return normalized.rstrip("/")

    @classmethod
    def _site_path_candidates(
        cls, *, source_url: str, configured_site_path: str
    ) -> tuple[str, ...]:
        """优先使用配置，并从附件 URL 补充实际站点路径。"""
        decoded_path = urlparse(unquote(source_url.strip())).path
        match = re.search(
            r"/(sites|teams)/[^/]+",
            decoded_path,
            re.IGNORECASE,
        )
        derived = cls._normalize_site_path(match.group(0)) if match else ""
        return tuple(dict.fromkeys(
            value for value in (configured_site_path, derived) if value
        ))

    @staticmethod
    def _graph_error_code(payload: object) -> str:
        """提取不包含凭据的 Graph 错误码。"""
        if not isinstance(payload, dict):
            return "unknown"
        error = payload.get("error")
        if not isinstance(error, dict):
            return "unknown"
        return str(error.get("code") or "unknown")[:128]

    async def download(self, source_url: str) -> SharePointFile:
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                token = await self._token(session)
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                share_id = self._share_id(source_url)
                async with session.get(f"{self.GRAPH}/shares/{share_id}/driveItem", headers=headers) as metadata_response:
                    metadata = await metadata_response.json(content_type=None)
                if metadata_response.status < 400:
                    async with session.get(f"{self.GRAPH}/shares/{share_id}/driveItem/content", headers=headers) as content_response:
                        if content_response.status < 400:
                            content = await content_response.read()
                            return self._file(metadata=metadata, fallback_id=share_id, content=content)
                metadata, content = await self._download_by_path(
                    session=session,
                    headers=headers,
                    source_url=source_url,
                )
        except (aiohttp.ClientError, TimeoutError, ValueError) as exc:
            raise SharePointDownloadError("SharePoint 暂时不可用") from exc
        return self._file(metadata=metadata, fallback_id=self._share_id(source_url), content=content)

    async def _download_by_path(self, *, session: aiohttp.ClientSession, headers: dict[str, str], source_url: str):
        decoded = unquote(source_url.strip())
        parsed = urlparse(decoded)
        host = parsed.hostname
        site_paths = self._site_path_candidates(
            source_url=source_url,
            configured_site_path=self._site_path,
        )
        if not host or not site_paths:
            raise SharePointDownloadError(
                "SharePoint 分享链接解析失败，且无法确定站点路径"
            )
        match = re.search(
            r"/Shared Documents?/(.+)$",
            parsed.path,
            re.IGNORECASE,
        )
        if match is None:
            relative_path = ""
            for site_path in site_paths:
                site_marker = site_path.rstrip("/") + "/"
                index = parsed.path.lower().find(site_marker.lower())
                if index >= 0:
                    relative_path = parsed.path[
                        index + len(site_marker):
                    ]
                    break
            if relative_path.lower().startswith("shared documents/"):
                relative_path = relative_path[len("shared documents/"):]
        else:
            relative_path = match.group(1)
        relative_path = relative_path.strip("/")
        if not relative_path:
            raise SharePointDownloadError("SharePoint 文件路径无法解析")
        site = None
        resolution_errors = []
        for site_path in site_paths:
            async with session.get(
                f"{self.GRAPH}/sites/{host}:{site_path}",
                headers=headers,
            ) as site_response:
                candidate = await site_response.json(content_type=None)
                if (
                    site_response.status < 400
                    and isinstance(candidate, dict)
                    and candidate.get("id")
                ):
                    site = candidate
                    break
                resolution_errors.append(
                    f"path={site_path} status={site_response.status} "
                    f"code={self._graph_error_code(candidate)}"
                )
        if site is None:
            raise SharePointDownloadError(
                "SharePoint 站点解析失败：" + "; ".join(resolution_errors)
            )
        encoded = quote(relative_path, safe="/")
        item_url = f"{self.GRAPH}/sites/{site['id']}/drive/root:/{encoded}"
        async with session.get(item_url, headers=headers) as metadata_response:
            metadata = await metadata_response.json(content_type=None)
            if metadata_response.status >= 400:
                raise SharePointDownloadError(
                    "SharePoint 路径文件元数据读取失败："
                    f"status={metadata_response.status} "
                    f"code={self._graph_error_code(metadata)}"
                )
        async with session.get(f"{item_url}:/content", headers=headers) as content_response:
            if content_response.status >= 400:
                raise SharePointDownloadError(
                    "SharePoint 路径文件下载失败："
                    f"status={content_response.status}"
                )
            return metadata, await content_response.read()

    @staticmethod
    def _file(*, metadata: dict, fallback_id: str, content: bytes) -> SharePointFile:
        file_info = metadata.get("file") or {}
        return SharePointFile(external_document_id=str(metadata.get("id") or fallback_id), name=str(metadata.get("name") or "asset.bin"), mime_type=str(file_info.get("mimeType") or "application/octet-stream"), content=content)
