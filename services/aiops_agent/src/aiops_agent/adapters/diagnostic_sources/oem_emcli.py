"""原厂 EMCLI 的有界只读执行器；不接受任意命令、路径或 SQL。"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import ssl
import tempfile
import time
from urllib.parse import urlparse

from aiops_agent.application.exadata_catalog import load_catalog
from .base import DiagnosticSourceAdapterError


def fail(code):
    raise DiagnosticSourceAdapterError(code, "原厂 EMCLI 只读通道不可执行或响应未通过核验")


@dataclass
class NativeSession:
    directory: tempfile.TemporaryDirectory
    expires_at: float
    initialized: bool = False
    client_home: Path | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class OEMEMCLIClient:
    def __init__(self, *, executable, session_root, java_home=None, timeout_seconds=20,
                 session_ttl_seconds=300, max_sessions=4):
        self.executable = executable
        self.session_root = Path(session_root)
        self.java_home = java_home
        self.timeout_seconds = timeout_seconds
        self.session_ttl_seconds = session_ttl_seconds
        self.max_sessions = max_sessions
        self._sessions = OrderedDict()
        self._pepper = secrets.token_bytes(32)
        self._processes = set()
        self._closed = False

    def _session(self, context):
        if self._closed:
            fail("SOURCE_EMCLI_UNAVAILABLE")
        username = context.credentials.get("username")
        password = context.credentials.get("password")
        if (not isinstance(username, str) or not isinstance(password, str) or not username or not password
                or len(username) > 256 or len(password) > 16384
                or any(char in username + password for char in "\r\n\0")):
            fail("SOURCE_AUTH_FAILED")
        if not self.executable or not Path(self.executable).is_absolute() or not os.access(self.executable, os.X_OK):
            fail("SOURCE_EMCLI_UNAVAILABLE")
        now = time.monotonic()
        for key, entry in list(self._sessions.items()):
            if entry.expires_at <= now and not entry.lock.locked():
                entry.directory.cleanup()
                del self._sessions[key]
        identity = json.dumps([str(context.source_id), context.config_version, context.endpoint,
            context.config.get("exadata_emcli_certificate_sha256"),
            context.config.get("exadata_emcli_timezone"), username, password])
        key = hmac.digest(self._pepper, identity.encode(), "sha256")
        if key in self._sessions:
            self._sessions.move_to_end(key)
            return self._sessions[key]
        if len(self._sessions) >= self.max_sessions:
            victim = next((key for key, entry in self._sessions.items() if not entry.lock.locked()), None)
            if victim is None:
                fail("SOURCE_BUSY")
            self._sessions.pop(victim).directory.cleanup()
        if self.session_root.is_symlink():
            fail("SOURCE_EMCLI_UNAVAILABLE")
        self.session_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        status = self.session_root.stat()
        if status.st_uid != os.geteuid() or status.st_mode & 0o077:
            fail("SOURCE_EMCLI_UNAVAILABLE")
        entry = NativeSession(tempfile.TemporaryDirectory(prefix="session-", dir=self.session_root),
                              now + self.session_ttl_seconds)
        self._sessions[key] = entry
        return entry

    @staticmethod
    async def _stop(process):
        # 仅回收本执行器通过 start_new_session 创建的进程组。
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), 2)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    async def _run(self, entry, arguments, *, password=None, max_bytes=262144):
        # 不将平台 JWT、数据库密钥等进程环境转交给原厂 Java 客户端。
        environment = {name: os.environ[name] for name in
            ("PATH", "HOME", "JAVA_HOME", "LANG", "LC_ALL", "TZ", "LD_LIBRARY_PATH") if name in os.environ}
        environment["EMCLI_STATE_DIR"] = entry.directory.name
        environment["HOME"] = entry.directory.name
        # 原厂客户端以空密码读取证书库；现代 JVM 默认 PKCS12 会丢失证书项。
        # 仅此私有子进程使用 JKS，不降低 TLS 协议、证书或系统 JVM 安全配置。
        security = Path(entry.directory.name) / "emcli.security"
        security.write_text("keystore.type=JKS\n")
        security.chmod(0o600)
        environment["EMCLI_OPTS"] = ("-Djava.security.properties=" + str(security)
            + " -Duser.home=" + entry.directory.name)
        if entry.client_home is not None:
            environment["EMCLI_OPTS"] += " -Demcli.dir=" + str(entry.client_home)
        if self.java_home:
            environment["JAVA_HOME"] = self.java_home
        try:
            process = await asyncio.create_subprocess_exec(self.executable, *arguments,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                cwd=entry.directory.name, env=environment, start_new_session=True)
        except OSError:
            fail("SOURCE_EMCLI_UNAVAILABLE")
        self._processes.add(process)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                if password is not None:
                    process.stdin.write((password + "\n").encode())
                    await process.stdin.drain()
                process.stdin.close()
                chunks, size = [], 0
                while chunk := await process.stdout.read(8192):
                    size += len(chunk)
                    if size > max_bytes:
                        fail("SOURCE_RESPONSE_TOO_LARGE")
                    chunks.append(chunk)
                code = await process.wait()
                if code:
                    fail("SOURCE_AUTH_FAILED" if password is not None else "SOURCE_PROVIDER_ERROR")
                return b"".join(chunks)
        except TimeoutError:
            await self._stop(process)
            fail("SOURCE_EMCLI_TIMEOUT")
        except BaseException:
            await self._stop(process)
            raise
        finally:
            self._processes.discard(process)

    async def _initialize(self, entry, context):
        endpoint = context.endpoint.rstrip("/").removesuffix("/api")
        parsed = urlparse(endpoint)
        if (not endpoint.endswith("/em") or parsed.scheme != "https" or not parsed.hostname
                or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
            fail("SOURCE_ENDPOINT_INVALID")
        pin = context.config.get("exadata_emcli_certificate_sha256", "")
        if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
            fail("SOURCE_TLS_PIN_REQUIRED")
        # 此连接不发送账号信息；只有固定叶证书指纹匹配后才允许原厂客户端接受证书。
        async with asyncio.timeout(self.timeout_seconds):
            _, writer = await asyncio.open_connection(parsed.hostname, parsed.port or 443,
                ssl=ssl._create_unverified_context(), server_hostname=parsed.hostname)
            try:
                certificate = writer.get_extra_info("ssl_object").getpeercert(binary_form=True)
                if not certificate or not hmac.compare_digest(hashlib.sha256(certificate).hexdigest(), pin):
                    fail("SOURCE_TLS_PIN_MISMATCH")
            finally:
                writer.close()
                await writer.wait_closed()
        if entry.initialized:
            return
        if entry.client_home is None:
            # setup/sync 要写原厂 bindings；不可写入共享安装目录或复用其他来源的会话。
            base = Path(self.executable).parent
            files = [path for path in base.iterdir() if path.suffix == ".jar" or path.name in {"version.txt", "emcli"}]
            if (not (base / "emCLI.jar").is_file() or len(files) > 128
                    or any(path.is_symlink() or not path.is_file() for path in files)
                    or sum(path.stat().st_size for path in files) > 128 * 1024 * 1024):
                fail("SOURCE_EMCLI_UNAVAILABLE")
            entry.client_home = Path(entry.directory.name) / "client"
            entry.client_home.mkdir(mode=0o700)
            for path in files:
                shutil.copy2(path, entry.client_home / path.name)
        await self._run(entry, ["setup", "-url=" + endpoint, "-username=" + context.credentials["username"],
            "-dir=" + entry.directory.name, "-localdirans=yes", "-licans=yes",
            "-certans=yes", "-noautologin"],
            password=context.credentials["password"], max_bytes=65536)
        # setup 已认证并建立会话；noautologin 只禁止保存明文登录凭据。
        entry.initialized = True

    async def read(self, context, target, definition, *, history, max_bytes, baseline_days=None):
        audited = load_catalog()[2].get(definition.get("code"))
        if audited != definition or target["targetType"] != definition["target_type"]:
            fail("SOURCE_QUERY_INVALID")
        name = target.get("targetName")
        if not isinstance(name, str) or not 1 <= len(name) <= 256 or any(char in name for char in "\r\n\0"):
            fail("SOURCE_TARGET_MISMATCH")
        if history and definition["native_value_type"] != "NUMBER":
            fail("SOURCE_METRIC_HISTORY_UNSUPPORTED")
        if baseline_days is not None and (not history or type(baseline_days) is not int
                or baseline_days not in (7, 30) or not definition.get("baseline_supported")):
            fail("SOURCE_QUERY_INVALID")
        group, column = definition["selector"].split("/")
        arguments = ["get_metric_data", "-target_type=" + definition["target_type"], "-target_name=" + name,
                     "-metric_name=" + group, "-metric_column=" + column, "-json"]
        if history:
            arguments.append("-last_n_days=" + str(baseline_days or 1))
        entry = self._session(context)
        async with entry.lock:
            try:
                await self._initialize(entry, context)
                body = await self._run(entry, arguments, max_bytes=max_bytes)
                value = json.loads(body, parse_float=str)
                if not isinstance(value, dict):
                    fail("SOURCE_RESPONSE_INVALID")
                return value, hashlib.sha256(body).hexdigest()
            except TimeoutError:
                entry.initialized = False
                fail("SOURCE_EMCLI_TIMEOUT")
            except (json.JSONDecodeError, UnicodeDecodeError):
                entry.initialized = False
                fail("SOURCE_RESPONSE_INVALID")
            except OSError:
                entry.initialized = False
                fail("SOURCE_EMCLI_UNAVAILABLE")
            except BaseException:
                entry.initialized = False
                raise

    async def aclose(self):
        self._closed = True
        await asyncio.gather(*(self._stop(process) for process in tuple(self._processes)))
        for entry in self._sessions.values():
            entry.directory.cleanup()
        self._sessions.clear()
