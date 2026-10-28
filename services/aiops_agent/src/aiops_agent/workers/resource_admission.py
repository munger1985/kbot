"""监控和日志的来源调用同数据库查询共用持久化配额。"""

import asyncio
from datetime import datetime, timezone
import hashlib

from loguru import logger

from aiops_agent.application.resource_governance import ResourceAdmissionError
from aiops_agent.contracts.evidence import LogEvidenceSet, ObservationGap, ObservationSet
from aiops_agent.workers.handlers import investigation_task_identity
from aiops_agent.resource_admission import RETRYABLE_RESOURCE_CODES, wait_for_resource
from platform_core.identity import uuid7


class ResourceBoundSourceHandler:
    def __init__(self, *, handler, governance, prefix):
        self._handler = handler
        self._governance = governance
        self._prefix = prefix

    async def execute(self, context):
        binding_id = investigation_task_identity(context.task_key, self._prefix)
        monitoring = context.plan_snapshot["monitoring"]
        snapshot = next((item for item in monitoring["bindings"] if item["binding_id"] == binding_id), None)
        if snapshot is None:
            return await self._handler.execute(context)
        source_id = snapshot["source"]["source_id"]
        receipt = None
        clean = False
        try:
            lease_id = uuid7()
            receipt = await wait_for_resource(
                lambda: self._governance.acquire_monitor(context=context, source_id=source_id, lease_id=lease_id),
                retry_errors=(ResourceAdmissionError,),
            )
            remaining = (receipt.execution_deadline_at - datetime.now(timezone.utc)).total_seconds()
            if remaining <= 0:
                raise ResourceAdmissionError("EXECUTION_DEADLINE_EXCEEDED")
            async with asyncio.timeout(remaining):
                result = await self._handler.execute(context)
                if datetime.now(timezone.utc) >= receipt.execution_deadline_at:
                    raise ResourceAdmissionError("EXECUTION_DEADLINE_EXCEEDED")
            clean = True
            return result
        except (ResourceAdmissionError, TimeoutError) as exc:
            code = exc.code if isinstance(exc, ResourceAdmissionError) else "EXECUTION_DEADLINE_EXCEEDED"
            gap = ObservationGap(source_id=source_id, binding_id=binding_id, code=code,
                detail="本次取证未取得共享资源额度或执行期限已到；不表示来源没有数据",
                retryable=code in RETRYABLE_RESOURCE_CODES)
            if self._prefix == "observe:":
                return ObservationSet(target_id=context.target_id, binding_id=binding_id,
                    source_id=source_id, gaps=(gap,), collected_at=datetime.now(timezone.utc))
            window = monitoring["window"]
            start, end = (datetime.fromisoformat(window[name].replace("Z", "+00:00")) for name in ("start", "end"))
            return LogEvidenceSet(target_id=context.target_id, binding_id=binding_id, source_id=source_id,
                window_start=start, window_end=end, gaps=(gap,), collected_at=datetime.now(timezone.utc),
                query_fingerprint=hashlib.sha256(f"{binding_id}|{window['start']}|{window['end']}".encode()).hexdigest())
        finally:
            if clean and receipt is not None:
                try:
                    async with asyncio.timeout(3):
                        await self._governance.release(receipt)
                except Exception:
                    logger.warning("来源取证租约未确认释放；保留控制面清理宽限")
