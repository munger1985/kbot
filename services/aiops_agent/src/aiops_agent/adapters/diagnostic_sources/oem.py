"""Oracle Enterprise Manager官方原生与标准映射只读Adapter。"""

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError
from .oem_native import OEMNativeMetrics
from .oem_events import OEMEvents
from .oem_standard import OEMStandardMetrics


class OEMAdapter(OEMEvents, OEMStandardMetrics, OEMNativeMetrics, BaseDiagnosticSourceAdapter):
    async def verify_and_normalize_webhook(self, request):
        raise DiagnosticSourceAdapterError(
            "SOURCE_QUERY_UNSUPPORTED", "OEM不直接接收Incident Webhook",
        )
