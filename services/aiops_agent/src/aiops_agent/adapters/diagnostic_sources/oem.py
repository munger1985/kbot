"""Oracle Enterprise Manager官方原生与标准映射只读Adapter。"""

from .base import BaseDiagnosticSourceAdapter, DiagnosticSourceAdapterError
from .oem_native import OEMNativeMetrics
from .oem_events import OEMEvents
from .oem_standard import OEMStandardMetrics
from .oem_discovery import OEMInstanceDiscovery
from .oem_exadata import OEMExadata


class OEMAdapter(OEMExadata, OEMInstanceDiscovery, OEMEvents, OEMStandardMetrics, OEMNativeMetrics, BaseDiagnosticSourceAdapter):
    def __init__(self, *, exadata_emcli_client=None, **kwargs):
        super().__init__(**kwargs)
        self._exadata_emcli_client = exadata_emcli_client

    async def verify_and_normalize_webhook(self, request):
        raise DiagnosticSourceAdapterError(
            "SOURCE_QUERY_UNSUPPORTED", "OEM不直接接收Incident Webhook",
        )
