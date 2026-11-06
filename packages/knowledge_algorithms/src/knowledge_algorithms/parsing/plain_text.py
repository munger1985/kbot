"""可直接执行的 UTF-8 原生文本适配器；复杂转换器仍留在服务侧。"""

import codecs
import re

from ..contracts.content import ContentKind, ContentNode, Provenance, ProvenanceKind
from ..contracts.errors import AlgorithmError, ErrorCode
from ..contracts.identity import EvidenceKey, Locator
from .contracts import ExtractedRegion, ExtractionBatch
from .events import local_key


class PlainTextAdapter:
    mime_types = frozenset({"text/plain"})
    revision = "native-utf8/v1"
    materializes_text = True

    async def extract(self, source, profile, run):
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        pieces, size = [], 0
        async for block in source.chunks():
            decoded = decoder.decode(block)
            size += len(decoded.encode("utf-8"))
            if size > profile.max_text_bytes:
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "原生文本超过物化上限", stage="extract")
            pieces.append(decoded)
        pieces.append(decoder.decode(b"", final=True))
        text = "".join(pieces)
        for index, match in enumerate(re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)):
            native_ref = f"text:{match.start()}:{match.end()}"
            object_key = local_key(algorithm_version=profile.algorithm_version, native_ref=native_ref,
                                   kind=ContentKind.PARAGRAPH.value)
            key = EvidenceKey(source.descriptor.owner_binding_ref, source.descriptor.document_version_ref,
                              run.parse_run_ref, object_key)
            node = ContentNode(key, ContentKind.PARAGRAPH, match.group(), (Locator(logical_ref=native_ref),),
                Provenance(ProvenanceKind.NATIVE, "native-utf8", self.revision, source.descriptor.sha256), native_ref, index)
            yield ExtractionBatch((), (ExtractedRegion(object_key, (node,), 1.0),), ())
