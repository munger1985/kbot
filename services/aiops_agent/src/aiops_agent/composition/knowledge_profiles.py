"""知识基础能力的正式服务配置和受控默认预算。"""

from pydantic import BaseModel, ConfigDict, Field
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.retrieval import RetrievalProfile, ChannelSpec
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.contracts import ParseProfile, LayoutProfile
from knowledge_algorithms.projections.contracts import ProjectionProfile, RepresentationSpec
from knowledge_algorithms.retrieval.selection import PayloadCost


class KnowledgeFoundationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    storage_root: str = "./var/data/app_knowledge"
    max_input_bytes: int = Field(default=67108864, ge=1024, le=268435456)
    parse_timeout_seconds: float = Field(default=180.0, ge=10, le=1800)
    max_pages: int = Field(default=1000, ge=1, le=10000)
    max_units: int = Field(default=20000, ge=1, le=100000)
    max_tokens_per_unit: int = Field(default=1024, ge=64, le=8192)
    image_token_cost: int = Field(default=4096, ge=1)
    payload_tokens: int = Field(default=24000, ge=256)
    payload_images: int = Field(default=8, ge=0, le=64)
    payload_bytes: int = Field(default=16777216, ge=65536)
    representations: list[dict] = Field(default_factory=list, max_length=8)
    served_model_names: dict[str, str] = Field(default_factory=dict)
    poll_seconds: float = Field(default=2.0, ge=0.1, le=10)
    lease_seconds: int = Field(default=360, ge=15, le=600)

    def profiles(self):
        specs = tuple(decode(RepresentationSpec, value) for value in self.representations)
        require(set(self.served_model_names) == {spec.model_ref for spec in specs}
                and all(name == name.strip() and bool(name) and len(name) <= 128
                        for name in self.served_model_names.values()),
                "表示空间必须显式绑定模型 UUID 与 Model Serving 的 served_model_name")
        parse = ParseProfile("app-parse/v1", "knowledge-algorithms/v1", ("zh", "en"),
            self.max_input_bytes, self.max_pages, self.max_units, self.max_input_bytes,
            1048576, 64, self.parse_timeout_seconds, 0, (), .05, .8, False, LayoutProfile(.8, .05))
        projection = ProjectionProfile("app-projection/v1", self.max_tokens_per_unit,
            self.max_input_bytes, self.max_units, 64, 16000000, self.parse_timeout_seconds, specs)
        channels = [ChannelSpec("lexical", "lexical", "LEXICAL", "TEXT_TO_TEXT", 1.0, 100, 30., True),
                    ChannelSpec("exact-image", "image-identity", "NEAR_DUPLICATE", "IMAGE_TO_IMAGE",
                                1.0, 100, 30., False)]
        for spec in specs:
            for direction in sorted(spec.directions):
                key = spec.space_id + ":" + direction
                channels.append(ChannelSpec(key, key, spec.kind, direction, 1., 100, 60., False, spec))
        retrieval = RetrievalProfile("app-retrieval/v1", tuple(channels), 60., 200, 100,
            PayloadCost(self.payload_tokens, self.payload_images, self.payload_bytes),
            20000, 64, 8, 16000000, 120.)
        return parse, projection, {"default": retrieval}


class ConservativeTokenizer:
    """UTF-8 字节上界而非伪称真实模型 tokenizer；宁可少装载也不低估。"""
    revision = "utf8-byte-upper-bound/v1"

    def count(self, text):
        return len(text.encode("utf-8"))
