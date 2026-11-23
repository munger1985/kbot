"""有界原生图像/Docling 适配器；在隔离计算子进程内执行转换。"""

import asyncio
from dataclasses import dataclass
from errno import ENOMEM
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import warnings
from zipfile import ZipFile, BadZipFile

from knowledge_algorithms.contracts.content import ContentKind, ContentNode, Page, Provenance, ProvenanceKind, VisualAsset
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.identity import EvidenceKey, Locator, fingerprint
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.events import ParseArtifact, local_key
from knowledge_algorithms.parsing.formats import DOCUMENT_TYPES, IMAGE_TYPES, OFFICE_TYPES
from knowledge_algorithms.references.figures import extract_figure_references
from .markdown_parsing import guarded_markdown_backend, validate_markdown
from .model_transcription import ImageDimensionsUnsupported, TranscriptionImageTooLarge, WorkerModelTranscriber
from .pdf_text_spacing import REVISION as PDF_SPACING_REVISION, restore_pdf_word_spacing


def _is_conversion_memory_error(exc: BaseException) -> bool:
    """识别 Docling 流水线包装的内存错误，不误报其他转换故障。"""
    seen = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, MemoryError) or (isinstance(current, OSError) and current.errno == ENOMEM):
            return True
        if isinstance(current, (OSError, RuntimeError)):
            message = str(current).lower()
            if "mmap" in message and "cannot allocate memory" in message:
                return True
        cause = current.__cause__
        current = cause if cause is not None else (None if current.__suppress_context__ else current.__context__)
    return False


class WorkerArtifactWriter:
    """只在确切计算任务目录写不透明制品；不创建可检索共享存储。"""
    def __init__(self, directory, *, expires_at, max_bytes, max_total_bytes):
        self.directory, self.expires_at = Path(directory), expires_at
        self.max_bytes, self.max_total_bytes, self.consumed = max_bytes, max_total_bytes, 0
        self.artifacts = {}

    def put(self, raw, *, object_key, mime_type):
        require(0 < len(raw) <= self.max_bytes and self.consumed + len(raw) <= self.max_total_bytes,
                "转换制品超过大小预算")
        digest = sha256(raw).hexdigest()
        handle = fingerprint([object_key, digest])
        path = self.directory / ("artifact-" + digest)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        except FileExistsError:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                require(os.fstat(fd).st_size == len(raw), "同摘要制品大小冲突")
            finally:
                os.close(fd)
        else:
            with os.fdopen(fd, "wb") as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            directory_fd = os.open(self.directory, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        descriptor = ParseArtifact(handle, object_key, digest, mime_type, len(raw), self.expires_at)
        require(object_key not in self.artifacts, "同一对象产生了重复原生制品")
        self.artifacts[object_key] = descriptor
        self.consumed += len(raw)
        return descriptor


async def read_source(source, profile):
    data = bytearray()
    async for block in source.chunks():
        data.extend(block)
        require(len(data) <= profile.max_input_bytes, "转换输入超过预算")
    return bytes(data)


def checked_image(raw, *, max_pixels):
    from PIL import Image
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(raw)) as opened:
            require(opened.width * opened.height <= max_pixels and getattr(opened, "n_frames", 1) == 1,
                    "图片像素或帧数超过预算")
            opened.load()
            return opened.convert("RGB")


def native_key(source, run, profile, native_ref, kind):
    return EvidenceKey(source.descriptor.owner_binding_ref, source.descriptor.document_version_ref,
        run.parse_run_ref, local_key(algorithm_version=profile.algorithm_version, native_ref=native_ref, kind=kind.value))


def image_asset(image, source, run, profile, *, figure, locator, provenance, writer):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    key = native_key(source, run, profile, figure.native_ref + ":native-image", ContentKind.IMAGE)
    try:
        artifact = writer.put(buffer.getvalue(), object_key=key.object_key, mime_type="image/png")
    except AlgorithmError as exc:
        raise AlgorithmError(exc.code, "图像制品合同无效", stage="artifact_write") from exc
    except OSError as exc:
        raise AlgorithmError(ErrorCode.CHANNEL_FAILED, "图像制品写入失败", stage="artifact_write") from exc
    return VisualAsset(key, figure.key, artifact.handle_ref, artifact.sha256, artifact.mime_type,
                       artifact.byte_count, locator, provenance)



@dataclass(frozen=True)
class RegionTranscription:
    nodes: tuple[ContentNode, ...] = ()
    image_input_unsupported: bool = False


def is_office_separator(image, item, mime, *, is_page_image):
    """仅过滤无语义标记且像素形态明确的 Office 线条。"""
    label = getattr(item.label, "value", item.label)
    if (mime not in OFFICE_TYPES or is_page_image or label != "picture"
            or any(getattr(item, name, None) for name in
                   ("captions", "references", "footnotes", "children", "meta", "text"))):
        return False
    short, long = sorted(image.size)
    if not (0 < short <= 4 and short * 64 <= long <= 8192):
        return False
    horizontal = image.width == long
    # 允许首尾各一个抗锯齿像素；每条长边方向的内部像素必须完全一致。
    for offset in range(short):
        bounds = (1, offset, long - 1, offset + 1) if horizontal else (offset, 1, offset + 1, long - 1)
        with image.crop(bounds) as scanline, scanline.convert("RGBA") as rgba:
            if any(lo != hi for lo, hi in rgba.getextrema()):
                return False
    return True


async def transcribe_nodes(image, source, run, profile, *, figure, transcriber, max_nodes, complex_layout=False):
    if transcriber is None or "OCR" not in profile.repair_strategies:
        return RegionTranscription()
    if profile.ocr_model is not None:
        return await bound_transcription_nodes(image, source, run, profile, figure=figure,
                                               transcriber=transcriber, max_nodes=max_nodes, complex_layout=complex_layout)
    lines = await transcriber.transcribe(image, profile=profile)
    require(len(lines) <= max_nodes, "OCR 区域超过节点预算")
    result = []
    for index, line in enumerate(lines):
        native = figure.native_ref + ":ocr-line:" + str(index)
        key = native_key(source, run, profile, native, ContentKind.PARAGRAPH)
        parent = figure.locators[0]
        if parent.page_no is None:
            locator = Locator(logical_ref=native + ":bbox:" + ",".join(str(value) for value in line.bbox))
        else:
            x0, y0, x1, y1 = parent.bbox or (0.,0.,1.,1.)
            a,b,c,d = line.bbox
            locator = Locator(parent.page_no, (x0+(x1-x0)*a,y0+(y1-y0)*b,x0+(x1-x0)*c,y0+(y1-y0)*d))
        result.append(ContentNode(key, ContentKind.PARAGRAPH, line.text, (locator,),
            Provenance(ProvenanceKind.OCR, "local-ocr", transcriber.revision, source.descriptor.sha256),
            native, figure.reading_order + index + 1, parent_key=figure.key.object_key))
    return RegionTranscription(tuple(result))


def readable(text, profile):
    bad = sum(char == "\ufffd" or ord(char) < 32 and not char.isspace() for char in text)
    return any(char.isalnum() for char in text) and bad / max(len(text), 1) <= profile.max_bad_char_ratio


async def bound_transcription_nodes(image, source, run, profile, *, figure, transcriber, max_nodes, complex_layout=False):
    locator = figure.locators[0]
    box = locator.bbox
    full_page = figure.kind == ContentKind.PAGE or (box is not None and (box[2] - box[0]) * (box[3] - box[1]) >= 0.9)
    region = "page" if full_page else "figure"
    if isinstance(transcriber, WorkerModelTranscriber):
        transcriber.set_region(figure.key.object_key, locator.page_no, figure.kind == ContentKind.PAGE,
                               source.descriptor.mime_type in IMAGE_TYPES)
    try:
        text = await transcriber.transcribe_region(image, profile=profile, role="ocr", region=region)
    except TranscriptionImageTooLarge:
        # OCR 与 VLM 使用同一传输编码和上限，无须重复尝试同一张图。
        return RegionTranscription(image_input_unsupported=True)
    transcripts = [(text, profile.ocr_model, ProvenanceKind.OCR)] if readable(text, profile) else []
    image_input_unsupported = False
    if (profile.vlm_model is not None and (not full_page or profile.strategy == "VISUAL")
            and (not transcripts or (complex_layout and profile.strategy == "VISUAL"))):
        try:
            visual = await transcriber.transcribe_region(image, profile=profile, role="vlm", region=region)
        except (ImageDimensionsUnsupported, TranscriptionImageTooLarge):
            image_input_unsupported = True
        else:
            if readable(visual, profile) and visual != text:
                transcripts.append((visual, profile.vlm_model, ProvenanceKind.MODEL_TRANSCRIPTION))
    require(len(transcripts) <= max_nodes, "转写节点超过预算")
    result = []
    for order, (value, binding, kind) in enumerate(transcripts):
        native = figure.native_ref + ":" + kind.value.lower()
        key = native_key(source, run, profile, native, ContentKind.PARAGRAPH)
        result.append(ContentNode(key, ContentKind.PARAGRAPH, value, (locator,),
            Provenance(kind, binding.model_id, binding.revision, source.descriptor.sha256),
            native, figure.reading_order + order + 1, parent_key=figure.key.object_key))
    return RegionTranscription(tuple(result), image_input_unsupported)


class NativeImageAdapter:
    mime_types = IMAGE_TYPES
    revision = "native-image/v1"

    def __init__(self, writer, *, max_pixels=25000000, transcriber=None):
        self.writer, self.max_pixels, self.transcriber = writer, max_pixels, transcriber

    async def extract(self, source, profile, run):
        raw = await read_source(source, profile)
        image = checked_image(raw, max_pixels=self.max_pixels)
        try:
            native = "image:0"
            locator = Locator(logical_ref=native)
            key = native_key(source, run, profile, native, ContentKind.FIGURE)
            provenance = Provenance(ProvenanceKind.NATIVE, "pillow-image", self.revision, source.descriptor.sha256)
            node = ContentNode(key, ContentKind.FIGURE, "", (locator,), provenance, native, 0)
            asset = image_asset(image, source, run, profile, figure=node, locator=locator,
                                provenance=provenance, writer=self.writer)
            transcribed = await transcribe_nodes(image, source, run, profile, figure=node,
                transcriber=self.transcriber, max_nodes=profile.max_objects - 2)
            coverage = 1.0 if profile.strategy is None or transcribed.nodes and not transcribed.image_input_unsupported else 0.0
            regions = (ExtractedRegion(native, (node,), coverage),) + tuple(
                ExtractedRegion(text_node.native_ref, (text_node,), 1.0) for text_node in transcribed.nodes)
            yield ExtractionBatch((), regions, (asset,))
        finally:
            image.close()


class DoclingParsingAdapter:
    mime_types = frozenset(DOCUMENT_TYPES)
    revision = "docling-native/v1"

    def __init__(self, writer, *, converter_factory=None, artifacts_path=None, max_pixels=25000000, transcriber=None):
        self.writer, self.factory, self.artifacts_path = writer, converter_factory, artifacts_path
        self.max_pixels = max_pixels
        self.transcriber = transcriber

    def converter(self):
        if self.factory is not None:
            return self.factory()
        # 不允许转换器下载模型、访问外部服务或加载插件；PDF 模型必须预置。
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        from docling.document_converter import DocumentConverter, MarkdownFormatOption, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        options = PdfPipelineOptions(do_ocr=False, do_picture_description=False, do_picture_classification=False,
            generate_picture_images=True, generate_page_images=True, enable_remote_services=False, allow_external_plugins=False)
        if self.artifacts_path is not None:
            options.artifacts_path = Path(self.artifacts_path)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            InputFormat.MD: MarkdownFormatOption(backend=guarded_markdown_backend())})

    async def extract(self, source, profile, run):
        raw = await read_source(source, profile)
        mime = source.descriptor.mime_type
        if mime in OFFICE_TYPES:
            try:
                with ZipFile(BytesIO(raw)) as archive:
                    entries = archive.infolist()
                    require(len(entries) <= profile.max_objects
                            and sum(entry.file_size for entry in entries) <= profile.max_input_bytes * 8,
                            "Office 压缩展开超过预算")
                    require(all(not entry.flag_bits & 1 and not entry.filename.startswith("/")
                            and ".." not in entry.filename.split("/") for entry in entries), "Office 容器结构不受支持")
                    from xml.etree import ElementTree
                    for entry in entries:
                        if entry.filename.endswith(".rels"):
                            relationships = ElementTree.fromstring(archive.read(entry))
                            require(all(child.get("TargetMode") != "External"
                                or child.get("Type", "").endswith("/hyperlink") for child in relationships),
                                "Office 外部媒体不能由转换进程抓取")
            except BadZipFile as exc:
                raise AlgorithmError(ErrorCode.UNSUPPORTED_INPUT, "Office 容器无效", stage="convert") from exc
        if mime == "text/html":
            # 不渲染主动内容、不抓取 HTML 外部资源；图像只接受已嵌入的资源。
            from html.parser import HTMLParser
            class SafeHTML(HTMLParser):
                def handle_starttag(self, tag, attrs):
                    values = dict(attrs)
                    allowed = {"html", "head", "body", "title", "div", "span", "p", "br", "hr", "section",
                        "article", "header", "footer", "main", "aside", "nav", "h1", "h2", "h3", "h4", "h5",
                        "h6", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "colgroup",
                        "col", "ul", "ol", "li", "dl", "dt", "dd", "pre", "code", "blockquote", "b", "i",
                        "strong", "em", "u", "s", "sub", "sup", "a", "figure", "figcaption", "img"}
                    if tag not in allowed:
                        raise ValueError("HTML 包含不允许的主动内容或外部资源")
                    if any(k.lower().startswith("on") for k, _ in attrs):
                        raise ValueError("HTML 包含事件处理器")
                    if any(k.lower() in {"style", "background", "srcset", "poster", "data", "action",
                            "formaction", "xlink:href", "srcdoc"} or (k.lower() == "src" and tag != "img")
                            for k, _ in attrs):
                        raise ValueError("HTML 包含未登记资源属性")
                    if tag == "img" and not values.get("src", "").startswith("data:image/"):
                        raise ValueError("HTML 外部图片不允许由转换进程抓取")
            SafeHTML().feed(raw.decode("utf-8"))
        if mime == "text/markdown":
            validate_markdown(raw)
        try:
            from docling.datamodel.base_models import DocumentStream
            result = self.converter().convert(DocumentStream(name="source." + DOCUMENT_TYPES[mime], stream=BytesIO(raw)),
                max_num_pages=profile.max_pages, max_file_size=profile.max_input_bytes)
        except AlgorithmError:
            raise
        except Exception as exc:
            if _is_conversion_memory_error(exc):
                raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "文档转换内存不足", stage="convert") from None
            raise AlgorithmError(ErrorCode.PARSE_QUALITY_INSUFFICIENT, "原生转换异常", stage="convert") from exc
        status = getattr(result.status, "value", result.status)
        require(status == "success", "Docling 未完整成功转换；不静默发布部分文档", stage="convert")
        document = result.document
        pages = {int(number): page for number, page in document.pages.items()}
        require(len(pages) <= profile.max_pages, "转换页数超过预算")
        page_objects = tuple(Page(fingerprint(["page", n]), n, float(p.size.width), float(p.size.height), 0)
                             for n, p in sorted(pages.items()))
        entries = list(document.iterate_items())
        require(len(entries) <= profile.max_objects, "转换节点超过预算")
        kind_map = {"title": ContentKind.SECTION, "section_header": ContentKind.SECTION,
                    "text": ContentKind.PARAGRAPH, "paragraph": ContentKind.PARAGRAPH,
                    "list_item": ContentKind.LIST_ITEM, "caption": ContentKind.CAPTION,
                    "picture": ContentKind.FIGURE, "chart": ContentKind.FIGURE, "table": ContentKind.TABLE,
                    "code": ContentKind.CODE, "formula": ContentKind.FORMULA,
                    "footnote": ContentKind.FOOTNOTE, "page_header": ContentKind.HEADER, "page_footer": ContentKind.FOOTER}
        kinds = {item.self_ref: kind_map.get(str(getattr(item.label, "value", item.label)), ContentKind.PARAGRAPH)
                 for item, _ in entries if hasattr(item, "label")}
        captions = {}
        for item, _ in entries:
            if kinds.get(item.self_ref) == ContentKind.FIGURE:
                for ref in getattr(item, "captions", ()):
                    require(ref.cref not in captions, "Docling 图题属于多个逻辑图，拒绝猜测归属")
                    captions[ref.cref] = item.self_ref
        revision = self.revision + "+" + PDF_SPACING_REVISION if mime == "application/pdf" else self.revision
        provenance = Provenance(ProvenanceKind.NATIVE, "docling", revision, source.descriptor.sha256)
        regions, assets = [], []
        table_gaps = set()
        for order, (item, _) in enumerate(entries):
            if item.self_ref not in kinds:
                continue
            kind, native = kinds[item.self_ref], str(item.self_ref)
            key = native_key(source, run, profile, native, kind)
            text = (str(item.export_to_markdown(doc=document)) if kind == ContentKind.TABLE
                    else str(getattr(item, "text", "") or ""))
            locators = []
            for prov in getattr(item, "prov", ()):
                page = pages.get(int(prov.page_no))
                require(page is not None and prov.bbox is not None, "物理来源页或坐标缺失")
                box = prov.bbox.to_top_left_origin(float(page.size.height)).normalized(page.size)
                locators.append(Locator(int(prov.page_no), (float(box.l), float(box.t), float(box.r), float(box.b))))
            if not locators:
                require(not pages, "物理文档节点缺少来源；不能编造逻辑定位")
                locators = [Locator(logical_ref=native)]
            parent_ref = captions.get(native) or getattr(getattr(item, "parent", None), "cref", None)
            parent_key = (native_key(source, run, profile, parent_ref, kinds[parent_ref]).object_key
                          if parent_ref in kinds else None)
            label = None
            if kind == ContentKind.FIGURE:
                refs = getattr(item, "captions", ())
                explicit = [ref.resolve(document) for ref in refs]
                values = [occurrence.raw_label for caption in explicit
                          for occurrence in extract_figure_references(str(getattr(caption, "text", "")), key,
                              numbering_scope="document", policy_version=profile.algorithm_version)
                          if occurrence.span.char_start == 0]
                label = values[0] if len(values) == 1 else None
            level = getattr(item, "level", 1) if kind == ContentKind.SECTION else None
            node = ContentNode(key, kind, text, tuple(locators), provenance, native, order,
                heading_level=min(9, max(1, int(level or 1))) if level is not None else None,
                parent_key=parent_key, numbering_scope="document", display_label=label, native_anchor=native)
            if profile.strategy is not None and kind == ContentKind.TABLE and not readable(text, profile):
                table_gaps.update(loc.page_no for loc in locators)
            regions.append(ExtractedRegion(native, (node,), 1.0))
            if kind == ContentKind.FIGURE:
                image = item.get_image(document)
                if image is not None:
                    try:
                        require(image.width * image.height <= self.max_pixels, "Docling 图像超过像素预算")
                        assets.append(image_asset(image, source, run, profile, figure=node, locator=locators[0],
                                                 provenance=provenance, writer=self.writer))
                        box = locators[0].bbox
                        is_page_image = box is not None and (box[2] - box[0]) * (box[3] - box[1]) >= 0.9
                        if not ((profile.strategy is not None and is_page_image)
                                or is_office_separator(image, item, mime, is_page_image=is_page_image)):
                            transcribed = await transcribe_nodes(image, source, run, profile, figure=node,
                                transcriber=self.transcriber, max_nodes=profile.max_objects - len(regions) - len(assets))
                            if transcribed.image_input_unsupported:
                                regions[-1] = ExtractedRegion(native, (node,), 0.0)
                            regions.extend(ExtractedRegion(text_node.native_ref, (text_node,), 1.0)
                                           for text_node in transcribed.nodes)
                    finally:
                        image.close()
            await asyncio.sleep(0)
        if mime == "application/pdf":
            regions = restore_pdf_word_spacing(raw, regions)
        covered_pages = {locator.page_no for region in regions for node in region.nodes
                         if node.kind not in {ContentKind.HEADER, ContentKind.FOOTER} for locator in node.locators}
        if profile.strategy is not None:
            covered_pages = {loc.page_no for region in regions for node in region.nodes
                if node.kind not in {ContentKind.HEADER, ContentKind.FOOTER, ContentKind.FIGURE}
                and node.provenance.kind == ProvenanceKind.NATIVE
                and readable(node.text, profile) for loc in node.locators}
        for number, page in sorted(pages.items()):
            complex_layout = number in table_gaps
            if number in covered_pages and not (profile.strategy == "VISUAL" and complex_layout):
                continue
            # 不能忽略扫描页或空白页；至少保留可定位的原始页面图像。
            page_image = getattr(getattr(page, "image", None), "pil_image", None)
            require(page_image is not None, "存在无内容且未渲染的页面，不能宣称解析完整")
            image = page_image.copy()
            try:
                require(image.width * image.height <= self.max_pixels, "页面图像超过像素预算")
                native = "#/pages/" + str(number) + "/image"
                key = native_key(source, run, profile, native, ContentKind.PAGE)
                locator = Locator(number, (0., 0., 1., 1.))
                page_node = ContentNode(key, ContentKind.PAGE, "", (locator,), provenance,
                    native, len(entries) + number, numbering_scope="document", native_anchor=native)
                assets.append(image_asset(image, source, run, profile, figure=page_node,
                    locator=locator, provenance=provenance, writer=self.writer))
                transcribed = await transcribe_nodes(image, source, run, profile, figure=page_node,
                    transcriber=self.transcriber, max_nodes=profile.max_objects-len(regions)-len(assets)-1,
                    complex_layout=complex_layout)
                coverage = 1.0 if profile.strategy is None or transcribed.nodes and not transcribed.image_input_unsupported else 0.0
                regions.append(ExtractedRegion(native, (page_node,), coverage))
                regions.extend(ExtractedRegion(node.native_ref, (node,), 1.0) for node in transcribed.nodes)
            finally:
                image.close()
        require(bool(regions), "文档没有可定位内容")
        yield ExtractionBatch(page_objects, tuple(regions), tuple(assets))
