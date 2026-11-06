"""有界原生图像/Docling 适配器；在隔离计算子进程内执行转换。"""

import asyncio
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
from knowledge_algorithms.references.figures import extract_figure_references


OFFICE_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
}
DOCUMENT_TYPES = {**OFFICE_TYPES, "application/pdf": "pdf", "text/html": "html", "text/markdown": "md"}
IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/tiff"})


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
    artifact = writer.put(buffer.getvalue(), object_key=key.object_key, mime_type="image/png")
    return VisualAsset(key, figure.key, artifact.handle_ref, artifact.sha256, artifact.mime_type,
                       artifact.byte_count, locator, provenance)



async def transcribe_nodes(image, source, run, profile, *, figure, transcriber, max_nodes):
    if transcriber is None or "OCR" not in profile.repair_strategies:
        return ()
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
    return tuple(result)


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
            regions = (ExtractedRegion(native, (node,), 1.0),) + tuple(
                ExtractedRegion(text_node.native_ref, (text_node,), 1.0) for text_node in transcribed)
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
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        options = PdfPipelineOptions(do_ocr=False, do_picture_description=False, do_picture_classification=False,
            generate_picture_images=True, generate_page_images=True, enable_remote_services=False, allow_external_plugins=False)
        if self.artifacts_path is not None:
            options.artifacts_path = Path(self.artifacts_path)
        return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})

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
            import re
            markdown = raw.decode("utf-8")
            # 仅允许明确内嵌的行内图片；引用式、嵌套式或外部图片一律拒绝。
            without_embedded = re.sub(r"!\[[^\]\n]*\]\(data:image/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+\)",
                                      "", markdown)
            require("![" not in without_embedded and "<" not in markdown,
                    "Markdown 含未登记媒体或 HTML；不能由转换进程抓取外部资源")
        from docling.datamodel.base_models import DocumentStream
        result = self.converter().convert(DocumentStream(name="source." + DOCUMENT_TYPES[mime], stream=BytesIO(raw)),
            max_num_pages=profile.max_pages, max_file_size=profile.max_input_bytes)
        status = getattr(result.status, "value", result.status)
        require(status == "success", "Docling 未完整成功转换；不静默发布部分文档")
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
        provenance = Provenance(ProvenanceKind.NATIVE, "docling", self.revision, source.descriptor.sha256)
        regions, assets = [], []
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
            regions.append(ExtractedRegion(native, (node,), 1.0))
            if kind == ContentKind.FIGURE:
                image = item.get_image(document)
                if image is not None:
                    try:
                        require(image.width * image.height <= self.max_pixels, "Docling 图像超过像素预算")
                        assets.append(image_asset(image, source, run, profile, figure=node, locator=locators[0],
                                                 provenance=provenance, writer=self.writer))
                        transcribed = await transcribe_nodes(image, source, run, profile, figure=node,
                            transcriber=self.transcriber, max_nodes=profile.max_objects - len(regions) - len(assets))
                        regions.extend(ExtractedRegion(text_node.native_ref,(text_node,),1.0) for text_node in transcribed)
                    finally:
                        image.close()
            await asyncio.sleep(0)
        covered_pages = {locator.page_no for region in regions for node in region.nodes
                         if node.kind not in {ContentKind.HEADER, ContentKind.FOOTER} for locator in node.locators}
        for number, page in sorted(pages.items()):
            if number in covered_pages:
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
                regions.append(ExtractedRegion(native, (page_node,), 1.0))
                assets.append(image_asset(image, source, run, profile, figure=page_node,
                    locator=locator, provenance=provenance, writer=self.writer))
                transcribed = await transcribe_nodes(image, source, run, profile, figure=page_node,
                    transcriber=self.transcriber, max_nodes=profile.max_objects-len(regions)-len(assets))
                regions.extend(ExtractedRegion(node.native_ref, (node,), 1.0) for node in transcribed)
            finally:
                image.close()
        require(bool(regions), "文档没有可定位内容")
        yield ExtractionBatch(page_objects, tuple(regions), tuple(assets))
