"""业务图片文字提取使用的独立 OCR Provider 实现。"""

from __future__ import annotations

import asyncio
import base64
import csv
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
import subprocess
from typing import Any

import aiohttp

from platform_core.dictionary import OCRProvider


_DEEPSEEK_GROUNDING_BLOCK = re.compile(
    r"^(text|title|table_caption|table|image|code|formula)"
    r"\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]",
    re.MULTILINE,
)


class OCRModel:
    """一个按目录配置创建的 OCR 引擎；不依赖 Knowledge Core 的 Docling 链路。"""

    def __init__(self, *, model_data: dict[str, Any]):
        self.model_id = str(model_data["model_id"])
        self.provider = str(model_data["provider"])
        self.model_name = str(model_data["provider_model_name"])
        self.api_endpoint = str(model_data.get("api_endpoint") or "").strip()
        self.api_key = str(model_data.get("api_key") or "").strip()
        self.params = dict(model_data.get("model_params") or {})
        self.revision = str(self.params.get("revision") or self.model_name)
        self._engine: object | None = None
        self._http: aiohttp.ClientSession | None = None

    async def startup(self) -> None:
        if self.provider == OCRProvider.API_DEEPSEEK_OCR.value:
            if not self.api_endpoint or not self.api_key:
                raise ValueError("DeepSeek OCR 缺少 API endpoint 或 API key")
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=float(self.params.get("timeout_seconds", 300)),
                ),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            return
        self._engine = await asyncio.to_thread(self._create_engine)

    async def shutdown(self) -> None:
        http, self._http = self._http, None
        if http is not None:
            await http.close()
        engine, self._engine = self._engine, None
        close = getattr(engine, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def health_check(self) -> None:
        if self._engine is None and self._http is None:
            raise RuntimeError("OCR 引擎尚未初始化")

    async def infer(self, image: bytes) -> tuple[str, list[dict[str, Any]]]:
        if self.provider == OCRProvider.API_DEEPSEEK_OCR.value:
            return await self._infer_deepseek_ocr(image)
        if self._engine is None:
            raise RuntimeError("OCR 引擎尚未初始化")
        return await asyncio.to_thread(self._infer_sync, image)

    def _create_engine(self) -> object:
        if self.provider == OCRProvider.LOCAL_RAPIDOCR.value:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("未安装 RapidOCR 依赖") from exc
            return RapidOCR(
                config_path=self.params.get("config_path"),
                params=self.params.get("rapidocr_params"),
            )
        if self.provider == OCRProvider.LOCAL_EASYOCR.value:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("未安装 EasyOCR 依赖") from exc
            model_path = str(self.params.get("model_path") or "").strip()
            return easyocr.Reader(
                list(self.params.get("languages") or ["ch_sim", "en"]),
                gpu=self._easyocr_gpu_value(),
                model_storage_directory=model_path or None,
                download_enabled=bool(self.params.get("download_enabled", False)),
                verbose=False,
            )
        if self.provider == OCRProvider.LOCAL_TESSERACT.value:
            executable = str(self.params.get("executable") or "tesseract")
            try:
                subprocess.run(
                    [executable, "--version"], check=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeError("Tesseract 可执行程序不可用") from exc
            return executable
        raise ValueError(f"不支持的 OCR Provider：{self.provider}")

    def _easyocr_gpu_value(self) -> bool | str:
        device = str(self.params.get("device") or "cpu").strip().lower()
        return False if device in {"", "cpu"} else device

    def _infer_sync(self, image: bytes) -> tuple[str, list[dict[str, Any]]]:
        if self.provider == OCRProvider.LOCAL_RAPIDOCR.value:
            return self._normalize_rapidocr(self._engine(image))  # type: ignore[operator]
        if self.provider == OCRProvider.LOCAL_EASYOCR.value:
            import numpy as np
            from PIL import Image

            with Image.open(BytesIO(image)) as source:
                pixels = np.asarray(source.convert("RGB"))
            result = self._engine.readtext(  # type: ignore[union-attr]
                pixels, detail=1, paragraph=False,
            )
            return self._normalize_easyocr(result)
        return self._run_tesseract(image)

    @staticmethod
    def _normalize_rapidocr(result: Any) -> tuple[str, list[dict[str, Any]]]:
        if result is None:
            return "", []
        raw_blocks = result.to_json() or []
        blocks = json.loads(raw_blocks) if isinstance(raw_blocks, str) else raw_blocks
        if isinstance(blocks, dict):
            blocks = blocks.get("res") or blocks.get("result") or [blocks]
        return "\n".join(result.txts or ()), list(blocks or ())

    @staticmethod
    def _normalize_easyocr(result: Any) -> tuple[str, list[dict[str, Any]]]:
        blocks = [
            {
                "text": str(text),
                "confidence": float(confidence),
                "polygon": [[float(x), float(y)] for x, y in polygon],
            }
            for polygon, text, confidence in result
        ]
        return "\n".join(block["text"] for block in blocks), blocks

    def _run_tesseract(self, image: bytes) -> tuple[str, list[dict[str, Any]]]:
        languages = self.params.get("languages") or ["chi_sim", "eng"]
        command = [
            str(self._engine), "stdin", "stdout", "-l", "+".join(map(str, languages)),
        ]
        model_path = str(self.params.get("model_path") or "").strip()
        if model_path:
            command.extend(["--tessdata-dir", str(Path(model_path))])
        command.append("tsv")
        try:
            completed = subprocess.run(
                command, input=image, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True, timeout=float(self.params.get("timeout_seconds", 120)),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Tesseract OCR 推理超时") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("Tesseract OCR 推理失败") from exc
        rows = csv.DictReader(
            StringIO(completed.stdout.decode("utf-8", errors="replace")), delimiter="\t",
        )
        blocks: list[dict[str, Any]] = []
        lines: dict[tuple[str, str, str, str], list[str]] = {}
        for row in rows:
            word = str(row.get("text") or "").strip()
            if not word:
                continue
            blocks.append({
                "text": word,
                "confidence": float(row.get("conf") or -1),
                "bbox": {
                    key: int(row.get(key) or 0)
                    for key in ("left", "top", "width", "height")
                },
            })
            line_key = tuple(
                str(row.get(key) or "0")
                for key in ("page_num", "block_num", "par_num", "line_num")
            )
            lines.setdefault(line_key, []).append(word)
        return "\n".join(" ".join(words) for words in lines.values()), blocks

    async def _infer_deepseek_ocr(self, image: bytes) -> tuple[str, list[dict[str, Any]]]:
        if self._http is None:
            raise RuntimeError("DeepSeek OCR API 客户端尚未初始化")
        mime_type = self._image_mime_type(image)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": str(self.params.get("prompt") or "<|grounding|>Convert the document to markdown.")},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"}},
            ]}],
            "stream": True,
            "max_tokens": int(self.params.get("max_tokens", 8192)),
            "temperature": float(self.params.get("temperature", 0.0)),
        }
        try:
            async with self._http.post(self.api_endpoint, json=payload) as response:
                if response.status >= 400:
                    raise RuntimeError(f"DeepSeek OCR API 返回 HTTP {response.status}")
                chunks: list[str] = []
                while not response.content.at_eof():
                    raw_line = await response.content.readline()
                    if not raw_line:
                        break
                    chunks.extend(self._decode_deepseek_stream_line(raw_line))
        except asyncio.TimeoutError as exc:
            raise RuntimeError("DeepSeek OCR API 响应超时") from exc
        except aiohttp.ClientError as exc:
            raise RuntimeError("DeepSeek OCR API 不可用") from exc
        output = "".join(chunks).strip()
        if not output:
            raise RuntimeError("DeepSeek OCR API 返回空内容")
        return output, self._parse_deepseek_grounding_blocks(output)

    @staticmethod
    def _image_mime_type(image: bytes) -> str:
        from PIL import Image

        with Image.open(BytesIO(image)) as source:
            image_format = str(source.format or "").upper()
        return Image.MIME.get(image_format, "image/jpeg")

    @staticmethod
    def _decode_deepseek_stream_line(raw_line: bytes | str) -> list[str]:
        line = (
            raw_line.decode("utf-8", errors="replace")
            if isinstance(raw_line, bytes) else raw_line
        ).strip()
        if not line or line in {"data: [DONE]", "[DONE]"}:
            return []
        if line.startswith("data: "):
            line = line[6:]
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        return [
            str(
                choice.get("delta", {}).get("content")
                or choice.get("message", {}).get("content") or ""
            )
            for choice in payload.get("choices", ())
            if (
                choice.get("delta", {}).get("content")
                or choice.get("message", {}).get("content")
            )
        ]

    @staticmethod
    def _parse_deepseek_grounding_blocks(output: str) -> list[dict[str, Any]]:
        matches = list(_DEEPSEEK_GROUNDING_BLOCK.finditer(output))
        return [
            {
                "type": match.group(1),
                "bbox": [int(match.group(position)) for position in range(2, 6)],
                "text": output[
                    match.end(): matches[index + 1].start()
                    if index + 1 < len(matches) else len(output)
                ].strip(),
            }
            for index, match in enumerate(matches)
        ]
