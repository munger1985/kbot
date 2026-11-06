"""可选本地 OCR 转录；固定二进制、TSV 原生坐标、有界输出，不生成图片描述。"""

import asyncio
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import re

from knowledge_algorithms.contracts.errors import require


@dataclass(frozen=True, slots=True)
class OCRLine:
    text: str
    bbox: tuple[float, float, float, float]


class TesseractTranscriber:
    def __init__(self, *, executable, revision, languages):
        require(Path(executable).is_absolute() and revision
                and re.fullmatch(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*", languages), "本地 OCR 配置无效")
        self.executable, self.revision, self.languages = executable, revision, languages

    async def transcribe(self, image, *, profile):
        raw = BytesIO()
        image.save(raw, format="PNG")
        process = await asyncio.create_subprocess_exec(self.executable, "stdin", "stdout", "-l", self.languages,
            "tsv", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        async def write():
            process.stdin.write(raw.getvalue())
            await process.stdin.drain()
            process.stdin.close()
        writing = asyncio.create_task(write())
        chunks, size = [], 0
        try:
            async with asyncio.timeout(profile.timeout_seconds):
                while block := await process.stdout.read(65536):
                    size += len(block)
                    require(size <= min(profile.max_text_bytes * 8, 67108864), "OCR TSV 超过输出预算")
                    chunks.append(block)
                await writing
                require(await process.wait() == 0, "OCR 转录没有成功")
            return parse_tsv(b"".join(chunks).decode("utf-8"), width=image.width, height=image.height,
                             max_lines=profile.max_objects, max_text_bytes=profile.max_text_bytes)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if not writing.done():
                writing.cancel()
            await asyncio.gather(writing, return_exceptions=True)


def parse_tsv(value, *, width, height, max_lines, max_text_bytes):
    require(width > 0 and height > 0, "OCR 原图尺寸无效")
    lines, total = {}, 0
    for row in csv.DictReader(StringIO(value), delimiter="\t"):
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        key = tuple(int(row[name]) for name in ("page_num", "block_num", "par_num", "line_num"))
        x, y, w, h = (int(row[name]) for name in ("left", "top", "width", "height"))
        require(0 <= x < x+w <= width and 0 <= y < y+h <= height, "OCR 原生坐标越界")
        total += len(row["text"].encode("utf-8"))
        require(total <= max_text_bytes and (key in lines or len(lines) < max_lines), "OCR 文本或行数超过预算")
        lines.setdefault(key, []).append((row["text"], x, y, x+w, y+h))
    result = []
    for words in lines.values():
        result.append(OCRLine(" ".join(word[0] for word in words),
            (min(word[1] for word in words)/width, min(word[2] for word in words)/height,
             max(word[3] for word in words)/width, max(word[4] for word in words)/height)))
    return tuple(result)
