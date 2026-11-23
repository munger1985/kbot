"""授权图片候选的有界截图核验：白底归一化、缩放和裁剪匹配。"""

from io import BytesIO
import math
import warnings

import numpy as np
from PIL import Image, ImageFilter, ImageOps


_MAX_EDGE = 224
_MIN_EDGE = 20
_PIXEL_LIMIT = 25_000_000
_MIN_ENERGY = 2.
_SOURCE_CORRELATION = .82
_SIMILAR_CORRELATION = .68
IMAGE_VERIFICATION_LIMIT = 16


def image_for_verification(raw: bytes) -> Image.Image | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as opened:
                if opened.width * opened.height > _PIXEL_LIMIT or getattr(opened, "n_frames", 1) != 1:
                    return None
                # 在合成白底前缩小，避免给高像素图片额外分配多份彩色副本。
                opened.thumbnail((_MAX_EDGE, _MAX_EDGE), Image.Resampling.LANCZOS)
                rgba = ImageOps.exif_transpose(opened).convert("RGBA")
                white = Image.new("L", rgba.size, 255)
                white.paste(rgba.convert("L"), mask=rgba.getchannel("A"))
                return white
    except (OSError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        return None


def _contrast(image):
    smoothed = image.filter(ImageFilter.GaussianBlur(radius=.75))
    background = smoothed.filter(ImageFilter.GaussianBlur(radius=2.5))
    return np.asarray(smoothed, dtype=np.float64) - np.asarray(background, dtype=np.float64)


def _window_sum(integral, h, w):
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def _integral(array):
    return np.pad(array.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _correlate(search: Image.Image, template: Image.Image, *, acceptance: float) -> float:
    fit = min(search.width / template.width, search.height / template.height)
    if min(template.size) * fit < _MIN_EDGE:
        return 0.
    canvas = _contrast(search)
    shape = tuple(1 << (size - 1).bit_length() for size in canvas.shape)
    spectrum = np.fft.rfft2(canvas, s=shape, axes=(0, 1))
    integral, energy = _integral(canvas), _integral(canvas * canvas)
    best, seen = 0., set()
    # 逐级缩放查找图块；输入和变换画布都有上限，不读取整页原文。
    for level in range(57):
        scale = fit * math.exp(-.025 * level)
        w, h = round(template.width * scale), round(template.height * scale)
        if min(w, h) < _MIN_EDGE:
            break
        if (w, h) in seen:
            continue
        seen.add((w, h))
        patch = _contrast(template.resize((w, h), Image.Resampling.LANCZOS))
        patch -= patch.mean()
        patch_energy = float(np.sum(patch * patch))
        if patch_energy < _MIN_ENERGY ** 2 * h * w:
            continue
        variance = _window_sum(energy, h, w) - _window_sum(integral, h, w) ** 2 / (h * w)
        viable = variance >= _MIN_ENERGY ** 2 * h * w
        if not viable.any():
            continue
        cross = np.fft.irfft2(spectrum * np.fft.rfft2(patch, s=shape, axes=(0, 1)).conj(),
                             s=shape, axes=(0, 1))[:viable.shape[0], :viable.shape[1]]
        score = float(np.max(cross[viable] / np.sqrt(patch_energy * variance[viable])))
        best = max(best, score)
        if best >= acceptance:
            break
    return best


def source_correlation(query: Image.Image, candidate: Image.Image) -> float:
    return max(_correlate(query, candidate, acceptance=1.), _correlate(candidate, query, acceptance=1.))


def verifies_image(query: Image.Image, candidate_raw: bytes, *, similar: bool) -> bool:
    candidate = image_for_verification(candidate_raw)
    if candidate is None:
        return False
    floor = _SIMILAR_CORRELATION if similar else _SOURCE_CORRELATION
    return (_correlate(query, candidate, acceptance=floor) >= floor
            or _correlate(candidate, query, acceptance=floor) >= floor)
