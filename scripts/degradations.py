"""Apply image degradations and calculate SSIM."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.metrics import structural_similarity


IMAGE_SIZE = (224, 224)

PARAMETER_GRID = {
    "blur": (0.25, 0.5, 1.0, 2.0, 4.0),
    "jpeg": (90, 70, 50, 30, 10),
    "resolution": (192, 128, 96, 64, 32),
    "brightness": (0.4, 0.7, 1.2, 1.5, 1.8),
}


def prepare_image(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)


def degrade(image: Image.Image, kind: str, parameter: float) -> Image.Image:
    if image.mode != "RGB" or image.size != IMAGE_SIZE:
        raise ValueError("Degradations expect a 224 x 224 RGB image.")
    if kind == "blur":
        if parameter <= 0:
            raise ValueError("Blur radius must be positive.")
        return image.filter(ImageFilter.GaussianBlur(radius=parameter))
    if kind == "jpeg":
        if not 1 <= parameter <= 95 or int(parameter) != parameter:
            raise ValueError("JPEG quality must be an integer from 1 to 95.")
        with BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=int(parameter), subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as compressed:
                return compressed.convert("RGB")
    if kind == "resolution":
        if not 1 <= parameter < IMAGE_SIZE[0] or int(parameter) != parameter:
            raise ValueError("Resolution must be an integer from 1 to 223.")
        small = image.resize((int(parameter), int(parameter)), Image.Resampling.BILINEAR)
        return small.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    if kind == "brightness":
        if parameter <= 0:
            raise ValueError("Brightness factor must be positive.")
        return ImageEnhance.Brightness(image).enhance(parameter)
    raise ValueError(f"Unknown degradation: {kind}")


def calculate_ssim(original: Image.Image, degraded: Image.Image) -> float:
    return float(structural_similarity(
        np.asarray(original),
        np.asarray(degraded),
        data_range=255,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    ))
