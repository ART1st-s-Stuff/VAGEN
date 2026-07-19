"""Image normalization shared by rollout managers."""

from __future__ import annotations

from PIL import Image

ROLLOUT_IMAGE_SIZE = 255


def prepare_rollout_image(image: Image.Image) -> Image.Image:
    """Return the RGB 255x255 image consumed and persisted by rollout."""

    if not isinstance(image, Image.Image):
        raise TypeError(f"rollout image must be PIL.Image.Image, got {type(image)!r}")
    image = image.convert("RGB")
    if image.size != (ROLLOUT_IMAGE_SIZE, ROLLOUT_IMAGE_SIZE):
        image = image.resize(
            (ROLLOUT_IMAGE_SIZE, ROLLOUT_IMAGE_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
    return image
