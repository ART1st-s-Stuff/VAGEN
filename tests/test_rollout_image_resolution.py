from PIL import Image

from vagen.rollout.image_utils import prepare_rollout_image


def test_prepare_rollout_image_preserves_255_rgb_pixels():
    image = Image.new("RGB", (255, 255), color=(12, 34, 56))

    actual = prepare_rollout_image(image)

    assert actual.mode == "RGB"
    assert actual.size == (255, 255)
    assert actual.getpixel((127, 127)) == (12, 34, 56)


def test_prepare_rollout_image_converts_mode_and_resizes_to_255():
    image = Image.new("RGBA", (512, 384), color=(12, 34, 56, 78))

    actual = prepare_rollout_image(image)

    assert actual.mode == "RGB"
    assert actual.size == (255, 255)
