"""Validate that an allocated GPU can produce non-uniform AI2-THOR frames."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image


@dataclass(frozen=True)
class DirectRenderProbeResult:
    scene: str
    elapsed_seconds: float
    image_width: int
    image_height: int
    image_dynamic_range: int
    gpu_device: int
    cuda_visible_devices: str
    cuda_vulkan_mapping: dict[str, int]


def navigation_image_dynamic_range(image: Image.Image) -> int:
    extrema = image.convert("RGB").getextrema()
    return max(high - low for low, high in extrema)


def validate_navigation_image(image: Image.Image) -> Image.Image:
    rgb = image.convert("RGB")
    dynamic_range = navigation_image_dynamic_range(rgb)
    if dynamic_range == 0:
        raise RuntimeError(
            "navigation observation is a uniform image; AI2-THOR/Vulkan rendering is invalid"
        )
    return rgb


def _ai2thor_mapping_path() -> Path:
    home_root = os.environ.get("AI2THOR_HOME_ROOT")
    if home_root:
        return Path(home_root) / ".ai2thor" / "cuda-vulkan-mapping.json"
    return Path.home() / ".ai2thor" / "cuda-vulkan-mapping.json"


def probe_navigation_render(
    controller_factory: Callable[..., Any],
    *,
    scene: str = "FloorPlan1",
    gpu_device: int = 0,
) -> DirectRenderProbeResult:
    if gpu_device < 0:
        raise ValueError("gpu_device must be non-negative")

    started_at = time.monotonic()
    controller = controller_factory(
        agentMode="default",
        gridSize=0.1,
        visibilityDistance=10,
        renderDepthImage=False,
        renderInstanceSegmentation=False,
        width=255,
        height=255,
        fieldOfView=100,
        gpu_device=gpu_device,
        server_timeout=60,
        server_start_timeout=60,
    )
    try:
        event = controller.reset(scene=scene)
        frame = getattr(event, "frame", None)
        if frame is None:
            raise RuntimeError("AI2-THOR render probe returned no frame")
        image = validate_navigation_image(Image.fromarray(frame))
    finally:
        controller.stop()

    mapping_path = _ai2thor_mapping_path()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError(f"invalid CUDA/Vulkan mapping: {mapping!r}")
    if str(gpu_device) not in mapping:
        raise RuntimeError(f"CUDA/Vulkan mapping has no GPU ordinal {gpu_device}: {mapping!r}")

    return DirectRenderProbeResult(
        scene=scene,
        elapsed_seconds=round(time.monotonic() - started_at, 3),
        image_width=image.width,
        image_height=image.height,
        image_dynamic_range=navigation_image_dynamic_range(image),
        gpu_device=gpu_device,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        cuda_vulkan_mapping={str(key): int(value) for key, value in mapping.items()},
    )


def main() -> int:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    parser = argparse.ArgumentParser(description="Validate one allocated AI2-THOR GPU ordinal")
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--scene", default="FloorPlan1")
    args = parser.parse_args()

    result = probe_navigation_render(
        lambda **kwargs: Controller(platform=CloudRendering, **kwargs),
        scene=args.scene,
        gpu_device=args.gpu_device,
    )
    print(json.dumps({"status": "AI2THOR_RENDER_OK", **asdict(result)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DirectRenderProbeResult",
    "navigation_image_dynamic_range",
    "probe_navigation_render",
    "validate_navigation_image",
]
