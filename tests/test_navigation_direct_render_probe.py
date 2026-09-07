import json
import os
import sys
import unittest
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = REPO_ROOT / "vagen/utils/navigation_direct_render_probe.py"
PROBE_SPEC = importlib.util.spec_from_file_location("navigation_direct_render_probe", PROBE_PATH)
PROBE_MODULE = importlib.util.module_from_spec(PROBE_SPEC)
sys.modules[PROBE_SPEC.name] = PROBE_MODULE
PROBE_SPEC.loader.exec_module(PROBE_MODULE)

navigation_image_dynamic_range = PROBE_MODULE.navigation_image_dynamic_range
probe_navigation_render = PROBE_MODULE.probe_navigation_render
validate_navigation_image = PROBE_MODULE.validate_navigation_image


class _FakeEvent:
    def __init__(self, frame):
        self.frame = frame


class _FakeController:
    def __init__(self, frame):
        self.frame = frame
        self.kwargs = None
        self.reset_scene = None
        self.stopped = False

    def reset(self, scene):
        self.reset_scene = scene
        return _FakeEvent(self.frame)

    def stop(self):
        self.stopped = True


class NavigationDirectRenderProbeTest(unittest.TestCase):
    def test_dynamic_range_detects_uniform_and_nonuniform_images(self):
        uniform = Image.new("RGB", (2, 2), (7, 7, 7))
        nonuniform = Image.fromarray(
            np.array(
                [
                    [[0, 0, 0], [10, 0, 0]],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            )
        )

        self.assertEqual(0, navigation_image_dynamic_range(uniform))
        self.assertEqual(10, navigation_image_dynamic_range(nonuniform))

    def test_validate_navigation_image_rejects_uniform_frames(self):
        uniform = Image.new("RGB", (2, 2), (7, 7, 7))

        with self.assertRaisesRegex(RuntimeError, "uniform image"):
            validate_navigation_image(uniform)

    def test_probe_uses_requested_gpu_and_reports_mapping(self):
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        frame[0, 0, 0] = 42
        controllers = []

        def factory(**kwargs):
            controller = _FakeController(frame)
            controller.kwargs = kwargs
            controllers.append(controller)
            return controller

        with TemporaryDirectory() as tmpdir:
            mapping_dir = Path(tmpdir) / ".ai2thor"
            mapping_dir.mkdir()
            (mapping_dir / "cuda-vulkan-mapping.json").write_text(
                json.dumps({"2": 0}),
                encoding="utf-8",
            )
            old_home_root = os.environ.get("AI2THOR_HOME_ROOT")
            os.environ["AI2THOR_HOME_ROOT"] = tmpdir
            try:
                result = probe_navigation_render(factory, scene="FloorPlan2", gpu_device=2)
            finally:
                if old_home_root is None:
                    os.environ.pop("AI2THOR_HOME_ROOT", None)
                else:
                    os.environ["AI2THOR_HOME_ROOT"] = old_home_root

        self.assertEqual("FloorPlan2", result.scene)
        self.assertEqual(2, result.gpu_device)
        self.assertEqual({"2": 0}, result.cuda_vulkan_mapping)
        self.assertEqual(42, result.image_dynamic_range)
        self.assertEqual(2, controllers[0].kwargs["gpu_device"])
        self.assertEqual("FloorPlan2", controllers[0].reset_scene)
        self.assertTrue(controllers[0].stopped)

    def test_probe_rejects_mapping_without_requested_gpu(self):
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        frame[0, 0, 0] = 42

        with TemporaryDirectory() as tmpdir:
            mapping_dir = Path(tmpdir) / ".ai2thor"
            mapping_dir.mkdir()
            (mapping_dir / "cuda-vulkan-mapping.json").write_text(
                json.dumps({"0": 0}),
                encoding="utf-8",
            )
            old_home_root = os.environ.get("AI2THOR_HOME_ROOT")
            os.environ["AI2THOR_HOME_ROOT"] = tmpdir
            try:
                with self.assertRaisesRegex(RuntimeError, "no GPU ordinal 2"):
                    probe_navigation_render(lambda **kwargs: _FakeController(frame), gpu_device=2)
            finally:
                if old_home_root is None:
                    os.environ.pop("AI2THOR_HOME_ROOT", None)
                else:
                    os.environ["AI2THOR_HOME_ROOT"] = old_home_root


if __name__ == "__main__":
    unittest.main()
