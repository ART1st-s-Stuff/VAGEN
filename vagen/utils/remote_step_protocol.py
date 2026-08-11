"""Strict decoding for environment step fields returned over HTTP."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def parse_remote_step_fields(data: Mapping[str, Any]) -> tuple[float, bool, dict[str, Any]]:
    """Validate transport types before normalizing a remote step result."""

    if "reward" not in data:
        raise ValueError("remote step response is missing reward")
    reward = data["reward"]
    if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
        raise ValueError(f"remote step reward must be a finite number, got {reward!r}")

    if "done" not in data:
        raise ValueError("remote step response is missing done")
    done = data["done"]
    if not isinstance(done, bool):
        raise ValueError(f"remote step done must be bool, got {done!r}")

    info = data.get("info")
    if not isinstance(info, Mapping):
        raise ValueError(f"remote step info must be a mapping, got {type(info).__name__}")

    return float(reward), done, dict(info)


__all__ = ["parse_remote_step_fields"]
