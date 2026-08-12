"""Navigation environment exports loaded lazily to keep format utilities light."""

from __future__ import annotations

from typing import Any

__all__ = ["NavigationEnv", "NavigationEnvConfig", "NavigationHandler"]


def __getattr__(name: str) -> Any:
    if name in {"NavigationEnv", "NavigationEnvConfig"}:
        from .navigation_env import NavigationEnv, NavigationEnvConfig

        return {"NavigationEnv": NavigationEnv, "NavigationEnvConfig": NavigationEnvConfig}[name]
    if name == "NavigationHandler":
        from .handler import NavigationHandler

        return NavigationHandler
    raise AttributeError(name)
