__all__ = [
    "NavigationEnv",
    "NavigationEnvConfig",
    "NavigationServiceConfig",
    "NavigationService",
]


def __getattr__(name):
    if name == "NavigationEnv":
        from .env import NavigationEnv

        return NavigationEnv
    if name == "NavigationEnvConfig":
        from .env_config import NavigationEnvConfig

        return NavigationEnvConfig
    if name == "NavigationServiceConfig":
        from .service_config import NavigationServiceConfig

        return NavigationServiceConfig
    if name == "NavigationService":
        from .service import NavigationService

        return NavigationService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
