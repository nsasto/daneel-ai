from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


@dataclass
class ClientSettings:
    base_url: Optional[str]
    api_key: Optional[str]
    timeout: float = 5.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)


@dataclass
class AppSettings:
    memobase: ClientSettings
    ragdoll: ClientSettings
    graph: ClientSettings


def load_settings() -> AppSettings:
    return AppSettings(
        memobase=ClientSettings(
            base_url=_env("MEMOBASE_URL"),
            api_key=_env("MEMOBASE_API_KEY"),
            timeout=float(_env("MEMOBASE_TIMEOUT", "5.0")),
        ),
        ragdoll=ClientSettings(
            base_url=_env("RAGDOLL_URL"),
            api_key=_env("RAGDOLL_API_KEY"),
            timeout=float(_env("RAGDOLL_TIMEOUT", "5.0")),
        ),
        graph=ClientSettings(
            base_url=_env("GRAPH_RAG_URL"),
            api_key=_env("GRAPH_RAG_API_KEY"),
            timeout=float(_env("GRAPH_RAG_TIMEOUT", "5.0")),
        ),
    )
