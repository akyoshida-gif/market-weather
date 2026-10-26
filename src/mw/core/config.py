"""Configuration loading and validation (pydantic). Secrets come only from env vars."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS = REPO_ROOT / "config" / "settings.yaml"


class GuardSettings(BaseModel):
    soft_limit_bytes: int = 8_000_000_000
    soft_limit_class_a_per_month: int = 800_000
    soft_limit_class_b_per_month: int = 8_000_000
    exempt_prefixes: list[str] = Field(default_factory=lambda: ["ledger/", "registry/"])


class R2Settings(BaseModel):
    bucket: str = "market-weather-actions"


class StorageSettings(BaseModel):
    backend: Literal["local", "r2"] = "local"
    local_root: str = "./data"
    r2: R2Settings = Field(default_factory=R2Settings)
    guard: GuardSettings = Field(default_factory=GuardSettings)


class MarketSettings(BaseModel):
    calendar: str
    timezone: str
    benchmark: str


class Settings(BaseModel):
    storage: StorageSettings = Field(default_factory=StorageSettings)
    markets: dict[str, MarketSettings] = Field(default_factory=dict)
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 20, 60])


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines into os.environ (existing vars win). Values are never logged."""
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def load_settings(path: Path | None = None) -> Settings:
    """Load settings.yaml; env MW_STORAGE_BACKEND / MW_LOCAL_ROOT override storage."""
    env_file = os.environ.get("MW_ENV_FILE")
    if env_file:
        load_env_file(Path(env_file))
    p = path or Path(os.environ.get("MW_SETTINGS", DEFAULT_SETTINGS))
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    settings = Settings.model_validate(raw)
    backend = os.environ.get("MW_STORAGE_BACKEND")
    if backend:
        settings.storage.backend = backend  # type: ignore[assignment]
        settings = Settings.model_validate(settings.model_dump())
    local_root = os.environ.get("MW_LOCAL_ROOT")
    if local_root:
        settings.storage.local_root = local_root
    return settings
