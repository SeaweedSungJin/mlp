"""Runtime configuration for pipeline controls (logging, VLM, etc.)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RuntimeConfig:
    """General-purpose runtime toggles for evaluation scripts."""

    log_samples: bool = False
    samples_dir: str = "runs/samples"
    samples_prefix: str = "row"
    samples_max_sections: int = 5
    samples_pretty_json: bool = True
    vlm_max_new_tokens: int = 128

    extra: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}
        self.samples_max_sections = max(1, int(self.samples_max_sections))
        self.vlm_max_new_tokens = max(8, int(self.vlm_max_new_tokens))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RuntimeConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Runtime config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        known_keys = {field.name for field in cls.__dataclass_fields__.values() if field.init}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            if key in known_keys:
                kwargs[key] = value
            else:
                extra[key] = value
        cfg = cls(**kwargs)
        cfg.extra = extra

        samples_dir = Path(cfg.samples_dir)
        if not samples_dir.is_absolute():
            samples_dir = (cfg_path.parent / samples_dir).resolve()
        cfg.samples_dir = str(samples_dir)
        return cfg

    @classmethod
    def default(cls) -> "RuntimeConfig":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "log_samples": self.log_samples,
            "samples_dir": self.samples_dir,
            "samples_prefix": self.samples_prefix,
            "samples_max_sections": self.samples_max_sections,
            "samples_pretty_json": self.samples_pretty_json,
            "vlm_max_new_tokens": self.vlm_max_new_tokens,
        }
        payload.update(self.extra)
        return payload
