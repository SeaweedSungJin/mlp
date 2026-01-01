"""Utilities for dumping per-sample results to JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class SampleLogger:
    def __init__(self, directory: str, prefix: str = "row", pretty: bool = True) -> None:
        self.dir = Path(directory).expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.pretty = pretty

    def log(self, sample_id: int, payload: Dict[str, Any]) -> Path:
        filename = f"{self.prefix}{sample_id}.json"
        target = self.dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2 if self.pretty else None)
        return target
