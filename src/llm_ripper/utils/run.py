"""
Run management utilities: standardized directories and artifact writers.

Creates a stamped run directory under `runs/<stamp>/` with the structure
requested in the implementation plan and exposes helpers to save JSON/JSONL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


SUBDIRS = (
    "knowledge_bank",
    "activations",
    "analysis",
    "transplants",
    "validation",
    "causal",
    "traces",
    "counterfactuals",
    "uq",
    "catalog",
    "provenance",
    "reports",
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


@dataclass
class RunContext:
    root: Path
    stamp: str = field(default_factory=_now_stamp)

    @classmethod
    def create(
        cls, base: str | Path = "runs", stamp: Optional[str] = None
    ) -> "RunContext":
        base = Path(base)
        if stamp is None:
            stamp = _now_stamp()
        root = base / stamp
        root.mkdir(parents=True, exist_ok=True)
        for sd in SUBDIRS:
            (root / sd).mkdir(parents=True, exist_ok=True)
        return cls(root=root, stamp=stamp)

    # Directory getters
    def dir(self, name: str) -> Path:
        p = self.root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    def traces_dir(self) -> Path:
        return self.dir("traces")

    def cf_dir(self) -> Path:
        return self.dir("counterfactuals")

    def uq_dir(self) -> Path:
        return self.dir("uq")

    def catalog_dir(self) -> Path:
        return self.dir("catalog")

    def prov_dir(self) -> Path:
        return self.dir("provenance")

    def reports_dir(self) -> Path:
        return self.dir("reports")

    # Writers
    def write_json(self, relpath: str | Path, data: Dict[str, Any]) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def write_jsonl(self, relpath: str | Path, rows: Iterable[Dict[str, Any]]) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str))
                f.write("\n")
        tmp.replace(path)
        return path
