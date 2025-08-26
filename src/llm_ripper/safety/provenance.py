from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ScanConfig:
    root: str
    fail_on_violation: bool = False


class ProvenanceScanner:
    """Scan transplanted artifacts for provenance and license compliance hints."""

    def scan(self, cfg: ScanConfig) -> Dict[str, Any]:
        root = Path(cfg.root)
        report: Dict[str, Any] = {"root": str(root), "violations": [], "artifacts": {}}
        if not root.exists():
            report["violations"].append("root_missing")
            return report
        # Transplant metadata
        meta = root / "transplant_metadata.json"
        if not meta.exists():
            report["violations"].append("missing_transplant_metadata")
        else:
            try:
                data = json.loads(meta.read_text())
                report["artifacts"]["transplant_metadata"] = {
                    "file": str(meta),
                    "keys": list(data.keys()),
                }
            except Exception:
                report["violations"].append("invalid_transplant_metadata")
        # Model and tokenizer
        model_dir = root / "model"
        tok_dir = root / "tokenizer"
        for p in (model_dir, tok_dir):
            if not p.exists():
                report["violations"].append(f"missing_{p.name}")
            else:
                files = [str(x) for x in p.glob("*") if x.is_file()]
                report["artifacts"][p.name] = {"files": files}
        # Licenses present?
        for name in ("LICENSE", "LICENSE.txt", "NOTICE"):
            fp = root / name
            if fp.exists():
                report.setdefault("licenses", []).append(str(fp))
        if "licenses" not in report:
            report["violations"].append("missing_license_files")
        # Hash important files
        hashes: List[Dict[str, Any]] = []
        for p in (meta, model_dir / "config.json", tok_dir / "tokenizer.json"):
            if p.exists():
                try:
                    hashes.append({"file": str(p), "sha256": _sha256_file(p)})
                except Exception:
                    pass
        report["hashes"] = hashes
        report["ok"] = len(report["violations"]) == 0
        if cfg.fail_on_violation and not report["ok"]:
            raise RuntimeError(f"Provenance violations: {report['violations']}")
        return report
