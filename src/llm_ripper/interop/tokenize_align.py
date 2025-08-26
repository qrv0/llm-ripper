from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from transformers import AutoTokenizer


def align_tokenizers(source_tok: str, target_tok: str, out_path: str) -> Dict[str, Any]:
    src = AutoTokenizer.from_pretrained(source_tok)
    tgt = AutoTokenizer.from_pretrained(target_tok)
    common = set(src.get_vocab().keys()) & set(tgt.get_vocab().keys())
    # Build id mapping for common tokens
    mapping = {}
    for tok in list(common)[:100000]:  # safety cap
        try:
            mapping[tok] = {
                "src_id": int(src.get_vocab()[tok]),
                "tgt_id": int(tgt.get_vocab()[tok]),
            }
        except Exception:
            continue
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"overlap": len(common), "mapping": mapping}, indent=2))
    return {"overlap": len(common), "mapping_file": str(p)}
