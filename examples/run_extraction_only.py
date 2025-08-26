#!/usr/bin/env python3
"""
Extraction-only script to build a knowledge bank from a donor model.

Reads environment variables and/or examples/config.json.

Useful variables:
  - DONOR_MODEL_NAME: donor model path/ID (e.g., models/gpt-oss-20b)
  - KNOWLEDGE_BANK_DIR: directory to save extracted artifacts (e.g., ./knowledge_bank/gpt-oss-20b)
  - EXTRACT_COMPONENTS: comma-separated list (embeddings,attention_heads,ffn_layers,lm_head)
  - LOAD_IN_8BIT / LOAD_IN_4BIT: true/false (requires bitsandbytes)
  - TRUST_REMOTE_CODE: true/false
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_ripper.utils.config import ConfigManager
from llm_ripper.core.extraction import KnowledgeExtractor


def main():
    cfg = ConfigManager("examples/config.json")
    # Environment overrides (no placeholders)
    donor = os.getenv("DONOR_MODEL_NAME") or cfg.get("donor_model_name")
    if not donor:
        raise SystemExit(
            "DONOR_MODEL_NAME not set. Define env var or examples/config.json."
        )
    kb = os.getenv("KNOWLEDGE_BANK_DIR") or cfg.get("knowledge_bank_dir")
    if not kb:
        kb = "./knowledge_bank"
    # If user passed a local path, create a nominal subfolder
    donor_tag = Path(donor).name.replace("/", "_")
    out_dir = str(Path(kb) / donor_tag)

    components_env = os.getenv(
        "EXTRACT_COMPONENTS", "embeddings,attention_heads,ffn_layers,lm_head"
    )
    components = [c.strip() for c in components_env.split(",") if c.strip()]

    # Honor quantization/trust_remote_code flags
    cfg.update(
        {
            "donor_model_name": donor,
            "knowledge_bank_dir": out_dir,
            "load_in_8bit": os.getenv("LOAD_IN_8BIT", "false").lower() == "true",
            "load_in_4bit": os.getenv("LOAD_IN_4BIT", "false").lower() == "true",
            "trust_remote_code": os.getenv("TRUST_REMOTE_CODE", "false").lower()
            == "true",
        }
    )

    cfg.create_directories()
    extractor = KnowledgeExtractor(cfg)

    print(f"[LLM-Ripper] Extract donor={donor} -> {out_dir}")
    res = extractor.extract_model_components(
        model_name=donor,
        output_dir=out_dir,
        components=components,
    )
    print("[OK] Extracted components:", ", ".join(res["extracted_components"].keys()))
    print("Saved files in:", out_dir)


if __name__ == "__main__":
    main()
