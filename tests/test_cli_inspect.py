import json
import io
import sys
from pathlib import Path

from llm_ripper.cli import create_parser, inspect_command


def test_cli_inspect_prints_json(tmp_path: Path):
    kb = tmp_path / "kb"
    (kb / "embeddings").mkdir(parents=True)
    (kb / "heads" / "layer_0").mkdir(parents=True)
    (kb / "ffns" / "layer_0").mkdir(parents=True)
    (kb / "lm_head").mkdir(parents=True)
    (kb / "extraction_metadata.json").write_text(json.dumps({"source_model": "dummy"}))

    parser = create_parser()
    ns = parser.parse_args(["inspect", "--knowledge-bank", str(kb)])
    assert ns.command == "inspect"

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        inspect_command(ns)
    finally:
        sys.stdout = old
    out = buf.getvalue()
    data = json.loads(out)
    assert data["knowledge_bank"].endswith("kb")
    assert data["components"]["embeddings"] is True
