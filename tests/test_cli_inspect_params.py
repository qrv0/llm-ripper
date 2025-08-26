import json
import io
import sys
from pathlib import Path

import torch

from llm_ripper.cli import create_parser, inspect_command


def write_tensor(path: Path, shape):
    t = torch.randn(*shape)
    torch.save(t, path)


def test_cli_inspect_param_counts(tmp_path: Path):
    kb = tmp_path / "kb"
    # embeddings
    (kb / "embeddings").mkdir(parents=True)
    (kb / "embeddings" / "config.json").write_text(
        json.dumps({"dimensions": [4, 3], "vocab_size": 4, "hidden_size": 3})
    )
    write_tensor(kb / "embeddings" / "embeddings.pt", (4, 3))

    # heads layer_0
    h0 = kb / "heads" / "layer_0"
    h0.mkdir(parents=True)
    (h0 / "config.json").write_text(
        json.dumps(
            {
                "attention_type": "MHA",
                "layer_idx": 0,
                "hidden_size": 8,
                "num_heads": 2,
                "head_dim": 4,
            }
        )
    )
    write_tensor(h0 / "q_proj.pt", (8, 8))
    write_tensor(h0 / "o_proj.pt", (8, 8))

    # ffns layer_0
    f0 = kb / "ffns" / "layer_0"
    f0.mkdir(parents=True)
    (f0 / "config.json").write_text(
        json.dumps(
            {
                "layer_idx": 0,
                "activation_function": "relu",
                "hidden_size": 8,
                "intermediate_size": 16,
                "ffn_type": "dense",
            }
        )
    )
    write_tensor(f0 / "gate_proj.pt", (16, 8))
    write_tensor(f0 / "up_proj.pt", (16, 8))
    write_tensor(f0 / "down_proj.pt", (8, 16))

    # metadata
    (kb / "extraction_metadata.json").write_text(json.dumps({"source_model": "dummy"}))

    parser = create_parser()
    ns = parser.parse_args(["inspect", "--knowledge-bank", str(kb)])

    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        inspect_command(ns)
    finally:
        sys.stdout = old
    data = json.loads(buf.getvalue())

    assert data["details"]["embeddings"]["param_count"] == 12
    h0_info = data["details"]["heads"]["layer_0"]
    # Only q_proj and o_proj present: 8*8 + 8*8
    assert h0_info.get("param_count", 0) == 128
    f0_info = data["details"]["ffns"]["layer_0"]
    # 16*8 + 16*8 + 8*16 = 384
    assert f0_info.get("param_count", 0) == 384
