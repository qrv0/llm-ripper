from pathlib import Path
import json
import torch

from llm_ripper.utils.storage import load_sharded_pt


def test_load_sharded_pt(tmp_path: Path):
    # Create two shard files
    t0 = torch.randn(3, 4)
    t1 = torch.randn(2, 4)
    p0 = tmp_path / "w_part0.pt"
    p1 = tmp_path / "w_part1.pt"
    torch.save(t0, p0)
    torch.save(t1, p1)
    # Create index
    idx = tmp_path / "w.index.json"
    idx.write_text(json.dumps({"parts": [p0.name, p1.name], "sharded": True}))
    # Load
    t = load_sharded_pt(idx)
    assert t.shape == (5, 4)
    assert torch.allclose(t[:3], t0)
