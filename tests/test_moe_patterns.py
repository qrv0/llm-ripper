import json
from pathlib import Path
import torch
import torch.nn as nn

from llm_ripper.core.extraction import KnowledgeExtractor
from llm_ripper.utils.config import ConfigManager


class Expert(nn.Module):
    def __init__(self, h=8, inter=12):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class MoEContainerA(nn.Module):
    def __init__(self, n=2, h=8, inter=12):
        super().__init__()
        self.experts = nn.ModuleList([Expert(h, inter) for _ in range(n)])
        self.router = nn.Linear(h, n, bias=True)

    def forward(self, x):
        return x


class MoEContainerB(nn.Module):
    def __init__(self, n=3, h=8, inter=12):
        super().__init__()

        class Wrap(nn.Module):
            def __init__(self, n, h, inter):
                super().__init__()
                self.experts_list = nn.ModuleList([Expert(h, inter) for _ in range(n)])

        self.block_sparse_moe = Wrap(n, h, inter)
        self.gating = nn.Linear(h, n, bias=False)

    def forward(self, x):
        return x


def test_moe_extraction_variants(tmp_path: Path):
    cfg = ConfigManager(None)
    ke = KnowledgeExtractor(cfg)
    # Use private MoE extract on two container variants
    for idx, moe in enumerate([MoEContainerA(), MoEContainerB()]):
        layer_dir = tmp_path / f"layer_{idx}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        out = ke._extract_moe_ffn_layer(moe, idx, layer_dir)
        conf = json.loads((layer_dir / "config.json").read_text())
        assert conf.get("ffn_type") == "moe"
        assert "experts" in out and isinstance(out["experts"], dict)
        # Expect at least 2 experts saved
        assert len([k for k in out["experts"].keys() if k.startswith("expert_")]) >= 2
        # Router is optional but should save if present
        if hasattr(moe, "router") or hasattr(moe, "gating"):
            # files saved under router/ if linear exists
            rp = layer_dir / "router"
            assert rp.exists()
