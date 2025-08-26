import json
from pathlib import Path
import torch
import torch.nn as nn

from llm_ripper.core.extraction import KnowledgeExtractor
from llm_ripper.utils.config import ConfigManager


class _Expert(nn.Module):
    def __init__(self, h=8, inter=16):
        super().__init__()
        self.fc1 = nn.Linear(h, inter, bias=False)  # gate_proj alias
        self.fc2 = nn.Linear(h, inter, bias=False)  # up_proj alias
        self.fc3 = nn.Linear(inter, h, bias=False)  # down_proj alias


class _MoE_A(nn.Module):
    def __init__(self, h=8, inter=16, n=2):
        super().__init__()
        self.experts_list = nn.ModuleList([_Expert(h, inter) for _ in range(n)])
        self.gating_network = nn.Linear(h, n, bias=True)
        self.top_k = 1
        self.capacity_factor = 1.5
        self.router_temperature = 0.8


def test_moe_variant_extraction_wide_search(tmp_path: Path):
    cfg = ConfigManager()
    ke = KnowledgeExtractor(cfg)
    moe = _MoE_A()
    out_dir = tmp_path / "ffn_layer"
    out_dir.mkdir(parents=True)
    res = ke._extract_moe_ffn_layer(moe, layer_idx=0, layer_dir=out_dir)
    # config written
    cfg_json = json.loads((out_dir / "config.json").read_text())
    assert cfg_json["ffn_type"] == "moe"
    assert cfg_json["num_experts"] == 2
    assert "router" in cfg_json
    # experts saved
    assert (out_dir / "expert_0" / "gate_proj.pt").exists() or (
        out_dir / "expert_0" / "gate_proj.safetensors"
    ).exists()
    assert "experts" in res and "expert_0" in res["experts"]
