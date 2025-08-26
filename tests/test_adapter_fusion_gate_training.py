import torch
import torch.nn as nn
from pathlib import Path
import json

from llm_ripper.core.transplant import KnowledgeTransplanter, TransplantedModule
from llm_ripper.utils.config import ConfigManager


class _Block(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.mlp.weight)

    def forward(self, x):
        return self.mlp(x)


class _Model(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()

        class Cfg:
            model_type = "gpt2"
            hidden_size = hidden

        self.config = Cfg()
        self.transformer = type("T", (), {})()
        self.transformer.h = nn.ModuleList([_Block(hidden)])

    def forward(self, x):
        return self.transformer.h[0](x)


def test_adapter_fusion_gate_training_and_reattach(tmp_path: Path):
    cfg = ConfigManager()
    kt = KnowledgeTransplanter(cfg)
    m = _Model(hidden=8)

    # donor module: simple linear doubling output
    class Donor(nn.Module):
        def __init__(self, h=8):
            super().__init__()
            self.down_proj = nn.Linear(h, h, bias=False)
            nn.init.constant_(self.down_proj.weight, 2.0)

        def forward(self, x):
            return self.down_proj(x)

    donor = Donor(8)
    tm = TransplantedModule(donor, None, None, freeze_donor=True)

    # Inject, which will install a fusion gate
    kt._inject_module(m, tm, target_layer=0)
    assert hasattr(m, "transplant_fusion_gates")
    gate = m.transplant_fusion_gates["layer_0"]
    init = float(torch.sigmoid(gate.alpha).detach().cpu())

    # Train gate alpha synthetically
    report = kt.fine_tune_fusion_gates(m, steps=20, lr=5e-2, target_gate=0.2)
    assert report["updated"] >= 1
    final = float(
        torch.sigmoid(m.transplant_fusion_gates["layer_0"].alpha).detach().cpu()
    )
    assert abs(final - init) > 1e-4

    # Save artifacts json + state
    model_dir = tmp_path / "model"
    art_dir = model_dir / "transplant_artifacts"
    art_dir.mkdir(parents=True)
    gate_file = art_dir / "layer_0_fusion_gate.pt"
    torch.save(m.transplant_fusion_gates["layer_0"].state_dict(), gate_file)
    # json alongside model_dir
    (model_dir / "transplant_artifacts.json").write_text(
        json.dumps({"layers": {"layer_0": {"fusion_gate": str(gate_file)}}})
    )

    # Reattach gate on a fresh model
    m2 = _Model(hidden=8)
    rep = kt.load_transplant_artifacts(m2, str(art_dir))
    assert "attached_gates" in rep and "layer_0" in rep["attached_gates"]
