import torch
import torch.nn as nn

from llm_ripper.core.transplant import (
    KnowledgeTransplanter,
    TransplantedModule,
    BridgeNetwork,
)
from llm_ripper.utils.config import ConfigManager


class _Block(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        # base module returns a simple linear projection
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.mlp.weight)  # identity mapping

    def forward(self, x):
        return self.mlp(x)


class _Model(nn.Module):
    def __init__(self, layers=1, hidden=8):
        super().__init__()

        class Cfg:
            model_type = "gpt2"
            hidden_size = hidden

        self.config = Cfg()
        self.transformer = type("T", (), {})()
        self.transformer.h = nn.ModuleList([_Block(hidden) for _ in range(layers)])

    def forward(self, x):
        # pass through first block only (for this test)
        return self.transformer.h[0](x)


def test_inject_transplanted_module_replaces_or_hooks():
    cfg = ConfigManager()
    kt = KnowledgeTransplanter(cfg)
    model = _Model(layers=1, hidden=8)

    # donor: simple FFN-like module: gate_proj/up_proj/down_proj
    class DonorFFN(nn.Module):
        def __init__(self, h=8):
            super().__init__()
            self.gate_proj = nn.Linear(h, h, bias=False)
            self.up_proj = nn.Linear(h, h, bias=False)
            self.down_proj = nn.Linear(h, h, bias=False)
            nn.init.constant_(self.gate_proj.weight, 0.5)
            nn.init.constant_(self.up_proj.weight, 0.5)
            nn.init.constant_(self.down_proj.weight, 0.5)

        def forward(self, x):
            return self.down_proj(torch.relu(self.gate_proj(x)) * self.up_proj(x))

    donor = DonorFFN(8)
    tm = TransplantedModule(
        donor_module=donor, input_bridge=None, output_bridge=None, freeze_donor=True
    )

    # Pre-check: base output equals input due to identity weight
    x = torch.randn(2, 4, 8)
    base_out = model(x)

    # Inject
    kt._inject_module(model, tm, target_layer=0)

    # Expect registry of transplanted_modules
    assert hasattr(model, "transplanted_modules")
    assert "layer_0" in model.transplanted_modules

    # Forward now should be fused (not equal to base out for random x, almost surely)
    out = model(x)
    # Allow for rare equality; assert change in at least one element
    assert not torch.allclose(out, base_out)
