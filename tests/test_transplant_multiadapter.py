import torch
import torch.nn as nn

from llm_ripper.core.transplant import KnowledgeTransplanter, TransplantedModule
from llm_ripper.utils.config import ConfigManager


class TinyLayer(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        # Provide common aliases to be found by injector
        self.mlp = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self, nlayers=1, hidden=8):
        super().__init__()

        class C:  # simple config shim
            model_type = "gpt2"
            hidden_size = hidden

        self.config = C()

        # Common layer access patterns: model.layers.0
        class Layers(nn.Module):
            def __init__(self, n, h):
                super().__init__()
                self.layers = nn.ModuleList([TinyLayer(h) for _ in range(n)])

            def __getattr__(self, name):
                if name == "layers":
                    return super().__getattr__(name)
                return super().__getattr__(name)

        self.model = Layers(nlayers, hidden)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return x


def test_multiadapter_modulelist_order():
    cfg = ConfigManager(None)
    kt = KnowledgeTransplanter(cfg)
    tm = TinyModel(nlayers=1, hidden=8)
    # Two donor modules with different internal dims; simple linear works
    donor1 = nn.Linear(8, 8, bias=False)
    donor2 = nn.Linear(8, 8, bias=False)
    m1 = TransplantedModule(donor1)
    m2 = TransplantedModule(donor2)
    # Inject twice into layer 0
    kt._inject_module(tm, m1, 0)
    kt._inject_module(tm, m2, 0)
    assert hasattr(tm, "transplanted_modules")
    lkey = "layer_0"
    assert lkey in tm.transplanted_modules
    mods = tm.transplanted_modules[lkey]
    # Should be ModuleList with two entries, preserving order
    assert isinstance(mods, (nn.ModuleList, nn.Module))
    if isinstance(mods, nn.ModuleList):
        assert len(mods) == 2
        assert mods[0] is m1
        assert mods[1] is m2
