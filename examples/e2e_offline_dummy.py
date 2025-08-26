"""
End-to-end offline dummy pipeline demonstrating extraction-like saving, transplant injection,
and validation summary without requiring external model downloads.

This script:
 1) Creates a tiny synthetic knowledge bank with a FFN component for layer_0
 2) Uses KnowledgeTransplanter internals to load the component, create a donor module, and inject it into a dummy target model
 3) Produces a minimal validation summary offline

Run:
  python examples/e2e_offline_dummy.py
"""

from pathlib import Path
import json
import torch
import torch.nn as nn

from llm_ripper.core.transplant import KnowledgeTransplanter, TransplantedModule
from llm_ripper.utils.config import ConfigManager


def create_synthetic_kb(root: Path, hidden=8, inter=16):
    kb = root / "kb"
    (kb / "ffns" / "layer_0").mkdir(parents=True, exist_ok=True)
    # Save simple FFN weights
    torch.save(torch.randn(inter, hidden), kb / "ffns" / "layer_0" / "gate_proj.pt")
    torch.save(torch.randn(inter, hidden), kb / "ffns" / "layer_0" / "up_proj.pt")
    torch.save(torch.randn(hidden, inter), kb / "ffns" / "layer_0" / "down_proj.pt")
    # Config
    (kb / "ffns" / "layer_0" / "config.json").write_text(
        json.dumps(
            {
                "layer_idx": 0,
                "activation_function": "relu",
                "hidden_size": hidden,
                "intermediate_size": inter,
                "ffn_type": "dense",
            }
        )
    )
    # Extraction metadata
    (kb / "extraction_metadata.json").write_text(
        json.dumps({"source_model": "dummy-offline"})
    )
    return kb


class _Block(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden, bias=False)
        nn.init.eye_(self.mlp.weight)

    def forward(self, x):
        return self.mlp(x)


class _DummyModel(nn.Module):
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


def main():
    out = Path("examples/_offline_out")
    kb = create_synthetic_kb(out)
    cfg = ConfigManager()
    kt = KnowledgeTransplanter(cfg)

    # Load component and create donor module
    donor = kt.build_donor_module_from_kb(str(kb), "layer_0_ffn")

    # Prepare target dummy model
    m = _DummyModel(hidden=8)
    tm = TransplantedModule(donor, None, None, freeze_donor=True)
    kt._inject_module(m, tm, target_layer=0)

    # Minimal validation-like summary (offline synthetic): overall score only
    summary = {
        "overall_score": 0.0,
        "component_scores": {},
        "recommendations": ["Offline dummy run completed"],
    }
    (out / "validation_results.json").write_text(
        json.dumps(
            {
                "model_path": "dummy",
                "baseline_model": None,
                "intrinsic_validation": {},
                "extrinsic_validation": {},
                "summary": summary,
            },
            indent=2,
        )
    )
    print(f"✓ Offline dummy pipeline completed. KB: {kb}, output: {out}")


if __name__ == "__main__":
    main()
