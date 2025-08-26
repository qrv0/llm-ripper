"""
End-to-end pipeline example for LLM Ripper.

Requirements:
- Provide a real model path or HF model id via --model (and optional --baseline).
- The script runs offline-safe tasks where possible; some steps need a real model.

Usage:
  python examples/run_full_pipeline.py --model gpt2 --baseline gpt2

Outputs are written under runs/<stamp>/ according to the standard layout.
"""

from __future__ import annotations

import argparse
import json

from llm_ripper.utils.config import ConfigManager
from llm_ripper.utils.run import RunContext
from llm_ripper.utils.model_loader import ModelLoader
from llm_ripper.core import (
    KnowledgeExtractor,
    KnowledgeAnalyzer,
    KnowledgeTransplanter,
    ValidationSuite,
)
from llm_ripper.core.transplant import TransplantConfig
from llm_ripper.causal import Tracer, TraceConfig
from llm_ripper.bridge import orthogonal_procrustes_align
from llm_ripper.uq import UQRunner
from llm_ripper.counterfactual import generate_minimal_pairs, CounterfactualEvaluator
from llm_ripper.safety.stress import run_stress_and_drift
from llm_ripper.safety.report import generate_report


def pick_safe_targets(model) -> list[str]:
    cfg = getattr(model, "config", None)
    L = int(getattr(cfg, "num_hidden_layers", 1) or 1)
    # Use first attention projection and first FFN part
    targets = ["head:0.q", "ffn:0.up"]
    if L > 1:
        targets.append("head:1.o")
    return targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = ConfigManager()
    cfg.set("device", args.device)
    cfg.set("seed", args.seed)

    rc = RunContext.create(base=args.out)
    print(f"[pipeline] run_dir={rc.root}")

    # Load model once to probe architecture/targets
    loader = ModelLoader(cache_dir=cfg.get("model_cache_dir"), device=cfg.get("device"))
    model, tok, mcfg = loader.load_model_and_tokenizer(
        args.model,
        model_type="causal_lm",
        trust_remote_code=cfg.get("trust_remote_code"),
    )

    # 1) Extract: embeddings only (lightweight)
    extractor = KnowledgeExtractor(cfg)
    kb_dir = rc.dir("knowledge_bank")
    extractor.extract_model_components(
        args.model, str(kb_dir), components=["embeddings"]
    )

    # 2) Analyze (light)
    analyzer = KnowledgeAnalyzer(cfg)
    analyzer.analyze_knowledge_bank(
        str(kb_dir), activations_file=None, output_dir=str(rc.dir("analysis"))
    )

    # 3) Causal tracing on a few safe targets
    tracer = Tracer(cfg)
    targets = pick_safe_targets(model)
    tcfg = TraceConfig(
        model=args.model,
        targets=targets,
        metric="nll_delta",
        dataset="diverse",
        intervention="zero",
        seed=args.seed,
        max_samples=32,
    )
    tracer.run(tcfg, out_dir=str(rc.root))

    # 4) Alignment (embeddings)
    Wres = orthogonal_procrustes_align(
        cfg, str(kb_dir), args.model, str(rc.dir("transplants") / "W_align")
    )
    (rc.dir("transplants") / "align_report.json").write_text(json.dumps(Wres, indent=2))

    # 5) Transplant (embeddings init) into a copy of model under transplants/
    transplanter = KnowledgeTransplanter(cfg)
    tcfg2 = TransplantConfig(
        source_component="embeddings",
        target_layer=0,
        bridge_hidden_size=64,
        freeze_donor=True,
        freeze_target=False,
        strategy="embedding_init",
    )
    transplanter.transplant_knowledge(
        str(kb_dir), args.model, [tcfg2], str(rc.dir("transplants"))
    )
    transplanted_model_dir = str(rc.dir("transplants"))

    # 6) Mechanistic validation (offline) on transplanted model
    validator = ValidationSuite(cfg)
    validator.validate_transplanted_model(
        transplanted_model_dir,
        baseline_model_name=args.baseline or args.model,
        benchmarks=["mechanistic"],
        output_dir=str(rc.dir("validation")),
    )

    # 7) UQ on transplanted model
    uq = UQRunner(cfg)
    uq.run(
        uq_cfg=type(
            "UQCfg",
            (),
            {
                "model": transplanted_model_dir,
                "samples": 8,
                "max_texts": 32,
                "seed": args.seed,
            },
        )(),
        out_dir=str(rc.root),
    )

    # 8) Counterfactuals (generate and evaluate)
    pairs = rc.dir("counterfactuals") / "pairs_agreement.jsonl"
    generate_minimal_pairs("agreement", 200, str(pairs))
    ev = CounterfactualEvaluator(cfg)
    ev.evaluate(
        transplanted_model_dir,
        [str(pairs)],
        str(rc.dir("counterfactuals") / "results.jsonl"),
    )

    # 9) Stress & drift vs baseline
    run_stress_and_drift(
        cfg, transplanted_model_dir, args.baseline or args.model, str(rc.dir("reports"))
    )

    # 10) Report (JSON/MD/PDF)
    generate_report(True, str(rc.dir("reports")), transplanted_dir=str(rc.root))

    print("✓ End-to-end pipeline complete.")
    print(f"Artifacts: {rc.root}")


if __name__ == "__main__":
    main()
