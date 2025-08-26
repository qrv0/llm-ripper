#!/usr/bin/env python3
import time
import csv
import tracemalloc
from pathlib import Path

from llm_ripper.utils.config import ConfigManager
from llm_ripper.core import KnowledgeExtractor, ActivationCapture, KnowledgeAnalyzer


def _ts():
    return time.perf_counter()


def measure(func, *args, **kwargs):
    tracemalloc.start()
    t0 = _ts()
    res = func(*args, **kwargs)
    t1 = _ts()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return res, {"secs": t1 - t0, "peak_mb": peak / (1024 * 1024)}


def main():
    cfg = ConfigManager(None)
    out_dir = Path("./output/perf")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # Minimal offline run using fixtures/assumptions
    # In real usage, replace with actual model/paths
    kb = "./knowledge_bank"
    activ = "./activations.h5"

    extractor = KnowledgeExtractor(cfg)
    try:
        _, m = measure(
            extractor.extract_model_components,
            model_name="sshleifer/tiny-gpt2",
            output_dir=kb,
            components=["embeddings"],
        )  # example
        rows.append(["extract", m["secs"], m["peak_mb"]])
    except Exception:
        pass

    capture = ActivationCapture(cfg)
    try:
        from llm_ripper.utils.data_manager import DataManager

        ds = DataManager(cfg).load_probing_corpus("small")
        _, m = measure(
            capture.capture_model_activations,
            model_name="sshleifer/tiny-gpt2",
            corpus_dataset=ds,
            output_file=activ,
            max_samples=8,
        )
        rows.append(["capture", m["secs"], m["peak_mb"]])
    except Exception:
        pass

    analyzer = KnowledgeAnalyzer(cfg)
    try:
        _, m = measure(
            analyzer.analyze_knowledge_bank,
            knowledge_bank_dir=kb,
            activations_file=None,
            output_dir="./analysis",
        )
        rows.append(["analyze", m["secs"], m["peak_mb"]])
    except Exception:
        pass

    with open(out_dir / "perf_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "secs", "peak_mb"])
        w.writerows(rows)
    print("Wrote:", out_dir / "perf_report.csv")


if __name__ == "__main__":
    main()
