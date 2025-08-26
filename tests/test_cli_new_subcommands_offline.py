import io
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from llm_ripper.cli import create_parser


class TestNewCLISubcommandsOffline(unittest.TestCase):
    def run_cmd(self, argv):
        parser = create_parser()
        ns = parser.parse_args(argv)
        buf = io.StringIO()
        old = sys.stdout
        sys.stdout = buf
        try:
            ns.func(ns)
        finally:
            sys.stdout = old
        out = buf.getvalue()
        try:
            data = json.loads(out)
        except Exception:
            data = {}
        return out, data

    def test_trace_json(self):
        with patch("llm_ripper.cli.Tracer") as Tr:
            inst = MagicMock()

            class _Res:
                run_id = "r"
                summary_path = "s.json"
                traces_path = "t.jsonl"

            inst.run.return_value = _Res()
            Tr.return_value = inst
            out, data = self.run_cmd(
                ["trace", "--model", "m", "--targets", "head:0.q", "--json"]
            )
            assert data.get("run_id") == "r"

    def test_cfgen_and_cfeval(self, tmp_path: Path = None):
        td = Path("./output_test")
        td.mkdir(exist_ok=True)
        pairs = td / "pairs.jsonl"
        out, _ = self.run_cmd(
            ["cfgen", "--task", "agreement", "--n", "5", "--out", str(pairs)]
        )
        assert pairs.exists()
        with patch("llm_ripper.cli.CounterfactualEvaluator") as Ev:
            inst = MagicMock()
            inst.evaluate.return_value = {
                "results_file": str(td / "res.jsonl"),
                "summary": {"avg_delta": 0.0, "pairs": 5},
            }
            Ev.return_value = inst
            out, data = self.run_cmd(
                [
                    "cfeval",
                    "--model",
                    "m",
                    "--pairs",
                    str(pairs),
                    "--out",
                    str(td / "res.jsonl"),
                    "--json",
                ]
            )
            assert data.get("results_file")

    def test_uq_json(self):
        with patch("llm_ripper.cli.UQRunner") as UQ:
            inst = MagicMock()
            inst.run.return_value = {"run_id": "r", "summary_file": "uq/summary.json"}
            UQ.return_value = inst
            out, data = self.run_cmd(["uq", "--model", "m", "--json"])
            assert data.get("summary_file")

    def test_align_features_provenance(self):
        with patch("llm_ripper.cli.orthogonal_procrustes_align") as Align:
            Align.return_value = {
                "matrix_file": "W.npy",
                "cosine_before": 0.1,
                "cosine_after": 0.2,
            }
            out, _ = self.run_cmd(
                ["bridge-align", "--source", "kb", "--target", "m", "--out", "W"]
            )
        with patch("llm_ripper.cli.discover_features") as Feat:
            Feat.return_value = {"catalog_file": "catalog/heads.json", "features": 4}
            out, _ = self.run_cmd(
                ["features", "--activations", "a.h5", "--out", "catalog"]
            )
        with patch("llm_ripper.cli.ProvenanceScanner") as PS:
            inst = MagicMock()
            inst.scan.return_value = {"ok": True, "violations": []}
            PS.return_value = inst
            out, _ = self.run_cmd(["provenance", "--scan", "./transplanted", "--json"])

    def test_route_sim(self, tmp_path: Path = None):
        td = Path("./output_test")
        td.mkdir(exist_ok=True)
        metrics = td / "metrics.jsonl"
        metrics.write_text(
            "\n".join([json.dumps({"confidence": c}) for c in [0.2, 0.8, 0.5]])
        )
        out, data = self.run_cmd(
            ["route-sim", "--metrics", str(metrics), "--tau", "0.7", "--json"]
        )
        assert data.get("routed") == 2

    def test_merge_adapters_tokenize_report_stress(self):
        # create dummy spec file to satisfy path access if import path changes
        Path("spec.json").write_text('{"models": ["m1", "m2"]}')
        with patch("llm_ripper.interop.merge_models_average") as M:
            M.return_value = {"merged_keys": 10, "out": "out/pytorch_model.bin"}
            out, _ = self.run_cmd(["merge", "--global", "spec.json", "--out", "out"])
        with patch("llm_ripper.interop.import_lora_and_inject") as Imp, patch(
            "llm_ripper.interop.fuse_layer_adapters"
        ) as Fuse:
            Imp.return_value = {"layer": 0, "in": 8, "out": 8}
            Fuse.return_value = {"layer": 0, "adapters": 1}
            out, _ = self.run_cmd(
                [
                    "adapters",
                    "--model",
                    "./m",
                    "--import",
                    "lora.safetensors",
                    "--layer",
                    "0",
                    "--fuse",
                    "--json",
                ]
            )
        with patch("llm_ripper.interop.align_tokenizers") as Tok:
            Tok.return_value = {"overlap": 100, "mapping_file": "mapping.json"}
            out, _ = self.run_cmd(
                [
                    "tokenize-align",
                    "--source",
                    "A",
                    "--target",
                    "B",
                    "--out",
                    "mapping.json",
                ]
            )
        with patch("llm_ripper.safety.report.generate_report") as Rep:
            Rep.return_value = {"json": "reports/report.json"}
            out, _ = self.run_cmd(["report", "--out", "reports", "--ideal", "--json"])
        with patch("llm_ripper.safety.stress.run_stress_and_drift") as Str:
            Str.return_value = {"psi_entropy": 0.01}
            out, _ = self.run_cmd(
                [
                    "stress",
                    "--model",
                    "m",
                    "--baseline",
                    "b",
                    "--out",
                    "reports",
                    "--json",
                ]
            )


if __name__ == "__main__":
    unittest.main()
