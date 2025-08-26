import io
import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

from llm_ripper.cli import create_parser


class TestCLISubcommandsOffline(unittest.TestCase):
    def run_cmd(self, argv, patch_target, return_dict):
        parser = create_parser()
        ns = parser.parse_args(argv)
        # Patch the class used inside CLI module
        with patch(patch_target) as Cls:
            inst = MagicMock()
            # Decide method by command
            if ns.command == "extract":
                inst.extract_model_components.return_value = return_dict
            elif ns.command == "capture":
                inst.capture_model_activations.return_value = return_dict
            elif ns.command == "analyze":
                inst.analyze_knowledge_bank.return_value = return_dict
            elif ns.command == "transplant":
                inst.transplant_knowledge.return_value = return_dict
            elif ns.command == "validate":
                inst.validate_transplanted_model.return_value = return_dict
            Cls.return_value = inst

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
            return data

    def test_extract_json_offline(self):
        ret = {
            "source_model": "m",
            "extracted_components": {"embeddings": {}},
            "extraction_config": {},
        }
        data = self.run_cmd(
            ["extract", "--model", "m", "--output-dir", "out", "--json"],
            "llm_ripper.cli.KnowledgeExtractor",
            ret,
        )
        assert data.get("source_model") == "m"

    def test_capture_json_offline(self):
        ret = {"model_name": "m", "num_samples": 1, "output_file": "a.h5"}
        data = self.run_cmd(
            ["capture", "--model", "m", "--output-file", "a.h5", "--json", "--offline"],
            "llm_ripper.cli.ActivationCapture",
            ret,
        )
        assert data.get("output_file") == "a.h5"

    def test_analyze_json_offline(self):
        ret = {"source_model": "m", "component_analysis": {}, "head_catalog": []}
        data = self.run_cmd(
            ["analyze", "--knowledge-bank", "kb", "--output-dir", "out", "--json"],
            "llm_ripper.cli.KnowledgeAnalyzer",
            ret,
        )
        assert "component_analysis" in data

    def test_transplant_json_offline(self):
        ret = {
            "source_model": "donor",
            "target_model": "target",
            "transplanted_components": {},
        }
        data = self.run_cmd(
            [
                "transplant",
                "--source",
                "kb",
                "--target",
                "m",
                "--output-dir",
                "out",
                "--json",
            ],
            "llm_ripper.cli.KnowledgeTransplanter",
            ret,
        )
        assert data.get("target_model") == "target"

    def test_validate_json_offline(self):
        ret = {
            "model_path": "m",
            "summary": {"overall_score": 0.0, "recommendations": []},
        }
        data = self.run_cmd(
            ["validate", "--model", "m", "--output-dir", "out", "--json"],
            "llm_ripper.cli.ValidationSuite",
            ret,
        )
        assert "summary" in data


if __name__ == "__main__":
    unittest.main()
