import json
import tempfile
import unittest
from pathlib import Path
import importlib.util
import types
import sys


def load_config_manager():
    """Load ConfigManager class directly from file without importing package."""
    src_path = (
        Path(__file__).parent.parent / "src" / "llm_ripper" / "utils" / "config.py"
    )
    # Provide a dummy dotenv if not available
    if "dotenv" not in sys.modules:
        dummy = types.ModuleType("dotenv")

        def load_dotenv(*args, **kwargs):
            return None

        dummy.load_dotenv = load_dotenv  # type: ignore
        sys.modules["dotenv"] = dummy
    spec = importlib.util.spec_from_file_location("config_module", src_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore
    return module.ConfigManager


class TestConfigValidation(unittest.TestCase):
    def test_config_validation_requires_model_names(self):
        ConfigManager = load_config_manager()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            # Missing model names should fail
            cfg_path.write_text(
                json.dumps(
                    {
                        "knowledge_bank_dir": str(Path(td) / "kb"),
                        "output_dir": str(Path(td) / "out"),
                    }
                )
            )
            cm = ConfigManager(str(cfg_path))
            with self.assertRaises(ValueError) as ctx:
                cm.validate_config()
            self.assertTrue(
                "DONOR_MODEL_NAME" in str(ctx.exception)
                or "TARGET_MODEL_NAME" in str(ctx.exception)
            )

    def test_config_validation_passes_with_names(self):
        ConfigManager = load_config_manager()
        with tempfile.TemporaryDirectory() as td:
            cfg_path = Path(td) / "config.json"
            cfg_path.write_text(
                json.dumps(
                    {
                        "donor_model_name": "dummy-donor",
                        "target_model_name": "dummy-target",
                        "knowledge_bank_dir": str(Path(td) / "kb"),
                        "output_dir": str(Path(td) / "out"),
                    }
                )
            )
            cm = ConfigManager(str(cfg_path))
            self.assertTrue(cm.validate_config())
            # Directories creation shouldn't raise
            cm.create_directories()
            self.assertTrue((Path(td) / "kb").exists())
            self.assertTrue((Path(td) / "out").exists())
