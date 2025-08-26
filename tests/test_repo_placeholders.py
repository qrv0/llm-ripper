from pathlib import Path
import unittest


FORBIDDEN = (
    "PUT_MODEL_HERE",
    "<placeholder>",
    "PLACEHOLDER",
)


def scan_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


class TestRepoPlaceholders(unittest.TestCase):
    def test_no_placeholders_in_src_and_examples(self):
        root = Path(__file__).parent.parent
        targets = [root / "src", root / "examples", root / ".env.example"]
        for t in targets:
            if t.is_file():
                content = scan_text(t)
                for token in FORBIDDEN:
                    self.assertNotIn(
                        token, content, f"Found forbidden token {token} in {t}"
                    )
                continue
            for p in t.rglob("*"):
                if p.is_file():
                    content = scan_text(p)
                    for token in FORBIDDEN:
                        self.assertNotIn(
                            token, content, f"Found forbidden token {token} in {p}"
                        )

    def test_examples_pipeline_static_checks(self):
        pipeline = (
            Path(__file__).parent.parent / "examples" / "run_complete_pipeline.py"
        )
        text = scan_text(pipeline)
        self.assertIn("ActivationCapture", text)
        self.assertIn(
            "from llm_ripper.core.activation_capture import ActivationCapture", text
        )
        self.assertNotIn("example_", text, "Pipeline should not use example_* paths")
