import json
from pathlib import Path
from llm_ripper.utils.run import RunContext


def test_run_context_creates_dirs_and_writes(tmp_path: Path):
    rc = RunContext.create(base=tmp_path)
    # Must create standard subdirs
    for sub in (
        "knowledge_bank",
        "activations",
        "analysis",
        "transplants",
        "validation",
        "causal",
        "traces",
        "counterfactuals",
        "uq",
        "catalog",
        "provenance",
        "reports",
    ):
        assert (rc.root / sub).exists()

    # JSON write is atomic and readable
    p_json = rc.write_json("catalog/heads.json", {"heads": [1, 2, 3]})
    assert p_json.exists()
    data = json.loads(p_json.read_text(encoding="utf-8"))
    assert data["heads"] == [1, 2, 3]

    # JSONL write is atomic and has correct rows
    p_jsonl = rc.write_jsonl("traces/summary.jsonl", [{"a": 1}, {"a": 2}])
    assert p_jsonl.exists()
    lines = [json.loads(line) for line in p_jsonl.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"a": 1}, {"a": 2}]
