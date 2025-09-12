from llm_ripper.utils.run import RunContext

# Create a demo run with the files the Studio expects, using simple JSON
rc = RunContext.create()

# Minimal demo artifacts
rc.write_json("catalog/heads.json", {"heads": [{"layer": 0, "head": 0, "note": "demo"}]})
rc.write_json("traces/summary.json", {"metric": "nll_delta", "targets": ["head:0.q", "ffn:0.up"], "note": "demo"})
rc.write_json(
    "validation/validation_results.json",
    {
        "summary": {"overall_score": 0.75},
        "recommendations": [
            "Run more samples for stable metrics",
            "Try different transplant strategies",
        ],
    },
)

print("\n\u2713 Beginner demo created at:", rc.root)
print("Open Studio with: make studio RUN_ROOT=", rc.root)
