import json
from pathlib import Path


def _load_schema(p: Path):
    try:
        import jsonschema  # type: ignore

        return json.load(p.open()), jsonschema
    except Exception:
        return json.load(p.open()), None


def _validate_or_assert(schema: dict, data: dict, js):
    if js is None:
        # Minimal fallback: assert required keys exist
        for key in schema.get("required", []):
            assert key in data
        return
    js.validate(instance=data, schema=schema)  # type: ignore


def test_validation_results_schema(tmp_path: Path):
    schema_path = Path(__file__).parent / "schemas" / "validation_results.schema.json"
    schema, js = _load_schema(schema_path)
    # Minimal example matching our shape
    data = {
        "model_path": "./transplanted",
        "baseline_model": None,
        "intrinsic_validation": {
            "embeddings": {"perplexity": 0.0, "baseline_perplexity": None, "delta": 0.0}
        },
        "extrinsic_validation": {"general_benchmarks": {}},
        "summary": {
            "overall_score": 0.0,
            "component_scores": {},
            "recommendations": [],
        },
    }
    _validate_or_assert(schema, data, js)


def test_analysis_results_schema(tmp_path: Path):
    schema_path = Path(__file__).parent / "schemas" / "analysis_results.schema.json"
    schema, js = _load_schema(schema_path)
    data = {
        "source_model": "dummy",
        "analysis_config": {},
        "component_analysis": {
            "embeddings": {
                "metrics": {
                    "perplexity_score": 0.0,
                    "semantic_coverage": 0.5,
                    "dimension_analysis": {
                        "effective_dim": 1,
                        "pca_explained_variance": [0.5, 0.5],
                    },
                }
            }
        },
        "head_catalog": [],
    }
    _validate_or_assert(schema, data, js)
