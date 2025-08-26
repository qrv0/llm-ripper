"""
Generate minimal counterfactual pairs with simple, controlled templates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def _plural_pairs(n: int) -> List[Dict]:
    # Simple subject-verb agreement minimal pairs
    subjects_sg = ["The cat", "The student", "The company", "This book", "A bird"]
    verbs_sg = ["is", "runs", "jumps", "seems", "appears"]
    verbs_pl = ["are", "run", "jump", "seem", "appear"]
    objs = ["on the table.", "in the park.", "very interesting.", "ready.", "outside."]
    out: List[Dict] = []
    k = 0
    for i in range(max(1, n)):
        s = subjects_sg[i % len(subjects_sg)]
        v_sg = verbs_sg[i % len(verbs_sg)]
        v_pl = verbs_pl[i % len(verbs_pl)]
        o = objs[i % len(objs)]
        # original uses singular verb; counterfactual uses plural verb
        orig = f"{s} {v_sg} {o}"
        cf = f"{s} {v_pl} {o}"
        out.append(
            {
                "id": f"agr-{k}",
                "task": "agreement",
                "original": orig,
                "counterfactual": cf,
                "target_tokens": [v_sg, v_pl],
                "controls": {"subject_number": "sg"},
            }
        )
        k += 1
    return out


def _coref_pairs(n: int) -> List[Dict]:
    # Pronoun coreference minimal pairs using gender/number swap
    templates = [
        ("Alice gave the book to Bob because __ was leaving.", ["she", "he"]),
        ("The dogs chased the cat until __ escaped.", ["they", "it"]),
        ("Mary thanked John after __ helped with the move.", ["he", "she"]),
    ]
    out: List[Dict] = []
    k = 0
    for i in range(max(1, n)):
        tpl, toks = templates[i % len(templates)]
        orig = tpl.replace("__", toks[0])
        cf = tpl.replace("__", toks[1])
        out.append(
            {
                "id": f"coref-{k}",
                "task": "coref",
                "original": orig,
                "counterfactual": cf,
                "target_tokens": toks,
                "controls": {},
            }
        )
        k += 1
    return out


def generate_minimal_pairs(task: str, n: int, out_path: str) -> str:
    """Generate n counterfactual pairs and save as JSONL."""
    if task == "agreement":
        rows = _plural_pairs(n)
    elif task == "coref":
        rows = _coref_pairs(n)
    else:
        raise ValueError(f"Unknown task: {task}")
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")
    return str(p)
