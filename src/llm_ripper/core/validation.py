"""
Validation module for LLM Ripper.

This module implements comprehensive validation protocols for transplanted components
as described in Section 7 of the framework.
"""

import torch
import torch.nn as nn

try:
    from datasets import load_dataset, Dataset  # type: ignore
except Exception:

    def load_dataset(*args, **kwargs):  # type: ignore
        raise RuntimeError(
            "datasets library is required for online validation benchmarks but is not installed."
        )

    class Dataset:  # type: ignore
        ...


import numpy as np

try:
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef  # type: ignore
except Exception:

    def accuracy_score(y_true, y_pred):  # type: ignore
        import numpy as _np

        y_true = _np.asarray(y_true)
        y_pred = _np.asarray(y_pred)
        return float((y_true == y_pred).mean())

    def f1_score(y_true, y_pred, average="weighted"):  # type: ignore
        return 0.0

    def matthews_corrcoef(y_true, y_pred):  # type: ignore
        return 0.0


try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:

    def spearmanr(x, y):  # type: ignore
        return 0.0, 1.0


import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json

# tqdm not used directly in this module

from ..utils.config import ConfigManager
from ..utils.model_loader import ModelLoader
from ..utils.metrics import MetricsCalculator
from ..utils.data_manager import DataManager

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation results."""

    task_name: str
    metric_name: str
    score: float
    baseline_score: Optional[float] = None
    improvement: Optional[float] = None


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""

    name: str
    dataset_name: str
    task_type: str  # "classification", "regression", "generation"
    metric: str
    num_samples: Optional[int] = None


class ValidationSuite:
    """
    Comprehensive validation suite for transplanted models.

    Implements Section 7: Multi-level validation protocols.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        self.model_loader = ModelLoader(
            cache_dir=config.get("model_cache_dir"), device=config.get("device")
        )

        # Define standard benchmarks
        self.intrinsic_benchmarks = self._define_intrinsic_benchmarks()
        self.extrinsic_benchmarks = self._define_extrinsic_benchmarks()

    def validate_transplanted_model(
        self,
        transplanted_model_path: str,
        baseline_model_name: Optional[str] = None,
        benchmarks: Optional[List[str]] = None,
        output_dir: str = "./validation_results",
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation of a transplanted model.

        Args:
            transplanted_model_path: Path to transplanted model
            baseline_model_name: Baseline model for comparison
            benchmarks: List of benchmarks to run (default: all)
            output_dir: Directory to save validation results

        Returns:
            Dictionary containing all validation results
        """
        logger.info(
            f"Starting validation of transplanted model: {transplanted_model_path}"
        )

        # Load transplanted model with fallback
        try:
            transplanted_model, tokenizer, config = (
                self.model_loader.load_model_and_tokenizer(
                    transplanted_model_path,
                    load_in_8bit=self.config.get("load_in_8bit"),
                    load_in_4bit=self.config.get("load_in_4bit"),
                    trust_remote_code=self.config.get("trust_remote_code"),
                )
            )
        except Exception as e:
            logger.warning(
                f"Could not load transplanted model '{transplanted_model_path}': {e}"
            )
            # Minimal fallback result to not break pipeline
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            minimal_results = {
                "model_path": transplanted_model_path,
                "baseline_model": baseline_model_name,
                "intrinsic_validation": {},
                "extrinsic_validation": {},
                "summary": {
                    "overall_score": 0.0,
                    "component_scores": {},
                    "recommendations": [
                        "Validation skipped: unknown or custom model type; ensure transformers supports the model or set TRUST_REMOTE_CODE=true with internet access.",
                        "If using a local custom model, provide the custom code package or validate using the baseline model only.",
                    ],
                },
            }
            with open(output_path / "validation_results.json", "w") as f:
                json.dump(minimal_results, f, indent=2, default=self._json_serializer)
            return minimal_results

        # Load baseline model if provided
        baseline_model = None
        if baseline_model_name:
            baseline_model, _, _ = self.model_loader.load_model_and_tokenizer(
                baseline_model_name,
                load_in_8bit=self.config.get("load_in_8bit"),
                load_in_4bit=self.config.get("load_in_4bit"),
                trust_remote_code=self.config.get("trust_remote_code"),
            )

        # Determine which benchmarks to run
        if benchmarks is None:
            benchmarks = ["all"]

        import uuid

        validation_results = {
            "model_path": transplanted_model_path,
            "baseline_model": baseline_model_name,
            "intrinsic_validation": {},
            "extrinsic_validation": {},
            "summary": {},
            "run_id": str(uuid.uuid4()),
            "seed": self.config.get("seed"),
        }

        # Run intrinsic validation
        if "all" in benchmarks or any("intrinsic" in b for b in benchmarks):
            intrinsic_results = self.run_intrinsic_validation(
                transplanted_model, tokenizer, baseline_model
            )
            validation_results["intrinsic_validation"] = intrinsic_results

        # Run extrinsic validation
        if "all" in benchmarks or any("extrinsic" in b for b in benchmarks):
            extrinsic_results = self.run_extrinsic_validation(
                transplanted_model, tokenizer, baseline_model, benchmarks
            )
            validation_results["extrinsic_validation"] = extrinsic_results

        # Mechanistic synthetic batteries (offline)
        if any(b in ("mechanistic", "mechanistic_offline") for b in (benchmarks or [])):
            mech = self.run_mechanistic_offline(transplanted_model, tokenizer)
            validation_results.setdefault("mechanistic", {}).update(mech)

        # Generate summary
        validation_results["summary"] = self._generate_validation_summary(
            validation_results
        )

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "validation_results.json", "w") as f:
            json.dump(validation_results, f, indent=2, default=self._json_serializer)

        logger.info(f"Validation completed. Results saved to: {output_path}")

        return validation_results

    # ---------------- Mechanistic offline batteries ----------------
    def run_mechanistic_offline(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        dm = DataManager(self.config)
        out: Dict[str, Any] = {}
        # Agreement battery
        agr = self._battery_agreement(model, tokenizer, dm)
        # Coref battery
        coref = self._battery_coref(model, tokenizer, dm)
        # Simple arithmetic order battery
        add = self._battery_addition(model, tokenizer, dm)
        out["agreement"] = agr
        out["coref"] = coref
        out["addition"] = add
        return out

    def _bootstrap_ci(
        self, vals: List[float], n: int = 200, alpha: float = 0.05
    ) -> List[float]:
        import random
        import numpy as np

        if not vals:
            return [0.0, 0.0]
        means = []
        for _ in range(n):
            s = [random.choice(vals) for __ in range(len(vals))]
            means.append(float(np.mean(s)))
        lo = float(np.percentile(means, 100 * (alpha / 2)))
        hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
        return [lo, hi]

    def _battery_agreement(
        self, model: nn.Module, tok, dm: DataManager
    ) -> Dict[str, Any]:
        items = [
            ("The cat is on the mat.", 1),
            ("The cat are on the mat.", 0),
            ("This book is interesting.", 1),
            ("This book are interesting.", 0),
        ]
        scores: List[float] = []
        with torch.no_grad():
            for text, label in items:
                enc = tok(text, return_tensors="pt")
                enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
                out = model(**enc, labels=enc["input_ids"])  # type: ignore
                # Lower loss implies grammaticality
                pred = 1 if float(out.loss.item()) < 5.0 else 0
                scores.append(1.0 if pred == label else 0.0)
        acc = float(sum(scores) / len(scores))
        ci = self._bootstrap_ci(scores)
        return {"accuracy": acc, "ci": ci, "n": len(scores)}

    def _battery_coref(self, model: nn.Module, tok, dm: DataManager) -> Dict[str, Any]:
        items = [
            ("Alice gave the book to Bob because she was leaving.", ["she"]),
            ("Alice gave the book to Bob because he was leaving.", ["he"]),
        ]
        scores: List[float] = []
        with torch.no_grad():
            for text, toks in items:
                enc = tok(text, return_tensors="pt")
                enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits[:, -1, :]

                # pick token logits
                def _score_token(c):
                    try:
                        ids = tok.encode(c, add_special_tokens=False)
                        if len(ids) == 1:
                            return float(logits[0, ids[0]].item())
                    except Exception:
                        pass
                    return float(logits[0].max().item())

                scores_list = [_score_token(t) for t in toks]
                # favor higher logit for given pronoun
                scores.append(1.0 if max(scores_list) == scores_list[0] else 0.0)
        acc = float(sum(scores) / len(scores))
        ci = self._bootstrap_ci(scores)
        return {"accuracy": acc, "ci": ci, "n": len(scores)}

    def _battery_addition(
        self, model: nn.Module, tok, dm: DataManager
    ) -> Dict[str, Any]:
        # Simple arithmetic check by prompting and scoring next token
        items = [
            ("2 + 2 =", "4"),
            ("3 + 5 =", "8"),
            ("1 + 7 =", "8"),
        ]
        scores: List[float] = []
        with torch.no_grad():
            for prompt, ans in items:
                enc = tok(prompt, return_tensors="pt")
                enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
                out = model(**enc)
                logits = out.logits[:, -1, :]
                try:
                    ids = tok.encode(ans, add_special_tokens=False)
                    pred_ok = False
                    if len(ids) >= 1:
                        pred_ok = bool(
                            int(torch.argmax(logits, dim=-1)[0].item()) == ids[0]
                        )
                    scores.append(1.0 if pred_ok else 0.0)
                except Exception:
                    scores.append(0.0)
        acc = float(sum(scores) / len(scores))
        ci = self._bootstrap_ci(scores)
        return {"accuracy": acc, "ci": ci, "n": len(scores)}

    def run_intrinsic_validation(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """
        Run intrinsic validation (module-level).

        Implements Section 7.1: Intrinsic validation.
        """
        logger.info("Running intrinsic validation...")

        intrinsic_results = {}

        # Embedding validation
        embedding_results = self._validate_embeddings(model, tokenizer, baseline_model)
        intrinsic_results["embeddings"] = embedding_results

        # Attention pattern validation
        attention_results = self._validate_attention_patterns(
            model, tokenizer, baseline_model
        )
        intrinsic_results["attention_patterns"] = attention_results

        # FFN cluster validation
        ffn_results = self._validate_ffn_clusters(model, tokenizer, baseline_model)
        intrinsic_results["ffn_clusters"] = ffn_results

        return intrinsic_results

    def run_extrinsic_validation(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
        benchmarks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run extrinsic validation (system-level).

        Implements Section 7.2: Extrinsic validation.
        """
        logger.info("Running extrinsic validation...")

        extrinsic_results = {}

        # Run probing tasks
        probing_results = self._run_probing_tasks(model, tokenizer, baseline_model)
        extrinsic_results["probing_tasks"] = probing_results

        # Run general benchmarks
        benchmark_results = self._run_general_benchmarks(
            model, tokenizer, baseline_model
        )
        extrinsic_results["general_benchmarks"] = benchmark_results

        # Run targeted evaluation
        targeted_results = self._run_targeted_evaluation(
            model, tokenizer, baseline_model
        )
        extrinsic_results["targeted_evaluation"] = targeted_results

        return extrinsic_results

    def _validate_embeddings(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Validate transplanted embeddings."""

        results = {}

        # Test semantic similarity preservation
        word_pairs = [
            ("dog", "puppy"),
            ("car", "automobile"),
            ("happy", "joyful"),
            ("big", "large"),
            ("run", "sprint"),
            ("house", "home"),
        ]

        similarities = []
        baseline_similarities = []

        for word1, word2 in word_pairs:
            try:
                # Get embeddings for transplanted model
                token1 = tokenizer.encode(word1, add_special_tokens=False)[0]
                token2 = tokenizer.encode(word2, add_special_tokens=False)[0]

                emb1 = model.get_input_embeddings().weight[token1]
                emb2 = model.get_input_embeddings().weight[token2]

                sim = torch.cosine_similarity(emb1, emb2, dim=0).item()
                similarities.append(sim)

                # Get baseline similarities if available
                if baseline_model is not None:
                    base_emb1 = baseline_model.get_input_embeddings().weight[token1]
                    base_emb2 = baseline_model.get_input_embeddings().weight[token2]
                    base_sim = torch.cosine_similarity(
                        base_emb1, base_emb2, dim=0
                    ).item()
                    baseline_similarities.append(base_sim)

            except Exception as e:
                logger.warning(f"Could not compute similarity for {word1}-{word2}: {e}")

        results["semantic_similarity"] = {
            "mean_similarity": float(np.mean(similarities)),
            "word_pairs_tested": len(similarities),
        }

        if baseline_similarities:
            correlation, _ = spearmanr(similarities, baseline_similarities)
            results["baseline_correlation"] = float(correlation)

        # Test perplexity on a small corpus
        perplexity = self._compute_model_perplexity(model, tokenizer)
        results["perplexity"] = perplexity

        if baseline_model is not None:
            baseline_perplexity = self._compute_model_perplexity(
                baseline_model, tokenizer
            )
            results["baseline_perplexity"] = baseline_perplexity
            results["perplexity_ratio"] = perplexity / baseline_perplexity

        return results

    def _validate_attention_patterns(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Validate transplanted attention patterns."""

        results = {}

        # Test attention patterns on diagnostic sentences
        diagnostic_sentences = [
            "The cat sat on the mat.",
            "John gave Mary the book.",
            "The dog that chased the cat ran away.",
            "She said that he would come tomorrow.",
        ]

        pattern_similarities = []

        for sentence in diagnostic_sentences:
            try:
                # Tokenize sentence
                inputs = tokenizer(sentence, return_tensors="pt")

                # Get attention patterns
                with torch.no_grad():
                    outputs = model(**inputs, output_attentions=True)
                    attentions = outputs.attentions

                # Analyze attention patterns: average over heads
                if attentions and len(attentions) > 0:
                    # Average attention across heads and layers
                    avg_attention = torch.mean(
                        attentions[0], dim=1
                    )  # Average over heads

                    # Compute pattern metrics
                    diagonal_attention = torch.diag(avg_attention[0]).mean().item()
                    pattern_similarities.append(diagonal_attention)

            except Exception as e:
                logger.warning(
                    f"Could not analyze attention for: {sentence[:20]}... : {e}"
                )

        results["attention_patterns"] = {
            "mean_diagonal_attention": (
                float(np.mean(pattern_similarities)) if pattern_similarities else 0.0
            ),
            "sentences_analyzed": len(pattern_similarities),
        }

        return results

    def _validate_ffn_clusters(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Validate FFN cluster preservation."""

        results = {}

        # Test concept clustering with word categories
        word_categories = {
            "animals": ["dog", "cat", "bird", "fish", "lion"],
            "colors": ["red", "blue", "green", "yellow", "purple"],
            "tools": ["hammer", "screwdriver", "wrench", "saw", "drill"],
        }

        category_coherence = {}

        for category, words in word_categories.items():
            try:
                # Get activations for category words
                activations = []

                for word in words:
                    tokens = tokenizer.encode(word, add_special_tokens=False)
                    if tokens:
                        # Simple forward pass to get hidden states
                        with torch.no_grad():
                            inputs = tokenizer(word, return_tensors="pt")
                            outputs = model(**inputs, output_hidden_states=True)

                            # Use last hidden state
                            hidden_state = outputs.hidden_states[-1]
                            avg_activation = hidden_state.mean(dim=1).squeeze()
                            activations.append(avg_activation.cpu().numpy())

                if len(activations) > 1:
                    # Compute intra-category coherence
                    activations = np.array(activations)
                    coherence = self._compute_cluster_coherence(activations)
                    category_coherence[category] = coherence

            except Exception as e:
                logger.warning(
                    f"Could not analyze FFN clusters for category {category}: {e}"
                )

        results["cluster_coherence"] = category_coherence
        results["mean_coherence"] = (
            float(np.mean(list(category_coherence.values())))
            if category_coherence
            else 0.0
        )

        return results

    def _run_probing_tasks(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Run probing tasks for specific capabilities."""

        results = {}

        # POS Tagging probe: prefer real CoNLL2003 when online, else synthetic
        if not self.config.get("offline"):
            try:
                pos_result = self._run_pos_tagging_probe_real(model, tokenizer)
            except Exception as e:
                logger.warning(f"Real POS probe failed, falling back to synthetic: {e}")
                pos_result = self._run_pos_tagging_probe(model, tokenizer)
        else:
            pos_result = self._run_pos_tagging_probe(model, tokenizer)
        results["pos_tagging"] = pos_result

        # Semantic similarity probe: prefer real STS-B
        if not self.config.get("offline"):
            try:
                sts_result = self._run_semantic_similarity_probe_real(model, tokenizer)
            except Exception as e:
                logger.warning(
                    f"Real STS-B probe failed, falling back to synthetic: {e}"
                )
                sts_result = self._run_semantic_similarity_probe(model, tokenizer)
        else:
            sts_result = self._run_semantic_similarity_probe(model, tokenizer)
        results["semantic_similarity"] = sts_result

        # Factual knowledge / commonsense QA
        if not self.config.get("offline"):
            try:
                factual_result = self._run_commonsense_qa_real(model, tokenizer)
            except Exception as e:
                logger.warning(f"Real CommonsenseQA probe failed, falling back: {e}")
                factual_result = self._run_factual_knowledge_probe(model, tokenizer)
        else:
            factual_result = self._run_factual_knowledge_probe(model, tokenizer)
        results["factual_knowledge"] = factual_result

        # NER (CoNLL2003): prefer online
        if not self.config.get("offline"):
            try:
                ner_result = self._run_ner_probe_real(model, tokenizer)
            except Exception as e:
                logger.warning(f"Real NER probe failed, falling back to synthetic: {e}")
                ner_result = self._run_ner_probe_synthetic(model, tokenizer)
        else:
            ner_result = self._run_ner_probe_synthetic(model, tokenizer)
        results["ner"] = ner_result

        return results

    def _run_general_benchmarks(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Run general benchmark evaluations."""

        results = {}

        # Language modeling perplexity
        perplexity = self._compute_model_perplexity(model, tokenizer)
        results["language_modeling"] = {"perplexity": perplexity}

        # Common sense reasoning
        commonsense_result = self._evaluate_commonsense_reasoning(model, tokenizer)
        results["commonsense_reasoning"] = commonsense_result

        # CoLA (GLUE) MCC when online
        if not self.config.get("offline"):
            try:
                results["cola_mcc"] = self._run_cola_mcc_real(model, tokenizer)
            except Exception as e:
                logger.warning(f"CoLA MCC evaluation failed: {e}")

        return results

    def _run_targeted_evaluation(
        self,
        model: nn.Module,
        tokenizer: Any,
        baseline_model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Run targeted evaluation based on transplanted components."""

        results = {}

        # This would be customized based on what was transplanted
        # For now, run general capability tests

        # Grammatical acceptability
        grammatical_result = self._evaluate_grammatical_acceptability(model, tokenizer)
        results["grammatical_acceptability"] = grammatical_result

        # Knowledge retrieval
        knowledge_result = self._evaluate_knowledge_retrieval(model, tokenizer)
        results["knowledge_retrieval"] = knowledge_result

        return results

    def _run_pos_tagging_probe(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Run a real POS tagging linear probe on a small synthetic dataset (offline).
        Uses tokenizer with is_split_into_words to align tokens to hidden states.
        """
        from ..utils.data_manager import DataManager

        dm = DataManager(self.config)
        ds = dm.create_validation_dataset("pos")
        tag_vocab: Dict[str, int] = {}
        X: List[torch.Tensor] = []
        Y: List[int] = []
        model.eval()
        for ex in ds:
            tokens = ex["tokens"]
            tags = ex["pos_tags"]
            enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states[-1][0]  # [Tsub, H]
            # Map subwords to word index
            word_ids = (
                enc.word_ids() if hasattr(enc, "word_ids") else [None] * hs.shape[0]
            )
            used_word_idx = set()
            for i, wi in enumerate(word_ids):
                if wi is None or wi in used_word_idx or wi >= len(tags):
                    continue
                used_word_idx.add(wi)
                X.append(hs[i].detach().cpu())
                if tags[wi] not in tag_vocab:
                    tag_vocab[tags[wi]] = len(tag_vocab)
                Y.append(tag_vocab[tags[wi]])
        if not X or not Y:
            return {"accuracy": 0.0, "samples_tested": 0}
        X_t = torch.stack(X)  # [N, H]
        Y_t = torch.tensor(Y, dtype=torch.long)
        # Train a tiny linear classifier
        clf = nn.Linear(X_t.shape[1], len(tag_vocab))
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        clf.train()
        epochs = 50
        for _ in range(epochs):
            opt.zero_grad()
            logits = clf(X_t)
            loss = loss_fn(logits, Y_t)
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            pred = clf(X_t).argmax(dim=-1)
        acc = float((pred == Y_t).float().mean().item())
        return {"accuracy": acc, "samples_tested": int(X_t.shape[0])}

    def _run_semantic_similarity_probe(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Run semantic similarity probe."""

        sentence_pairs = [
            ("The cat is sleeping.", "A cat is napping."),
            ("I love pizza.", "Pizza is delicious."),
            ("The weather is nice.", "It's a beautiful day."),
        ]

        similarities = []

        for sent1, sent2 in sentence_pairs:
            try:
                # Get sentence embeddings (mean pooled last hidden state)
                inputs1 = tokenizer(sent1, return_tensors="pt")
                inputs2 = tokenizer(sent2, return_tensors="pt")

                with torch.no_grad():
                    outputs1 = model(**inputs1, output_hidden_states=True)
                    outputs2 = model(**inputs2, output_hidden_states=True)

                    # Use mean of last hidden state
                    emb1 = outputs1.hidden_states[-1].mean(dim=1)
                    emb2 = outputs2.hidden_states[-1].mean(dim=1)

                    sim = torch.cosine_similarity(emb1, emb2).item()
                    similarities.append(sim)

            except Exception as e:
                logger.warning(f"Could not compute similarity: {e}")

        return {
            "mean_similarity": float(np.mean(similarities)) if similarities else 0.0,
            "pairs_tested": len(similarities),
        }

    def _run_semantic_similarity_probe_real(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Real STS-B probe (validation split): Spearman correlation between cosine similarity and gold scores."""
        dm = DataManager(self.config)
        ds = dm.load_glue_stsb("validation")
        y_true: List[float] = []
        sims: List[float] = []
        for ex in ds.select(range(min(500, len(ds)))):
            s1, s2 = ex["sentence1"], ex["sentence2"]
            y_true.append(float(ex.get("score", ex.get("label", 0.0))))
            with torch.no_grad():
                emb1 = (
                    model(
                        **tokenizer(s1, return_tensors="pt"), output_hidden_states=True
                    )
                    .hidden_states[-1]
                    .mean(dim=1)
                )
                emb2 = (
                    model(
                        **tokenizer(s2, return_tensors="pt"), output_hidden_states=True
                    )
                    .hidden_states[-1]
                    .mean(dim=1)
                )
                sims.append(torch.cosine_similarity(emb1, emb2).item())
        rho = float(spearmanr(y_true, sims).correlation)
        return {"spearman": rho, "pairs_tested": len(sims)}

    def _run_ner_probe_synthetic(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """NER linear probe (synthetic offline dataset) with micro-F1."""
        dm = DataManager(self.config)
        ds = dm.create_validation_dataset("ner")
        X_list, Y_list = [], []
        for ex in ds:
            tokens = ex["tokens"]
            tags = ex["ner_tags"]
            enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states[-1][0]
            word_ids = (
                enc.word_ids() if hasattr(enc, "word_ids") else [None] * hs.shape[0]
            )
            used = set()
            for i, wi in enumerate(word_ids):
                if wi is None or wi in used or wi >= len(tags):
                    continue
                used.add(wi)
                X_list.append(hs[i].cpu())
                Y_list.append(tags[wi])
        if not X_list:
            return {"f1_micro": 0.0, "samples_tested": 0}
        uniq = {t: i for i, t in enumerate(sorted(set(Y_list)))}
        X = torch.stack(X_list)
        Y = torch.tensor([uniq[t] for t in Y_list], dtype=torch.long)
        clf = nn.Linear(X.shape[1], len(uniq))
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        clf.train()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(clf(X), Y)
            loss.backward()
            opt.step()
        clf.eval()
        pred = clf(X).argmax(dim=-1).numpy()
        f1 = float(f1_score(Y.numpy(), pred, average="micro"))
        return {"f1_micro": f1, "samples_tested": int(X.shape[0])}

    def _run_ner_probe_real(self, model: nn.Module, tokenizer: Any) -> Dict[str, Any]:
        """NER linear probe (CoNLL2003) with micro-F1."""
        dm = DataManager(self.config)
        train = dm.load_conll2003("train").select(range(1000))
        val = dm.load_conll2003("validation").select(range(1000))

        def featurize(ds):
            X_list, Y_list = [], []
            for ex in ds:
                tokens = ex["tokens"]
                tags = ex["ner_tags"]
                enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states[-1][0]
                word_ids = (
                    enc.word_ids() if hasattr(enc, "word_ids") else [None] * hs.shape[0]
                )
                used = set()
                for i, wi in enumerate(word_ids):
                    if wi is None or wi in used or wi >= len(tags):
                        continue
                    used.add(wi)
                    X_list.append(hs[i].cpu())
                    Y_list.append(int(tags[wi]))
            return (
                (torch.stack(X_list), torch.tensor(Y_list, dtype=torch.long))
                if X_list
                else (None, None)
            )

        Xtr, Ytr = featurize(train)
        Xva, Yva = featurize(val)
        if Xtr is None or Xva is None:
            return {"f1_micro": 0.0, "samples_tested": 0}
        clf = nn.Linear(Xtr.shape[1], int(max(int(Ytr.max()) + 1, 1)))
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        clf.train()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(clf(Xtr), Ytr)
            loss.backward()
            opt.step()
        clf.eval()
        pred = clf(Xva).argmax(dim=-1).numpy()
        f1 = float(f1_score(Yva.numpy(), pred, average="micro"))
        return {"f1_micro": f1, "samples_tested": int(Xva.shape[0])}

    def _run_pos_tagging_probe_real(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Real POS probe using CoNLL2003: train linear probe on small train subset and evaluate on validation."""
        dm = DataManager(self.config)
        train = dm.load_conll2003("train")
        val = dm.load_conll2003("validation")

        def featurize(ds):
            X_list: List[torch.Tensor] = []
            Y_list: List[int] = []
            for ex in ds:
                tokens = ex["tokens"]
                tags = ex["pos_tags"]
                enc = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states[-1][0]
                word_ids = (
                    enc.word_ids() if hasattr(enc, "word_ids") else [None] * hs.shape[0]
                )
                used = set()
                for i, wi in enumerate(word_ids):
                    if wi is None or wi in used or wi >= len(tags):
                        continue
                    used.add(wi)
                    X_list.append(hs[i].cpu())
                    Y_list.append(int(tags[wi]))
            return (
                (torch.stack(X_list), torch.tensor(Y_list, dtype=torch.long))
                if X_list
                else (None, None)
            )

        Xtr, Ytr = featurize(train.select(range(min(200, len(train)))))
        Xva, Yva = featurize(val.select(range(min(200, len(val)))))
        if Xtr is None or Xva is None:
            return {"accuracy": 0.0, "samples_tested": 0}
        clf = nn.Linear(Xtr.shape[1], int(max(int(Ytr.max()) + 1, 1)))
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        clf.train()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(clf(Xtr), Ytr)
            loss.backward()
            opt.step()
        clf.eval()
        pred = clf(Xva).argmax(dim=-1)
        acc = float((pred == Yva).float().mean().item())
        return {"accuracy": acc, "samples_tested": int(Xva.shape[0])}

    def _run_commonsense_qa_real(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """CommonsenseQA accuracy by option likelihood scoring."""
        dm = DataManager(self.config)
        ds = dm.load_commonsense_qa("validation")
        ds = ds.select(range(min(200, len(ds))))
        correct, total = 0, 0
        for ex in ds:
            q = ex["question"]
            labels = ex["choices"]["label"]
            options = ex["choices"]["text"]
            answer_key = ex["answerKey"]
            gold_idx = (
                labels.index(answer_key)
                if isinstance(answer_key, str)
                else int(answer_key)
            )
            scores: List[float] = []
            for opt in options:
                prompt_ids = tokenizer(q + " Answer:", return_tensors="pt")["input_ids"]
                opt_ids = tokenizer(" " + opt, add_special_tokens=False)["input_ids"]
                ctx = prompt_ids
                logp = 0.0
                for tid in opt_ids:
                    with torch.no_grad():
                        logits = model(input_ids=ctx).logits[:, -1, :]
                    logp += torch.log_softmax(logits, dim=-1)[0, tid].item()
                    ctx = torch.cat([ctx, torch.tensor([[tid]])], dim=-1)
                scores.append(logp)
            pred = int(torch.tensor(scores).argmax())
            total += 1
            if pred == gold_idx:
                correct += 1
        return {"accuracy": correct / total if total else 0.0, "total": total}

    def _run_factual_knowledge_probe(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Run factual knowledge probe."""

        # Simple factual questions
        facts = [
            ("The capital of France is", "Paris"),
            ("The largest planet is", "Jupiter"),
            ("The author of Romeo and Juliet is", "Shakespeare"),
        ]

        correct_predictions = 0
        total_questions = len(facts)

        for prompt, answer in facts:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_text = generated_text[len(prompt) :].strip()

                # Simple substring matching
                if answer.lower() in generated_text.lower():
                    correct_predictions += 1

            except Exception as e:
                logger.warning(f"Could not evaluate factual question: {e}")

        return {
            "accuracy": (
                correct_predictions / total_questions if total_questions > 0 else 0.0
            ),
            "correct": correct_predictions,
            "total": total_questions,
        }

    def _evaluate_commonsense_reasoning(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Evaluate commonsense reasoning with keyword matching (offline)."""
        qa = [
            ("If it's raining, you should take an", ["umbrella", "raincoat"]),
            ("When you're hungry, you should", ["eat", "food"]),
            ("To turn on a light, you need to", ["switch", "flip", "press"]),
        ]
        correct = 0
        for q, keywords in qa:
            try:
                inputs = tokenizer(q, return_tensors="pt")
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=getattr(tokenizer, "eos_token_id", None),
                    )
                ans = tokenizer.decode(outputs[0], skip_special_tokens=True)[
                    len(q) :
                ].lower()
                if any(k in ans for k in keywords):
                    correct += 1
            except Exception as e:
                logger.warning(f"Could not evaluate commonsense question: {e}")
        return {
            "accuracy": correct / len(qa) if qa else 0.0,
            "correct": correct,
            "total": len(qa),
        }

    def _evaluate_grammatical_acceptability(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Evaluate grammatical acceptability judgment."""

        # Grammatical vs ungrammatical sentences
        sentence_pairs = [
            ("The cat sits on the mat.", True),
            ("Cat the sits mat on the.", False),
            ("She is reading a book.", True),
            ("Is she book a reading.", False),
        ]

        correct_judgments = 0

        for sentence, is_grammatical in sentence_pairs:
            try:
                # Compute perplexity as proxy for grammaticality
                inputs = tokenizer(sentence, return_tensors="pt")

                with torch.no_grad():
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss.item()

                # Lower loss (perplexity) should indicate more grammatical
                predicted_grammatical = loss < 5.0  # Threshold

                if predicted_grammatical == is_grammatical:
                    correct_judgments += 1

            except Exception as e:
                logger.warning(f"Could not evaluate grammaticality: {e}")

        return {
            "accuracy": (
                correct_judgments / len(sentence_pairs) if sentence_pairs else 0.0
            ),
            "correct": correct_judgments,
            "total": len(sentence_pairs),
        }

    def _evaluate_knowledge_retrieval(
        self, model: nn.Module, tokenizer: Any
    ) -> Dict[str, Any]:
        """Evaluate knowledge retrieval capabilities."""

        # Knowledge retrieval prompts with expected answers
        qa = [
            (
                "The first president of the United States was",
                ["Washington", "George Washington"],
            ),
            ("The chemical symbol for gold is", ["Au"]),
            (
                "The speed of light is approximately",
                ["300000", "3e8", "299792", "three hundred thousand"],
            ),
        ]
        knowledge_retrievals = 0
        for prompt, answers in qa:
            try:
                inputs = tokenizer(prompt, return_tensors="pt")

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                ans = tokenizer.decode(outputs[0], skip_special_tokens=True)[
                    len(prompt) :
                ].lower()
                if any(a.lower() in ans for a in answers):
                    knowledge_retrievals += 1

            except Exception as e:
                logger.warning(f"Could not evaluate knowledge retrieval: {e}")

        return {
            "retrieval_rate": knowledge_retrievals / len(qa) if qa else 0.0,
            "successful_retrievals": knowledge_retrievals,
            "total_prompts": len(qa),
        }

    def _compute_model_perplexity(self, model: nn.Module, tokenizer: Any) -> float:
        """Compute model perplexity on a small test corpus using MetricsCalculator."""
        test_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "Pack my box with five dozen liquor jugs.",
            "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
        ]
        try:
            device = (
                next(model.parameters()).device
                if hasattr(model, "parameters")
                else "cpu"
            )
            return MetricsCalculator.compute_perplexity(
                model, tokenizer, test_sentences, device=str(device)
            )
        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return float("inf")

    def _run_cola_mcc_real(self, model: nn.Module, tokenizer: Any) -> Dict[str, Any]:
        """GLUE CoLA: train logistic probe on train subset, evaluate MCC on validation."""
        dm = DataManager(self.config)
        ds_train = dm.load_glue_cola("train").select(range(1000))
        ds_val = dm.load_glue_cola("validation")

        def embed(sentences):
            X: List[torch.Tensor] = []
            for s in sentences:
                with torch.no_grad():
                    hs = model(
                        **tokenizer(s, return_tensors="pt"), output_hidden_states=True
                    ).hidden_states[-1]
                X.append(hs.mean(dim=1).squeeze(0).cpu())
            return torch.stack(X)

        Xtr = embed([ex["sentence"] for ex in ds_train])
        Ytr = torch.tensor([int(ex["label"]) for ex in ds_train], dtype=torch.long)
        Xva = embed([ex["sentence"] for ex in ds_val])
        Yva = torch.tensor([int(ex["label"]) for ex in ds_val], dtype=torch.long)
        clf = nn.Linear(Xtr.shape[1], 2)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        clf.train()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(clf(Xtr), Ytr)
            loss.backward()
            opt.step()
        clf.eval()
        pred = clf(Xva).argmax(dim=-1).numpy()
        mcc = float(matthews_corrcoef(Yva.numpy(), pred))
        return {"mcc": mcc, "samples_tested": int(Xva.shape[0])}

    def _compute_cluster_coherence(self, activations: np.ndarray) -> float:
        """Compute coherence of activation clusters."""

        if len(activations) < 2:
            return 0.0

        # Compute pairwise similarities
        similarities = []
        for i in range(len(activations)):
            for j in range(i + 1, len(activations)):
                sim = np.dot(activations[i], activations[j]) / (
                    np.linalg.norm(activations[i]) * np.linalg.norm(activations[j])
                )
                similarities.append(sim)

        return float(np.mean(similarities))

    def _generate_validation_summary(
        self, validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a summary of validation results."""

        summary = {"overall_score": 0.0, "component_scores": {}, "recommendations": []}

        # Aggregate scores from different validation components
        scores = []

        # Intrinsic validation scores
        intrinsic = validation_results.get("intrinsic_validation", {})
        if "embeddings" in intrinsic:
            if "semantic_similarity" in intrinsic["embeddings"]:
                score = intrinsic["embeddings"]["semantic_similarity"][
                    "mean_similarity"
                ]
                scores.append(score)
                summary["component_scores"]["embedding_similarity"] = score

        # Extrinsic validation scores
        extrinsic = validation_results.get("extrinsic_validation", {})
        if "probing_tasks" in extrinsic:
            for task, result in extrinsic["probing_tasks"].items():
                # Include accuracy metrics
                if "accuracy" in result:
                    scores.append(result["accuracy"])
                    summary["component_scores"][f"{task}_accuracy"] = result["accuracy"]
                # Include micro-F1 (e.g., NER)
                if "f1_micro" in result:
                    scores.append(result["f1_micro"])
                    summary["component_scores"][f"{task}_f1_micro"] = result["f1_micro"]
                # Include correlation metrics (normalized to [0,1] by mapping [-1,1] to [0,1])
                if "spearman" in result:
                    sp = result["spearman"]
                    norm = (sp + 1.0) / 2.0
                    scores.append(norm)
                    summary["component_scores"][f"{task}_spearman_norm"] = norm

        # Calculate overall score
        if scores:
            summary["overall_score"] = float(np.mean(scores))

        # Generate recommendations
        if summary["overall_score"] > 0.8:
            summary["recommendations"].append(
                "Transplantation appears highly successful"
            )
        elif summary["overall_score"] > 0.6:
            summary["recommendations"].append("Transplantation shows moderate success")
        else:
            summary["recommendations"].append(
                "Transplantation may need further optimization"
            )

        return summary

    def _define_intrinsic_benchmarks(self) -> List[BenchmarkConfig]:
        """Define intrinsic validation benchmarks."""

        return [
            BenchmarkConfig(
                name="semantic_similarity",
                dataset_name="wordnet",
                task_type="regression",
                metric="spearman_correlation",
            ),
            BenchmarkConfig(
                name="attention_patterns",
                dataset_name="syntactic_probes",
                task_type="classification",
                metric="accuracy",
            ),
        ]

    def _define_extrinsic_benchmarks(self) -> List[BenchmarkConfig]:
        """Define extrinsic validation benchmarks."""

        return [
            BenchmarkConfig(
                name="cola",
                dataset_name="glue",
                task_type="classification",
                metric="matthews_corrcoef",
                num_samples=1000,
            ),
            BenchmarkConfig(
                name="stsb",
                dataset_name="glue",
                task_type="regression",
                metric="spearman_correlation",
                num_samples=1000,
            ),
            BenchmarkConfig(
                name="pos_tagging",
                dataset_name="universal_dependencies",
                task_type="classification",
                metric="accuracy",
                num_samples=1000,
            ),
        ]

    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
