# A Framework for Modular Deconstruction, Analysis, and Recomposition of Knowledge in Transformer-based Language Models

## Abstract

Transformer-based language models encode vast amounts of factual, syntactic, and semantic knowledge in their parameters and intermediate computations. This paper presents a comprehensive, engineering-oriented framework to: (i) deconstruct these models into reusable components, (ii) analyze static parameters and dynamic activations to quantify functional roles, and (iii) recombine selected components into new target architectures using parameter-efficient bridges. We articulate a taxonomy of modules (embeddings, attention heads/groups, and feed-forward networks), introduce metrics to evaluate their semantic coverage and functional interpretability, define a binary interface for a portable knowledge bank, and propose robust protocols for reinjection and validation. The framework is model-agnostic and avoids reliance on any specific model names, focusing instead on architectural patterns commonly shared across contemporary transformer families.

Keywords: modular interpretability, parameter-efficient transfer, transformer language models, adapter composition, knowledge bank, HDF5, torch.fx.

## 1. Introduction

Large transformer language models are not monoliths. They are compositions of interoperable computational blocks whose structure and behavior can be dissected, measured, and repurposed. Despite strong architectural convergence (e.g., RMS normalization, rotary positional encodings, gated linear units), crucial divergences remain (e.g., attention variants, attention windows, vocabulary sizes, hidden dimensions). These divergences complicate direct parameter transplantation but invite a more principled approach: learn compact bridging functions that translate features between closely related computational dialects.

This paper proposes a three-part framework. Part I covers static weight extraction and dynamic activation capture at scale. Part II introduces quantitative metrics and analyses to characterize component-level knowledge and build a catalog of functional modules. Part III formalizes knowledge reinjection using adapter-like bridges and establishes rigorous intrinsic and extrinsic validation protocols. Collectively, these parts turn implicit knowledge into an explicit engineering asset, enabling controlled composition and reuse.

## 2. Background and Architectural Landscape

Modern transformer families display both convergence and divergence:

- Convergence: RMS normalization, rotary positional encodings, gated linear units (e.g., SwiGLU/GeGLU), decoder-only stacks.
- Divergence: multi-head attention (MHA) vs. grouped/multi-query attention (GQA/MQA), global vs. sliding/windowed attention, vocabulary sizes, and hidden-state dimensionality.

We consider four representative, anonymized families across parameter scales: Small (~2B), Medium (~7–9B), Large (~13B), and Optional-XL (~70B). Throughout, we avoid any specific model names and focus on the engineering implications of their architectural choices.

### 2.1 Comparative architectural summary (illustrative)

| Feature | Model A (~2B) | Model B (~9B) | Model C (~7B) | Model D (~13B) |
| :-- | :-- | :-- | :-- | :-- |
| Parameters | ~2B | ~9B | ~7.3B | ~13B |
| Attention | MHA (Q:K:V separate) | GQA (Q heads > KV heads) | GQA (Q heads > KV heads) | MHA |
| Attention window | Global | Alternating Local/Global | Sliding (e.g., 4k) | Global |
| Normalization | RMSNorm | RMSNorm | RMSNorm | RMSNorm |
| Activation | GeGLU | GeGLU | SwiGLU (SiLU) | SwiGLU (SiLU) |
| Positional | RoPE | RoPE | RoPE | RoPE |
| Vocabulary | 256k | 256k | 32k | 32k |

This table acts as a map of incompatibilities. Differences in hidden size require linear projections; differences in attention regimes require careful handling of shared vs. unshared K/V projections; and differences in context strategies affect how knowledge disperses hierarchically across layers.

## 3. Part I — Static and Dynamic Knowledge Acquisition

### 3.1 Static weight extraction and serialization

We extract weights with widely used deep learning frameworks and save them in a language-agnostic, safe tensor format. To create self-contained modules, extraction scripts:

- Identify the target submodule programmatically.
- Traverse the global state dictionary and construct a new keyed dictionary that is local to the module.
- Persist tensors plus minimal metadata (shapes, attention type, group mappings) to safetensors/pt.

Component-specific considerations:
- Embeddings: input embeddings constitute the lexical basis; often tied with the output language head—avoid duplication.
- Attention heads/groups: for MHA, save Q/K/V/O per head; for GQA/MQA, respect shared K/V across query groups and record group mappings.
- Feed-forward networks (FFNs): for GLU variants, save gate/up/down projections together as a coherent block.
- Language modeling head: save once if tied to input embeddings and record tying.

### 3.2 Dynamic activation capture at scale

Beyond static parameters, activations reveal how knowledge is executed at inference time. Two approaches are contrasted:
- Forward hooks: simple, non-intrusive callbacks to capture intermediate inputs/outputs.
- Symbolic tracing (e.g., torch.fx): programmatically trace the forward graph, prune subsequent nodes, and recompile a feature-extractor that halts exactly at target activations. This substantially reduces compute for large-scale activation logging.

### 3.3 Corpus design for targeted probing

To elicit functional behaviors, combine general corpora with targeted datasets:
- Factual recall: open-domain QA (e.g., Natural Questions-like, Trivia-like) to activate entity/relation circuits.
- Syntax: treebank-style sentences and procedurally generated constructions (e.g., center-embedding, long-range dependencies).
- Semantics: textual similarity paraphrase pairs and minimal pairs (e.g., dog/doggy/canine) to expose clustering structure.
- Coreference: Winograd-style pronoun resolution to probe reference tracking.

### 3.4 Scalable storage with HDF5

Activation tensors with shape [batch, seq_len, hidden_size] can scale to terabytes. HDF5 offers:
- Hierarchical grouping reflecting model/layer/token hierarchies.
- Chunking, compression, and out-of-core subset reads.
- Robust scalability beyond RAM.

Store full per-token activations; defer aggregation to analysis to avoid irrecoverable information loss.

## 4. Part II — Quantifying Modular Knowledge

We define metrics that translate "semantic coverage," "head interpretability," and "conceptual clustering" into rigorous procedures that also signal transplantability.

### 4.1 Embeddings → Semantic coverage

Metric: downstream language modeling perplexity with frozen embeddings.

Method:
1) Build a small student LM (e.g., 2–4 transformer layers).
2) Initialize its embedding layer with the extracted embedding matrix; if hidden sizes differ, insert a single-layer bridge projection. Freeze the embeddings.
3) Train remaining parameters on a modest, high-quality LM corpus.
4) Evaluate test perplexity: `PPL(X) = exp{-(1/t) * sum_{i=1..t} log p(x_i | x_<i)}`.

Interpretation: lower perplexity implies the frozen space geometry better aligns with next-token prediction. This holistic metric complements intrinsic embedding benchmarks based on human similarity judgments.

### 4.2 Attention maps → Head interpretability

Metrics: Syntactic Head Score (SHS) and Factual Head Score (FHS).

- SHS: quantify alignment with dependency structure using structural-probe-inspired evaluation. For each sentence with gold dependencies, count edges predicted by strong attention weights; mitigate locality bias by stratifying by dependency distance.
- FHS: quantify contribution to factual recall via ablation. Measure average drop in log-probability of the correct answer token when ablating a specific head/group on QA-style tasks.

Interpretation: heads specialize along orthogonal axes. High SHS and low FHS suggests syntactic specialization; the reverse suggests factual retrieval specialization.

### 4.3 FFNs → Conceptual clustering

Metrics: Principal Component Variance (PCV) and Cluster Purity Score (CPS).

Method:
- Collect FFN outputs per token for a semantically annotated corpus (e.g., category labels like animal/tool/location).
- Apply PCA; compute fraction of variance explained by top-k components (PCV).
- Project onto top-k components; cluster (e.g., k-means); compute purity against ground-truth categories (CPS).

Interpretation: high PCV and CPS indicate compact, semantically organized subspaces consistent with FFNs serving as structured memory.

## 5. Part II(b) — From Scores to a Functional Catalog

Exploratory visualization with UMAP/t-SNE supports hypothesis formation; color by known labels and sanity-check cluster structure, but rely on quantitative validation for conclusions. The main deliverable is `head_catalog.json`, mapping each attention head/group to:
- Quantitative scores (e.g., SHS, FHS)
- Derived functional label (e.g., syntactic-dependency, previous-token, entity-retrieval)
- Triggering exemplars (diagnostic sentences with maximal activation)

Functional redundancy is common; track cosine similarity across score vectors to identify functional clusters and pick representative modules.

### 5.1 Knowledge bank schema (binary interface)

| Module | Path | Data | Shape | Metadata |
| :-- | :-- | :-- | :-- | :-- |
| Input embeddings | `embeddings/embeddings.pt` | tensor | [vocab_size, hidden_size] | config.json (dims, source family) |
| Attention head (MHA) | `heads/layer_{i}/head_{j}/` | tensors | q.pt, k.pt, v.pt, o.pt | config.json (dims, attention type) |
| Attention group (GQA) | `heads/layer_{i}/` | tensors | q_group_{g}.pt, kv_group_{g}.pt | config.json (Q↔KV mapping) |
| FFN (GLU variant) | `ffns/layer_{i}/` | tensors | gate.pt, up.pt, down.pt | config.json (activation, dims) |
| LM head | `lm_head/lm_head.pt` | tensor | [hidden_size, vocab_size] | config.json (tying) |
| Semantic clusters | `concepts/` | JSON/tensor | clusters.json, centroids.pt | n/a |
| Head catalog | `metadata/head_catalog.json` | JSON | list of {id, scores, label} | n/a |

This schema acts as an ABI between extraction (Part I) and recomposition (Part III), enabling independent development and reliable interoperation.

## 6. Part III — Knowledge Reinjection via Parameter-Efficient Bridges

### 6.1 Bridge networks as adapter modules

Bridge networks are small bottleneck MLPs inserted around transplanted components to resolve dimensionality and distribution shifts between donor and recipient. A typical adapter structure: down-projection → nonlinearity → up-projection with residual skip. During integration, freeze donor and recipient backbones; train only the bridges to avoid catastrophic interference while learning the translation.

Example: transplanting an attention head/group from a 4096-hidden donor into a 2048-hidden recipient requires (i) an input bridge 2048→4096 and (ii) an output bridge 4096→2048.

### 6.2 Reinjection strategies

1) Embedding initialization: initialize recipient embeddings from the bank; add a linear bridge if hidden sizes differ; fine-tune only bridges or shallow layers.
2) Functional module injection: wrap a specialized head/group or FFN with bridges and insert it frozen into a recipient layer; train only bridges (adapter composition).
3) Cluster-memory augmentation: use semantic centroids as external memory in retrieval-augmented generation; retrieve concept vectors on demand and condition the recipient.

### 6.3 Advanced composition and alternatives

- Adapter fusion: transplant multiple specialized modules and learn a lightweight fusion layer to combine their outputs contextually.
- Modular distillation: transfer functions rather than weights; train a native recipient module to mimic donor activations/attention maps via divergence-based losses, updating only recipient parameters.

## 7. Validation Protocols

Robust evaluation spans intrinsic (module-level) and extrinsic (system-level) tests. Always compare against strong baselines with matched parameter counts and compute budgets.

### 7.1 Intrinsic tests

- Embeddings: intrinsic similarity benchmarks (e.g., rank correlations vs. human similarity judgments) plus downstream PPL with frozen embeddings.
- Attention heads: visualize attention on diagnostic sentences; confirm preservation of syntactic or prior-token patterns.
- FFNs: recompute CPS on the recipient to verify preservation of conceptual clustering.

### 7.2 Extrinsic tests

Probing suite for targeted capabilities:
- POS tagging (treebank-style data)
- Named entity recognition (standard CoNLL-style data)
- Semantic textual similarity (STS-style regression)

General-purpose suite:
- Language quality: perplexity on held-out corpora
- Reasoning/knowledge: broad academic benchmarks (e.g., multi-task QA, ARC-style science QA, truthful reasoning)
- Common-sense reasoning: narrative plausibility and physical reasoning benchmarks

Targeted evaluation: match the benchmark to the transplanted function (e.g., QA for factual heads; code benchmarks for code-specialized heads).

### 7.3 Hypothesis-testing matrix (illustrative)

| Transplanted module | Level | Task | Datasets | Metric(s) | Success criterion |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Input embeddings | Intrinsic | Semantic similarity | WordSim-style, STS-style | Spearman | Comparable to donor; > 0.6 |
|  | Extrinsic | Language quality | WikiText-style | Perplexity | Lower than baseline |
| Syntactic attention head | Intrinsic | Attention visualization | Diverse syntactic corpus | Qualitative | Patterns align with dependencies |
|  | Extrinsic | Grammatical acceptability | CoLA-style | Matthews corr. | Above baseline |
| Factual attention head | Extrinsic | Open-domain QA | NQ/Trivia-style | EM, F1 | Above baseline |
| Full FFN layer | Extrinsic | Knowledge breadth | Multi-task academic | Accuracy | Above baseline |

## 8. Discussion

The modular perspective reframes language-model development as principled composition of verified cognitive primitives. Key challenges include: (i) bridging dimension and distribution mismatches, (ii) avoiding interference via selective training of bridges, and (iii) ensuring portability and stability via a standardized ABI. Success would unlock reusable knowledge libraries and accelerate downstream system design.

## 9. Limitations and Risks

- Distribution shift: donor activations may not align with recipient dynamics; bridges may need regularization and curriculum schedules.
- Hidden entanglement: modules interact non-locally; ablations and causal tests help avoid spurious attributions.
- Data governance: activation logging can encode sensitive data; apply strict filtering, aggregation, and governance controls.
- Compute/storage: activation banks scale rapidly; efficient sampling, chunking, and compression are essential.

## 10. Reproducibility and Implementation Notes

- Use open-source frameworks for tracing (e.g., symbolic graph capture) and feature extraction; prune graphs to minimize compute.
- Store activations in HDF5 with a stable hierarchical layout and metadata.
- Publish the knowledge bank schema, metadata conventions, and validation scripts.
- Provide seed control, data splits, and hardware specs.

## 11. Conclusion and Outlook

We presented a model-agnostic framework for deconstructing, analyzing, and recomposing modular knowledge in transformer language models. By unifying static extraction, dynamic probing, functional metrics, adapter-based reinjection, and rigorous validation, the framework elevates latent knowledge into a portable engineering asset. Future directions include modular distillation at scale, adapter fusion of heterogeneous skills, and community-curated knowledge banks that democratize composition.

## References (selected)

- Baldi, P., & Hornik, K. (1989). Neural Networks and Principal Component Analysis.
- Houlsby, N., et al. (2019). Parameter-Efficient Transfer Learning for NLP.
- Janson, L., et al. (2024). Optimal Ablation for Interpretability.
- PyTorch Team (2021). torch.fx: Practical Program Capture and Transformation for Deep Learning in Python.
- UMAP-learn documentation; t-SNE original paper.
- HDF5 Group documentation.
- Standard resources on perplexity and intrinsic embedding evaluation.
