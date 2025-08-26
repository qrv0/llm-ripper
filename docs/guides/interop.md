# Interoperability (Merge, Adapters, Tokenizers)

## Global Merge

Average weights across models listed in a spec.

```bash
llm-ripper merge --global examples/merge_spec.yaml --out ./transplants/merged
llm-ripper merge --global examples/merge_spec.yaml --micro --out ./transplants/merged  # + microtransplants
```

Spec (YAML/JSON):

```yaml
models:
  - ./models/modelA
  - ./models/modelB
base_dir: ./models/modelA
micro:
  - knowledge_bank: ./knowledge_bank
    source_component: embeddings
    target_layer: 0
    strategy: embedding_init
```

## Adapters

Import a simple LoRA file and inject as an additional adapter; optionally attach a fusion gate.

```bash
llm-ripper adapters --model ./transplanted/model --import lora.safetensors --layer 3 --fuse
```

## Tokenizer Alignment

Emit token overlap and id mapping.

```bash
llm-ripper tokenize-align --source gpt2 --target distilgpt2 --out ./catalog/tok_align.json
```
