# LLM Ripper (Português)

Este repositório contém um framework de produção para desconstrução, análise e recomposição modular de conhecimento em LLMs baseados em Transformers.

- Como instalar
  - `pip install -r requirements.txt`
  - `pip install -e .`
- Como usar (CLI)
  - `llm-ripper extract --model "<seu-modelo>" --output-dir ./knowledge_bank`
    - Opcional: `--model-type base` (mais leve quando não extrai `lm_head`)
  - `llm-ripper capture --model "<seu-modelo>" --output-file activations.h5 --dataset wikitext`
    - Offline/dataset-mode: `HDF5_STORE_MODE=dataset llm-ripper capture --model "<seu-modelo>" --output-file activations.h5 --offline`
  - `llm-ripper analyze --knowledge-bank ./knowledge_bank --output-dir ./analysis`
    - JSON: `--json`
  - `llm-ripper transplant --source ./knowledge_bank --target "<modelo-alvo>" --output-dir ./transplanted`
- `llm-ripper validate --model ./transplanted --baseline "<modelo-alvo>" --output-dir ./validation_results`
  - `llm-ripper inspect --knowledge-bank ./knowledge_bank`
  - `llm-ripper reattach --transplanted-dir ./transplanted --offline --json`

Atenção: substitua os nomes de modelos de exemplo pelos seus.

Opções úteis:
- `--offline`: executa sem baixar datasets (usa corpora sintéticos).
- `--seed 42`: fixa semente global.
- `--json`: imprime saídas estruturadas para automação.
- `inspect`: `llm-ripper inspect --knowledge-bank ./knowledge_bank` lista components/metadata e conta parâmetros (suporte a sharded .pt via .index.json).
- Validação online (padrão): GLUE (CoLA/STS-B), CoNLL2003 (POS/NER) via linear probes, CommonsenseQA.
- Validação offline: datasets sintéticos reprodutíveis (sem rede).

## Matriz de Compatibilidade

- GPT-2/GPT-J/OPT: atenção em `transformer.h.{i}.attn`/`h.{i}.attn`; FFN em `transformer.h.{i}.mlp`/`h.{i}.mlp`.
- LLaMA/Mistral: atenção em `model.layers.{i}.self_attn`; FFN em `model.layers.{i}.mlp`.
- BLOOM: atenção em `transformer.h.{i}.self_attention`; FFN em `transformer.h.{i}.mlp`.
- Genérico: fallback para `transformer.layers.{i}.attention|ffn`, `decoder.layers.{i}.self_attn|mlp`, `encoder.layers.{i}.self_attn|mlp`.

Extração, análise e transplante usam esses padrões via `utils/architecture.py`.

## Troubleshooting

- `--trust-remote-code` (com `--yes`) para permitir código custom de modelos (cautela).
- Quantização 8/4-bit requer bitsandbytes; sem ele, flags são ignoradas (carrega precisão completa).
- `--offline` evita downloads; use paths locais ou cache em `MODEL_CACHE_DIR`.
- Device: `--device cuda|cpu|mps` (default `auto`).
- Captura: `USE_MODEL_OUTPUTS=true` para `hidden_states/attentions` sem hooks/FX.
- Tensors shardados: `.index.json` suportado; `inspect` agrega bytes e parâmetros.

## Checklist de Recursos

- [x] Extração ciente de arquitetura (GPT-2/LLaMA/Mistral/BLOOM/OPT/GPT-J)
- [x] Atenção: MHA, GQA/MQA (mapeamento q_to_kv)
- [x] Armazenamento de pesos: safetensors + .pt shardado com índice
- [x] Captura de ativações: via FX/hooks e outputs; HDF5 groups/dataset; opção por head
- [x] Análise: catálogo de heads, cobertura de embeddings, loaders cientes de shard
- [x] Transplante: injeção de módulo com bridges + gates; salvar/reativar artefatos; injeção ciente de arquitetura
- [x] Validação (online): GLUE (CoLA/STS-B), CoNLL2003 (POS/NER), CommonsenseQA
- [x] Validação (offline): datasets sintéticos determinísticos
- [x] CLI: inspect (tamanhos + parâmetros), reattach, flags globais (offline/seed/json/quiet/verbose)
- [x] CI/Testes: unit offline e smoke de CLI; testes de loaders shardados
- [ ] Famílias estendidas (T5, variantes MoE), hiperparâmetros de router
- [ ] Matriz de CI mais ampla + cache de pip
- [ ] Packaging: migração PEP 621 completa (aposentar setup.py)
- `--yes` para confirmar `--trust-remote-code` (obrigatório por segurança).
- `USE_MODEL_OUTPUTS=true` para capturar via outputs do modelo (sem hooks/FX).
- `STORE_ATTN_PER_HEAD=true` salva atenções por head no dataset-mode.
- `COMPILE_MODEL=true` tenta compilar o modelo (PyTorch 2+).
