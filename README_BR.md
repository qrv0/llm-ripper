# LLM Ripper (Português)

Este repositório contém um framework de produção para desconstrução, análise e recomposição modular de conhecimento em LLMs baseados em Transformers.

- Como instalar
  - `pip install -r requirements.txt`
  - `pip install -e .`
- Como usar (CLI)
  - `llm-ripper extract --model "<seu-modelo>" --output-dir ./knowledge_bank`
  - `llm-ripper capture --model "<seu-modelo>" --output-file activations.h5 --dataset wikitext`
  - `llm-ripper analyze --knowledge-bank ./knowledge_bank --output-dir ./analysis`
  - `llm-ripper transplant --source ./knowledge_bank --target "<modelo-alvo>" --output-dir ./transplanted`
  - `llm-ripper validate --model ./transplanted --baseline "<modelo-alvo>" --output-dir ./validation_results`

Atenção: substitua os nomes de modelos de exemplo pelos seus.
