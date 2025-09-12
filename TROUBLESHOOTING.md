# Troubleshooting

## Installation problems
- Ensure a clean virtual environment and Python 3.8–3.11.
- On GPUs, install torch with the appropriate CUDA index URL (see Makefile install-cuda).
- If optional extras fail (spacy, wandb), install them via extras: `pip install .[spacy,wandb,viz]`.

## Runtime errors
- Use `--log-format json` and `--verbose` to get more context.
- If loading remote models, avoid `--trust-remote-code` unless strictly necessary.
- Run with fewer samples to isolate issues and share `--seed` for reproducibility.

## Tests
- Run `make test` or `tox` to validate your environment.

If issues persist, open a GitHub Issue with logs and environment details.
