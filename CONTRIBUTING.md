# Contributing to LLM Ripper

Thanks for your interest in contributing! Please follow these guidelines.

## Development setup

- Python >= 3.8
- Create and activate a virtualenv
- Install dependencies and package in editable mode:
  - pip install -r requirements.txt
  - pip install -r requirements-dev.txt
  - pip install -e .
- Install pre-commit hooks:
  - pre-commit install
- Quality checks:
  - make lint
  - make test
  - make test-cov
- Optional: tox for multi-Python matrix:
  - tox

## Pull requests

- Keep PRs small and focused
- Ensure CI passes (ruff, mypy, pytest, docs build)
- Add/update tests with behavior changes
- Update documentation where applicable
- Prefer conventional commits: feat:, fix:, docs:, chore:

## Code style

- Ruff for linting, black/ruff-format for formatting
- Mypy for type checking

## Security

- Be cautious with `--trust-remote-code`; preserve safety prompts and flags in model loading

## Reporting issues
- Use GitHub Issues, include:
  - Version (`pip show llm-ripper`), OS, Python version
  - Steps to reproduce, expected vs actual
  - Logs/tracebacks

## License
By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
