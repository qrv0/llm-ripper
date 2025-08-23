# Contributing to LLM Ripper

Thanks for your interest in contributing! Please follow these guidelines.

## Getting started
- Fork the repo and create your branch from `main`.
- Ensure Python 3.8+.
- Create a virtualenv and install dependencies: `pip install -r requirements.txt -e .`.

## Development workflow
- Run linters and tests before submitting a PR.
- Add/adjust unit tests for new features and bug fixes.
- Update docs (README and examples) when behavior or interfaces change.

## Coding standards
- Follow PEP8.
- Type hints required for new/changed public APIs.
- Prefer small, focused PRs.

## Commit / PR checklist
- [ ] Feature or bugfix has tests
- [ ] Docs updated
- [ ] CI passing

## Reporting issues
- Use GitHub Issues, include:
  - Version (`pip show llm-ripper`), OS, Python version
  - Steps to reproduce, expected vs actual
  - Logs/tracebacks

## License
By contributing, you agree that your contributions will be licensed under the Apache-2.0 License.
