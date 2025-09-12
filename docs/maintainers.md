# Maintainers Guide

This guide lists the steps to enable CI/CD and publishing for this repository.

## 1) Enable GitHub Pages (Docs)
- Go to Settings → Pages → Build and deployment → Source: GitHub Actions
- After pushing to `main`, the `Docs Deploy` workflow publishes to:
  https://qrv0.github.io/LLM-Ripper

## 2) Codecov (coverage)
- Create a Codecov account and obtain a repository token
- In GitHub: Settings → Secrets and variables → Actions → New repository secret
  - Name: `CODECOV_TOKEN`
  - Value: your token
- The CI workflow already uploads coverage (XML) via codecov-action

## 3) PyPI releases
- Create an API token on PyPI
- In GitHub: Settings → Secrets and variables → Actions → New repository secret
  - Name: `PYPI_API_TOKEN`
  - Value: `<pypi-AgEN...>`
- Tag a release to publish:
```bash
git tag v1.0.0
git push origin v1.0.0
```

## 4) Branch protection for main
- Settings → Branches → Add rule for `main`
  - Require pull request before merging
  - Require status checks to pass (CI)

## 5) Local quality checks
- Pre-commit: `pre-commit install`
- Lint/format: `make lint` / `make format`
- Tests: `make test` / `make test-cov`
- Docs: `make docs-serve`

