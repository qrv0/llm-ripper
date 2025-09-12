# Beginner's Guide (Start Here)

Welcome! This guide is for developers new to Python ML projects. You'll get a working demo in minutes, no GPU required.

## Prerequisites
- Python 3.8–3.11
- Git (optional)

## 1) Set up your environment

```bash
# inside your project folder
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## 2) Run the beginner demo (no downloads)

Option A (Makefile):
This creates a sample run folder with the files the Studio viewer expects.

```bash
make beginner
```

Option B (CLI only):

```bash
llm-ripper quickstart --open
```

The command will:
- Generate a run under `runs/<timestamp>/` with demo JSON files
- Launch the Studio viewer at http://localhost:8000

If the page shows empty panels or error messages, that's okay — you can still explore the layout and JSON files.

## 3) Next steps
- Try the offline smoke test: `make smoke-offline`
- Explore CLI help: `python -m llm_ripper.cli --help`
- Read the Quickstart for full pipeline steps

## Troubleshooting
- If `mkdocs` or `ruff` commands are missing, install dev deps: `pip install -r requirements-dev.txt`
- If ports are busy, change the Studio port: `make studio PORT=8001`

You're set! As you gain confidence, switch from the demo to actual models and data.
