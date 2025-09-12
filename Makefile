# Simple developer Makefile for LLM Ripper
# You can override variables on the command line, e.g.:
#   make extract MODEL=./models/model OUT=./knowledge_bank DEVICE=cuda E8=1

PY=python
PIP=pip
MODEL?=./models/model
OUT?=./knowledge_bank
ACT?=./activations.h5
ANALYSIS?=./analysis
TRANSPLANTED?=./transplanted
VALIDATION?=./validation_results
DEVICE?=auto
E8?=0
E4?=0
TRUST?=0

BITS8=$(if $(filter 1,$(E8)),--load-in-8bit,)
BITS4=$(if $(filter 1,$(E4)),--load-in-4bit,)
TRUST_FLAG=$(if $(filter 1,$(TRUST)),--trust-remote-code,)
DEVICE_FLAG=--device $(DEVICE)
RUN_ROOT?=
LATEST:=$(shell ls -1dt runs/* 2>/dev/null | head -n 1)

.PHONY: help
help:
	@echo "Targets: install, install-cuda, lint, format, lint-fix, test, test-cov, docs-build, docs-serve, extract, capture, analyze, transplant, validate, inspect, smoke-offline, demo-full, demo-trace, demo-uq, studio, print-latest, precommit"
	@echo "Vars: MODEL, OUT, ACT, ANALYSIS, TRANSPLANTED, VALIDATION, DEVICE (auto|cuda|cpu|mps), E8=1, E4=1, TRUST=1, RUN_ROOT=<runs/...>"

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install
	$(PIP) install -r requirements-dev.txt
	pre-commit install || true

.PHONY: dev-setup
dev-setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -e .
	pre-commit install || true

# Adjust the CUDA version as needed
.PHONY: install-cuda
install-cuda:
	$(PIP) install --upgrade pip
	$(PIP) install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

.PHONY: lint
lint:
	ruff check .
	mypy src || true

.PHONY: format
format:
	black .
	ruff check --fix .

.PHONY: docs-build
docs-build:
	mkdocs build --strict

.PHONY: docs-serve
docs-serve:
	mkdocs serve -a 0.0.0.0:8000

.PHONY: lint-fix
lint-fix:
	ruff check --fix .
	black .

.PHONY: test
test:
	pytest -q

.PHONY: test-cov
test-cov:
	pytest --cov=src --cov-report=term-missing -q

.PHONY: release-dry-run
release-dry-run:
	python -m build || true
	python -m pip install --upgrade twine || true
	twine check dist/* || true
	@echo "Release dry run complete."

.PHONY: extract
extract:
	$(PY) -m llm_ripper.cli extract --model $(MODEL) --output-dir $(OUT) $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: capture
capture:
	$(PY) -m llm_ripper.cli capture --model $(MODEL) --output-file $(ACT) --dataset wikitext --max-samples 64 $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: capture-nodl
capture-nodl:
	$(PY) -m llm_ripper.cli capture --model $(MODEL) --output-file $(ACT) --max-samples 32 $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: analyze
analyze:
	$(PY) -m llm_ripper.cli analyze --knowledge-bank $(OUT) --activations $(ACT) --output-dir $(ANALYSIS) $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: transplant
transplant:
	$(PY) -m llm_ripper.cli transplant --source $(OUT) --target $(MODEL) --output-dir $(TRANSPLANTED) --strategy module_injection --source-component embeddings --target-layer 0 $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: validate
validate:
	$(PY) -m llm_ripper.cli validate --model $(TRANSPLANTED) --baseline $(MODEL) --output-dir $(VALIDATION) $(DEVICE_FLAG) $(BITS8) $(BITS4) $(TRUST_FLAG)

.PHONY: inspect
inspect:
	$(PY) -m llm_ripper.cli inspect --knowledge-bank $(OUT) --json

.PHONY: smoke-offline
smoke-offline:
	@echo "[smoke] Building minimal KB structure in $(OUT) and running inspect"
	@mkdir -p $(OUT)/embeddings $(OUT)/heads/layer_0 $(OUT)/ffns/layer_0
	@python - << 'PY'
import json, os, torch
out=os.environ.get('OUT','./knowledge_bank')
emb_cfg={"dimensions":[4,3],"vocab_size":4,"hidden_size":3}
open(f"{out}/embeddings/config.json","w").write(json.dumps(emb_cfg))
torch.save(torch.randn(4,3), f"{out}/embeddings/embeddings.pt")
open(f"{out}/extraction_metadata.json","w").write(json.dumps({"source_model":"dummy"}))
open(f"{out}/heads/layer_0/config.json","w").write(json.dumps({"layer_idx":0}))
open(f"{out}/ffns/layer_0/config.json","w").write(json.dumps({"layer_idx":0}))
PY
	$(PY) -m llm_ripper.cli inspect --knowledge-bank $(OUT)

.PHONY: precommit
precommit:
	pre-commit run --all-files || true

.PHONY: demo-full
demo-full:
	$(PY) examples/run_full_pipeline.py --model $(MODEL) --baseline $(MODEL)

.PHONY: demo-trace
demo-trace:
	$(PY) -m llm_ripper.cli trace --model $(MODEL) --targets head:0.q,ffn:0.up --metric nll_delta --intervention zero --max-samples 32 --seed 42

.PHONY: demo-uq
demo-uq:
	$(PY) -m llm_ripper.cli uq --model $(MODEL) --samples 8 --max-texts 32

.PHONY: studio
studio:
	@echo "Using run root: $(or $(RUN_ROOT),$(LATEST))"
	$(PY) -m llm_ripper.cli studio --root $(or $(RUN_ROOT),$(LATEST)) --port $(or $(PORT),8000)

.PHONY: beginner
beginner:
	$(PY) examples/beginner_quickstart.py
	$(MAKE) studio

.PHONY: print-latest
print-latest:
	@echo $(LATEST)
