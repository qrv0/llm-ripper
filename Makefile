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

.PHONY: help
help:
	@echo "Targets: install, install-cuda, lint, format, test, extract, capture, analyze, transplant, validate"
	@echo "Vars: MODEL, OUT, ACT, ANALYSIS, TRANSPLANTED, VALIDATION, DEVICE (auto|cuda|cpu|mps), E8=1, E4=1, TRUST=1"

.PHONY: install
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

.PHONY: install-dev
install-dev: install
	$(PIP) install -r requirements-dev.txt
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

.PHONY: test
test:
	pytest -q

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
