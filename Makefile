PY ?= python
PIP ?= pip

# DDP defaults (override at call site, e.g., make ddp-train N_PROCS_PER_NODE=4)
N_PROCS_PER_NODE ?= 1
NNODES ?= 1
NODE_RANK ?= 0
MASTER_ADDR ?= 127.0.0.1
MASTER_PORT ?= 29500

.PHONY: install prepare-data test lint fmt ddp-test ddp-train precommit generate

install:
	$(PIP) install -r requirements.txt
	-$(PIP) install -r requirements-dev.txt

prepare-data:
	$(PY) -m data.prepare_data

test:
	pytest -q

lint:
	ruff .

fmt:
	black .
	ruff --fix .

precommit:
	pre-commit run -a

ddp-test:
	N_PROCS_PER_NODE=$(N_PROCS_PER_NODE) \
	NNODES=$(NNODES) \
	NODE_RANK=$(NODE_RANK) \
	MASTER_ADDR=$(MASTER_ADDR) \
	MASTER_PORT=$(MASTER_PORT) \
	bash scripts/test_ddp.sh

ddp-train:
	N_PROCS_PER_NODE=$(N_PROCS_PER_NODE) \
	NNODES=$(NNODES) \
	NODE_RANK=$(NODE_RANK) \
	MASTER_ADDR=$(MASTER_ADDR) \
	MASTER_PORT=$(MASTER_PORT) \
	bash scripts/train_ddp.sh

# Example: make generate CHECKPOINT=runs_ddp/latest/model_final.pt PROMPT="Hello"
CHECKPOINT ?= runs_ddp/latest/model_final.pt
PROMPT ?= Hello
generate:
	$(PY) src/generate.py --checkpoint "$(CHECKPOINT)" --prompt "$(PROMPT)"

