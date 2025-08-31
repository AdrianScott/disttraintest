# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Core code — `model.py`, `layers.py`, `train.py`, `train_ddp.py`, `generate.py`, and utilities under `src/utils/`.
- `data/`: Data tools — `prepare_data.py` trains the tokenizer and writes to `artifacts/`.
- `scripts/`: Operational helpers — `train_ddp.sh` (training), `test_ddp.sh` (DDP comms check).
- `tests/`: PyTest suite (e.g., `tests/test_model.py`).
- Runtime dirs: `artifacts/`, `runs_ddp/`, `logs/` (created by scripts).
- Infra: `Dockerfile`, `requirements.txt`, `README.md`.

## Build, Test, and Development Commands
- Install: `pip install -r requirements.txt`
- Dev tools: `pip install -r requirements-dev.txt` (adds pytest, Ruff, Black)
- Pre-commit (optional): `pip install pre-commit && pre-commit install`; run all: `pre-commit run -a`
- Prepare data: `python -m data.prepare_data`
- Unit tests: `pytest -q`
- Local train (single GPU): `python src/train.py`
- DDP comm test: `bash scripts/test_ddp.sh` (set `N_PROCS_PER_NODE`, `NNODES`, etc.)
- DDP train: `bash scripts/train_ddp.sh` (e.g., `N_PROCS_PER_NODE=4 NNODES=1 ...`)
- Generate text: `python src/generate.py --checkpoint runs_ddp/latest/model_final.pt --prompt "Hello"`

Make targets (optional): `make install`, `make prepare-data`, `make test`, `make lint`, `make fmt`, `make ddp-test N_PROCS_PER_NODE=4`, `make ddp-train NNODES=2 NODE_RANK=0`.

## Coding Style & Naming Conventions
- Python 3.10+, PEP 8, 4‑space indentation, trailing commas where sensible.
- Names: modules/functions `snake_case`, classes `CamelCase`, constants `UPPER_SNAKE_CASE`.
- Prefer type hints for public APIs and clear docstrings.
- No enforced formatter/linter in repo; match existing style. Optional: run Black/Ruff locally before PRs.

## Testing Guidelines
- Framework: PyTest. Place tests under `tests/` named `test_*.py`.
- Add tests for new logic (e.g., tensor shapes, causal masking, basic flows). Keep tests deterministic and fast; avoid network calls.
- Run locally with `pytest -q`. Add fixtures when touching model configs.

## Commit & Pull Request Guidelines
- Commits: short, imperative, and focused (seen in history: “improve logging; add ddp diagnostics”). Group related changes.
- PRs must include: clear description, linked issues, how to run (key flags/env), and evidence (logs/screenshots) for DDP flows. Update README/scripts if interfaces change.

## Security & Configuration Tips
- Do not commit credentials. Configure W&B and cluster details via environment variables.
- DDP/NCCL envs commonly used: `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`, `N_PROCS_PER_NODE`, `NCCL_SOCKET_IFNAME`.
- Prefer the Docker workflow in `README.md` for reproducibility.
