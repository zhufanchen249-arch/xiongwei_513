This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Dev Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
# System dependencies for tests (Ubuntu/Debian)
sudo apt-get install -y build-essential git curl libglib2.0-0 libegl1-mesa-dev ffmpeg libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev
```

## Test & Validate

**Command order:** lint/format → typecheck → test

- **Pre-commit** (install via `pre-commit install`): runs ruff-format, ruff, mypy, typos, pyupgrade, bandit, gitleaks, zizmor, prettier. Manual run: `pre-commit run --all-files`
- **CI tiers**: runs the full test suite in 4 tiers (base → dataset → hardware → viz), each adding more extras. Tests skip automatically via `is_package_available()` guards in `tests/utils.py`.
- **Device**: E2E tests (Makefile) use `DEVICE` (default `cpu`). Regular pytest uses `LEROBOT_TEST_DEVICE` env var (from `tests/utils.py:26`); defaults to auto-detected device.

```bash
uv run pytest tests -svv --maxfail=10                          # All tests
uv run pytest tests/test_specific_file.py -svv                 # Single test file
uv run pytest tests/policies/ -svv                             # Single test directory
uv run ruff check src/lerobot/                                 # Lint only
uv run ruff format src/lerobot/                                # Format only
uv run mypy src/lerobot/envs/                                  # Typecheck strict modules only
DEVICE=cuda make test-end-to-end                              # All E2E (writes to tests/outputs/)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (tiered: base → dataset → hardware → viz, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags), `claude.yml` (Claude Code review via @claude mentions).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`). New imports for optional packages must be guarded or lazy. See `pyproject.toml [project.optional-dependencies]`.
- **Non-exposed extras**: `vlabench`, `robomme`, `robocasa` are not installable via `lerobot[extra]` due to dependency conflicts. Install manually per `docs/source/` guides.
- **Test fixtures**: conditional loading in `tests/conftest.py` via `pytest_plugins` — fixtures that depend on optional packages (e.g. `datasets`) are only registered when the package is available.
- **Gymnasium**: Core dependency (not optional) due to tight coupling with envs, policies, robots, and scripts.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Ruff config**: Line length 110, double quotes, Google docstring convention, target Python 3.12. Matches `pyproject.toml [tool.ruff]` settings.
- **Pre-commit**: Hooks run ruff-format, ruff, mypy, typos, pyupgrade, bandit, gitleaks, zizmor, prettier. Install with `pre-commit install`.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
