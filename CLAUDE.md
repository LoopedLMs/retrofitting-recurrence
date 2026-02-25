# Research Code - Looped/Recurrent LLMs from pretrained models

## Philosophy
Research code optimized for rapid iteration and debugging:
- Simple, hackable implementations > frameworks
- Missing error handling is GOOD (faster bug discovery)
- Understand every component > black-box abstractions

## Code Standards
- Type hints on all signatures (modern syntax: `str | None`, `list[int]`)
- Run ruff after changes: `uv run ruff format . && uv run ruff check --fix .`

## Package Management (CRITICAL)
- ALWAYS: `uv add <package>`
- NEVER: manually edit pyproject.toml
- NEVER: `pip install` or `uv pip install`

## Running Code
Python scripts must be run within the uv environment

## Debugging
Check `.venv` source code directly for library implementation details

## Background Knowledge
Paper summaries and research notes live in `./knowledge/`. Check there for context on relevant prior work (e.g. layer redundancy, recurrence retrofitting). The paper behind this repository is summarized in knowledge/summary_retrofitting_recurrence.md.
