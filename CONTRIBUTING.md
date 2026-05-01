# Contributing to lumina-sdk

Thanks for your interest in contributing! This guide covers the basics: dev setup, branching, testing, and the PR flow.

## Dev setup

`lumina-sdk` requires Python 3.10+ and an existing PyTorch / PyG install for your platform.

```bash
# 1. Clone
git clone https://github.com/argonne-gridfm/lumina-sdk.git
cd lumina-sdk

# 2. Create + activate a venv (or conda env)
python -m venv .venv
source .venv/bin/activate

# 3. Install PyTorch + PyG following their official instructions
#    https://pytorch.org/get-started/locally/
#    https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# 4. Install lumina-sdk in editable mode with dev extras
pip install -e ".[dev]"
```

`[dev]` self-references `[test,acopf]`, so you get pytest + pandapower + the rest of the ACOPF stack in one shot.

## Branches and commits

- Branch off `main` for any change: `git checkout -b <topic>/<short-name>` (e.g. `fix/ddp-launcher`, `docs/notebook-04`).
- Keep commits focused; squash noise locally before opening the PR.
- Commit messages should describe the *why*, not just the *what*. The first line is the title (≤72 chars), then a blank line, then a longer body if needed.

## Testing

```bash
# Full suite
pytest tests/

# A single file
pytest tests/dataset/test_opf.py

# A single test
pytest tests/dataset/test_opf.py::test_opf_dataset_loads_case14_and_exposes_expected_schema -v
```

Some tests download case14 from Google Cloud Storage on first run (~38 s). A handful of snapshot-dependent tests skip gracefully when their fixtures aren't in the repo.

If your change touches a public API, **add or update a test** that exercises it.

## Documentation

User-facing docs live under `docs/` and are rendered with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

```bash
pip install -e ".[doc]"
mkdocs serve   # http://localhost:8000
```

API reference pages (`docs/api/*.md`) auto-generate from docstrings via [`mkdocstrings`](https://mkdocstrings.github.io/), so updating a docstring updates the docs.

When adding tutorials or notebooks, follow the existing patterns:

- Use the `<UPPERCASE_PLACEHOLDER>` style for paths / accounts the user must substitute.
- Prefix any account/path-substitution blocks with a `!!! note` admonition.
- Don't commit anyone's personal paths, conda env paths, account names, or API tokens.

## Pull requests

1. Open the PR against `main`.
2. Fill out the PR template (Summary / Type / Test plan / Breaking changes / Checklist).
3. Make sure CI passes (when CI lands).
4. At least one maintainer review is required to merge.
5. If your change is user-visible, add a one-liner to `CHANGELOG.md` under an `## [Unreleased]` heading.

## Code style

No linter is enforced. Match the style of nearby code:

- 4-space indents, no trailing whitespace.
- Avoid adding new top-level dependencies without discussion — most extras (acopf, hps, doc, hf) live in `pyproject.toml` `[project.optional-dependencies]`.
- Don't introduce `lambda`s or closures into anything that needs to be picklable (DDP workers, multiprocessing).

## Reporting issues

Please use the [issue templates](https://github.com/argonne-gridfm/lumina-sdk/issues/new/choose). Bug reports without a reproduction or environment dump are hard to act on.

## License

By contributing, you agree your contributions are licensed under the project's [LICENSE](LICENSE).
