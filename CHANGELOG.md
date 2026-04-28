# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — `cleanup/remove-sensitive-features`

This branch consolidates a series of cleanup commits that strip features and
contributions deemed out-of-scope for the public LUMINA release.

### Removed

- **Frontier (OLCF) HPC setup** (`b3bb4fe`)
  - `install/frontier/` (setup_env.sh and historical files)
  - `job_submission_scripts/job-frontier*.sh`
  - `job_submission_scripts/module-to-load-frontier-rocm640.sh`
  - `configs/config.frontier.ddp.yaml`
  - Net: 5 files, 332 deletions. Perlmutter/Polaris `requirements.in` updates
    that were originally bundled in the Frontier PR are preserved.

- **Lagrangian loss family — final cleanup** (`7b6161c`)
  - Removed remaining references in evaluator, scripts, and tests after the
    main Lagrangian removal. 6 files, 85 deletions.

- **Lagrangian loss family — main removal** (`4f8fd1b`)
  - `lumina/model/opf/augmented_lagrangian.py` (1501 lines)
  - `lumina/model/opf/violated_lagrangian.py` (179 lines)
  - Lagrangian-specific paths in `lumina/model/opf/losses.py` (~1700 lines)
  - `configs/loss/augmented_lagrangian.yaml`, `configs/loss/violated_lagrangian.yaml`
  - Lagrangian options removed from per-platform DDP configs and trainer.
  - Associated tests in `tests/model/opf/`.
  - Net: 24 files, ~4200 deletions.

- **SCUC (security-constrained unit commitment) task** (`c9f6426`)
  - `example/scuc/train_scuc.py` (1074 lines)
  - `lumina/dataset/scuc/`, `lumina/evaluator/scuc/`, `lumina/model/scuc/`
  - Net: 9 files, ~1900 deletions.

- **Contingency analysis** (`97b1af0`)
  - `lumina/dataset/opf/contingency.py` (343 lines)
  - Contingency-handling paths in OPF dataset, schema, staging, losses, trainer,
    and data-processing scripts.
  - Contingency-specific tests in `tests/dataset/test_opf.py`.
  - Net: 12 files, ~1250 deletions.

### Reverted earlier (not on `main`)

- **KeunJu Song OPF loss refactors (#66–#68).** Lived only on `kj/dev` and
  `kj/hpo` branches; never merged to `main`. Tracked via the
  `remove-kj-commits` branch for transparency.

### Documentation

- Updated `CLAUDE.md` to drop the Frontier-only contributor entry and the
  Frontier HPC config reference; updated DDP-platform support note accordingly.

### Notes for downstream users

- Any code, configs, or training runs that depend on `augmented_lagrangian`
  / `violated_lagrangian` loss types, on `--task scuc`, on contingency
  features in the OPF dataset, or on Frontier-specific install/job scripts
  must be updated. Use Perlmutter or Polaris configs as references for new
  HPC platforms.
