# Check for .venv and use it if available
ifneq (,$(wildcard .venv/bin/python))
    PYTHON = .venv/bin/python
    PIP = .venv/bin/pip
else
    PYTHON ?= python
    PIP ?= pip
endif

# Default versions (fallback if PyTorch is not installed yet)
DEFAULT_TORCH := 2.8.0
DEFAULT_CUDA := cu128

# Try to detect system versions
DETECTED_TORCH := $(shell $(PYTHON) -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
DETECTED_CUDA := $(shell $(PYTHON) -c "import torch; v=torch.version.cuda; print('cu' + v.replace('.', '') if v else 'cpu')" 2>/dev/null)
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(sys.version.split()[0])")

# Use detected versions if available, otherwise use defaults
TORCH_VERSION := $(if $(DETECTED_TORCH),$(DETECTED_TORCH),$(DEFAULT_TORCH))
CUDA_VERSION := $(if $(DETECTED_CUDA),$(DETECTED_CUDA),$(DEFAULT_CUDA))

PYG_URL := https://data.pyg.org/whl/torch-$(TORCH_VERSION)+$(CUDA_VERSION).html

.PHONY: help install clean info dev install-test install-dev install-acopf install-hps install-doc install-benchmark install-all

help:
	@echo "LUMINA-CORE Makefile"
	@echo "===================="
	@echo "Usage: make [target]"
	@echo ""
	@echo "Core Targets:"
	@echo "  install            : Install the package (standard mode)"
	@echo "  dev                : Install the package in editable mode (for development)"
	@echo "  clean              : Remove build artifacts and cache"
	@echo "  info               : Show detected Python/Torch/CUDA configuration"
	@echo "  help               : Show this help message"
	@echo ""
	@echo "Optional Dependency Targets:"
	@echo "  install-test       : Install testing dependencies (pytest)"
	@echo "  install-dev        : Install development tools (ipykernel)"
	@echo "  install-acopf      : Install ACOPF solvers (pandapower, pypower, etc.)"
	@echo "  install-hps        : Install hyperparameter search tools (wandb, optuna)"
	@echo "  install-doc        : Install documentation tools (sphinx)"
	@echo "  install-benchmark  : Install benchmarking tools (lightning)"
	@echo "  install-all        : Install ALL optional dependencies"

info:
	@echo "Configuration:"
	@echo "  Python: $(PYTHON) ($(PYTHON_VERSION))"
	@echo "  Torch:  $(TORCH_VERSION) (Detected: $(if $(DETECTED_TORCH),Yes,No))"
	@echo "  CUDA:   $(CUDA_VERSION) (Detected: $(if $(DETECTED_CUDA),Yes,No))"
	@echo "  pyg-lib URL:    $(PYG_URL)"

install:
	@echo "Installing lumina-core with dependencies from $(PYG_URL)..."
	$(PIP) install . -f $(PYG_URL)

# Install in editable mode for development
dev:
	@echo "Installing lumina-core in editable mode..."
	$(PIP) install -e . -f $(PYG_URL)

# Optional dependencies targets
install-test:
	@echo "Installing test dependencies..."
	$(PIP) install ".[test]" -f $(PYG_URL)

install-dev:
	@echo "Installing dev dependencies..."
	$(PIP) install ".[dev]" -f $(PYG_URL)

install-acopf:
	@echo "Installing ACOPF dependencies..."
	$(PIP) install ".[acopf]" -f $(PYG_URL)

install-hps:
	@echo "Installing HPS dependencies..."
	$(PIP) install ".[hps]" -f $(PYG_URL)

install-doc:
	@echo "Installing documentation dependencies..."
	$(PIP) install ".[doc]" -f $(PYG_URL)

install-benchmark:
	@echo "Installing benchmark dependencies..."
	$(PIP) install ".[benchmark]" -f $(PYG_URL)

install-all:
	@echo "Installing ALL dependencies..."
	$(PIP) install ".[all]" -f $(PYG_URL)

clean:
	rm -rf build/ dist/ *.egg-info
	find . -name "__pycache__" -type d -exec rm -rf {} +
