.PHONY: install install-dev hooks lint format-check secrets test coverage build ci doctor status gui pipeline geom fem sensors forward viz clean clean-outputs

PYTHON ?= python3
CONFIG ?= configs/default.yaml
INOB = $(PYTHON) -m inob.cli.main

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

# Install the pre-commit hooks (.pre-commit-config.yaml) into .git/hooks.
hooks:
	$(PYTHON) -m pip install -q pre-commit
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check .

# Reports what `ruff format` would change; not a gate (see CONTRIBUTING.md).
format-check:
	$(PYTHON) -m ruff format --check . || true

# Secret scan over the whole history. Needs gitleaks (brew install gitleaks).
secrets:
	gitleaks git . --no-banner --redact

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest -q --cov=inob --cov-report=term-missing:skip-covered

# sdist + wheel, then the metadata check PyPI would run.
build:
	rm -rf dist && $(PYTHON) -m build && $(PYTHON) -m twine check dist/*

# The full local gate: what CI runs, in one target.
ci: lint test build

doctor:
	$(INOB) doctor --config $(CONFIG)

status:
	$(INOB) status --config $(CONFIG)

pipeline:
	$(INOB) run --config $(CONFIG)

geom:
	$(INOB) build-geom --config $(CONFIG)

fem:
	$(INOB) build-fem --config $(CONFIG)

sensors:
	$(INOB) sensors --config $(CONFIG)

forward:
	$(INOB) forward --config $(CONFIG)

viz:
	$(INOB) visualise --config $(CONFIG)

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-outputs:
	rm -rf outputs

# One action to get the whole GUI up: API, web app, browser. Stops both on
# Ctrl-C. See scripts/gui.sh for the interpreter and port handling.
gui:
	./scripts/gui.sh
