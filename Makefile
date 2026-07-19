.PHONY: install install-dev lint test doctor status pipeline geom fem sensors forward viz clean clean-outputs

PYTHON ?= python
CONFIG ?= configs/default.yaml
INOB = $(PYTHON) -m inob.cli.main

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q

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
