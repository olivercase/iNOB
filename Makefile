.PHONY: install install-dev lint test pipeline geom fem sensors forward viz clean clean-outputs

PYTHON ?= python
CONFIG ?= configs/default.yaml

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e .[dev]

lint:
	$(PYTHON) -m ruff check src tests

test:
	$(PYTHON) -m pytest -q

pipeline:
	$(PYTHON) -m vagus_fm.cli.pipeline --config $(CONFIG)

geom:
	$(PYTHON) -m vagus_fm.cli.build_geom --config $(CONFIG)

fem:
	$(PYTHON) -m vagus_fm.cli.build_fem --config $(CONFIG)

sensors:
	$(PYTHON) -m vagus_fm.cli.generate_sensors --config $(CONFIG)

forward:
	$(PYTHON) -m vagus_fm.cli.run_forward --config $(CONFIG)

viz:
	$(PYTHON) -m vagus_fm.cli.visualise --config $(CONFIG)

clean:
	rm -rf build dist *.egg-info src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache

clean-outputs:
	rm -rf outputs
