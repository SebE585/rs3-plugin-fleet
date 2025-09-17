.PHONY: setup lint test fmt run simulate simulate-vl1 simulate-veh

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip
	. .venv/bin/activate && pip install -e .[dev]

lint:
	. .venv/bin/activate && ruff check src tests

fmt:
	. .venv/bin/activate && ruff format src tests

test:
	. .venv/bin/activate && pytest -q

run:
	. .venv/bin/activate && python -m rs3_plugin_fleet.runner.run_fleet --config src/rs3_plugin_fleet/config/coin-coin-delivery.yaml

simulate:
	@set -e; \
		test -d .venv && . .venv/bin/activate || true; \
		export PYTHONPATH=$$PWD:$$PWD/../RoadSimulator3; \
		python -m rs3_plugin_fleet.runner.run_fleet --config src/rs3_plugin_fleet/config/coin-coin-delivery.yaml

simulate-vl1:
	@set -e; \
		test -d .venv && . .venv/bin/activate || true; \
		export PYTHONPATH=$$PWD:$$PWD/../RoadSimulator3; \
		python -m rs3_plugin_fleet.runner.run_fleet --config src/rs3_plugin_fleet/config/coin-coin-delivery.yaml --vehicle-id CCD-VL-01

# Generic: make simulate-veh VEH=CCD-VL-01
simulate-veh:
	@set -e; \
		test -d .venv && . .venv/bin/activate || true; \
		export PYTHONPATH=$$PWD:$$PWD/../RoadSimulator3; \
		python -m rs3_plugin_fleet.runner.run_fleet --config src/rs3_plugin_fleet/config/coin-coin-delivery.yaml --vehicle-id $${VEH}