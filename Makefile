.PHONY: setup dev test lint type build docker bench space

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

dev:
	pre-commit install || true

test:
	pytest -q

lint:
	ruff check . && black --check .

type:
	mypy polyreact

build:
	python -m build

docker:
	docker build -t polyreact:latest .

bench:
	python -m polyreact.benchmarks.run_benchmarks --config configs/default.yaml

space:
	python space/app.py
