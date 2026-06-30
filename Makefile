.PHONY: stage1 test lint clean

PYTHON := python
SRC    := src
TESTS  := tests

stage1:
	bash scripts/run_stage1.sh

test:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)
	$(PYTHON) -m ruff format --check $(SRC) $(TESTS)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf outputs/ multirun/ wandb/ .pytest_cache/ htmlcov/ .coverage
