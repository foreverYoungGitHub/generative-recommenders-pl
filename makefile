
format:
	ruff check src tests --fix
	isort src tests

lint:
	ruff check src tests
	isort --check-only --diff --filter-files src tests

style: lint format

test:
	coverage erase
	coverage run --source=src/ -m pytest tests --durations=10 -vv
	coverage report --format=markdown
