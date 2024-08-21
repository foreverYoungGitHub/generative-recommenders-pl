
format:
	black src
	ruff check src --fix
	isort src

lint:
	ruff check src
	isort --check-only --diff --filter-files src
	black --check src

style: lint format
