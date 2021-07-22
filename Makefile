init: ## initialize environment and install requirements
	sudo apt-get install libmagic1 poppler-utils
	python -m pip install poetry
	python -m poetry install

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	poetry run flake8 plagdef tests

test-ci: ## run tests, GUI tests excluded
	poetry run pytest

test-all-ci: ## run tests on every Python version with tox
	poetry run tox

coverage-ci: ## check code coverage quickly with the default Python
	poetry run coverage erase
	poetry run coverage run --source plagdef --omit="*/test*" -m pytest
	poetry run coverage report -m
	poetry run coverage xml

run: ## starts the CLI
	poetry run app.py
