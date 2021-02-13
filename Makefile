init: ## initialize environment and install requirements
	pip install pipenv
	pipenv install --dev

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	pipenv run flake8 plagdef tests

test: ## run tests quickly with the default Python
	pipenv run pytest

test-all: ## run tests on every Python version with tox
	pipenv run tox

coverage: ## check code coverage quickly with the default Python
	pipenv run coverage erase
	pipenv run coverage run --source plagdef -m pytest
	pipenv run coverage report -m
	pipenv run coverage xml

run: ## starts the CLI
	pipenv run cli.py
