#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = kaxman
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 $(PROJECT_NAME)
	isort --check --diff --profile black $(PROJECT_NAME)
	black --check --config pyproject.toml $(PROJECT_NAME)

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(PROJECT_NAME)

test:
	coverage run -m pytest ./tests

coverage: test
	coverage report --fail-under=90