.PHONY: requirements
requirements:
	pip-compile -o requirements.txt pyproject.toml
