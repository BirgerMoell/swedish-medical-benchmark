.PHONY: build
build:
	docker-compose build app

.PHONY: bash
bash:
	docker-compose run --rm app bash

.PHONY: requirements
requirements:
	docker-compose run --rm app uv pip compile -o requirements.txt pyproject.toml
