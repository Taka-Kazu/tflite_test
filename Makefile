.PHONY: build run

build:
	docker-compose build

run:
	docker-compose run --rm tensorflow
