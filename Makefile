DOCKER := $(shell command -v docker 2> /dev/null)
TAG := optbnn

PWD := $(shell pwd)
##
docker:
ifndef DOCKER
   	$(error Docker not available. Please install Docker)
endif
	@printf ''

## Create image
image:
	docker build --tag $(TAG) .

## Run interactive
run-terminal:
	docker run --rm -it $(TAG)

run-jupyter:
	docker run -i -t --rm -p 8888:8888 $(TAG) /bin/bash -c "/opt/conda/bin/jupyter lab --ip='*' --port=8888 --no-browser"

run-jupyter-persistent:
	docker run -i -t --rm -p 8888:8888 -v $(PWD):/optbnn  $(TAG) /bin/bash -c "/opt/conda/bin/jupyter lab --ip='*' --port=8888 --no-browser"

run-python:
	docker run --rm -it $(TAG) ipython


remove:
	docker image rm $(TAG)