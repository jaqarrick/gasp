PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := PYTHONPATH=src $(VENV)/bin/python
IMAGE := gasp
PLATFORM := linux/amd64
ARGS ?=
DOCKER_FLAGS ?=

.PHONY: all build install run test lint lint-fix format pre-commit clean docker-clean docker-run help

$(VENV):
	PYTHON=$(PYTHON) ./scripts/init_python.sh

install: $(VENV)
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	$(VENV)/bin/pre-commit install
	./scripts/install_faial.sh
	./scripts/check_requirements.sh

lint:
	$(PY) -m ruff check src/

lint-fix:
	$(PY) -m ruff check src/ --fix

format:
	$(PY) -m ruff format src/ tests/

pre-commit:
	$(VENV)/bin/pre-commit run --all-files

test:
	$(PY) -m pytest tests/ -v

run:
	$(PY) -m gasp $(ARGS)

# FOR DOCKER
build:
	docker build --platform $(PLATFORM) -t $(IMAGE) .

docker-run:
	docker run --rm --platform $(PLATFORM) $(DOCKER_FLAGS) \
		-v $(PWD)/benchmarks:/workspace/benchmarks \
		-v $(PWD)/hardware:/workspace/hardware \
		$(IMAGE) $(ARGS)

clean:
	rm -rf $(VENV) __pycache__ src/gasp/__pycache__


docker-clean:
	docker rmi $(IMAGE)


help:
	@echo "make install                       - create venv and install deps"
	@echo "make run ARGS='...'              - run locally"
	@echo "make test                        - run tests"
	@echo "make lint                        - lint Python code"
	@echo "make format                      - format Python code"
	@echo "make pre-commit                  - run all pre-commit hooks on all files"
	@echo "make build                       - build Docker image"
	@echo "make docker-run ARGS='...'       - run in Docker"
	@echo "make clean                       - remove build artifacts"
	@echo "make docker-clean                - remove Docker image"
	@echo ""
	@echo "Example:"
	@echo "  make run ARGS='kernel.cu --gpu b200 --verbose'"
	@echo "  make run ARGS='--gpu b200 good.cu poor.cu'"
	@echo "  make run ARGS='--gpu b200 kernels.cu::good kernels.cu::poor'"
	@echo "  make run ARGS='--gpu b200 --config benchmarks/custom/vecadd/vecadd.toml'"
	@echo "  make docker-run DOCKER_FLAGS='--gpus all' ARGS='--gpu b200 --run --config benchmarks/custom/vecadd/vecadd.toml'"
