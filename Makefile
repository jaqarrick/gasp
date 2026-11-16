PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := PYTHONPATH=src $(VENV)/bin/python
IMAGE := gasp
PLATFORM := linux/amd64
ARGS ?=
DOCKER_FLAGS ?=
GPU ?=
VERSION ?=
ARCH ?=
BMKS ?=
DRY_RUN ?=
BATCH ?=

.PHONY: all build install run test lint lint-fix format pre-commit clean docker-clean docker-run help microbenchmarks microbenchmarks-hpmll microbenchmarks-gpu-benches check_spec ablation batch-experiment batch-results

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
	@$(PY) -m gasp $(ARGS)

ablation:
	$(PY) experiments/ablation/run_ablation.py $(ARGS)

# DRY_RUN=1 only predicts (no GPU or ncu); PROFILE=<path> uses that profile instead of GPU's built-in one;
# DESC='<text>' gives the experiment description instead of asking for it
batch-experiment:
	@test -n "$(GPU)" || { echo "usage: make batch-experiment GPU=<gpu> [BMKS='mm vecadd'] [DRY_RUN=1] [PROFILE=<path>] [DESC='<text>']"; exit 1; }
	./scripts/batch_experiment.sh $(if $(DRY_RUN),--dry-run) $(if $(PROFILE),--profile $(PROFILE)) $(if $(DESC),--description "$(DESC)") $(GPU) $(BMKS)

batch-results:
	@test -n "$(BATCH)" || { echo "usage: make batch-results BATCH=experiments/results/e2e/<gpu>/batch_<timestamp> [ARGS='--csv --model all']"; exit 1; }
	@$(PYTHON) scripts/read_batch.py $(ARGS) $(BATCH)

# The complete profile: HPMLL (its valid fields), gpu-stream (DRAM bandwidth) and gasp-bm (the
# rest, and HPMLL's fallback), all at base clock; needs a GPU, nvcc, ncu and both submodules.
microbenchmarks:
	@test -n "$(GPU)" -a -n "$(VERSION)" || { echo "usage: make microbenchmarks GPU=<gpu> VERSION=<label> [ARCH=sm_XX] [ARGS='--only hpmll,... --force']"; exit 1; }
	$(PYTHON) hardware/microbenchmarks/gasp-bm/run_all.py --gpu $(GPU) --version $(VERSION) $(if $(ARCH),--arch $(ARCH)) $(ARGS)

microbenchmarks-hpmll:
	@test -n "$(GPU)" -a -n "$(VERSION)" || { echo "usage: make microbenchmarks-hpmll GPU=<gpu> VERSION=<label> [ARCH=sm_XX] [ARGS='--only ... --reps N --force']"; exit 1; }
	$(PYTHON) scripts/run/run_hpmll_bmks.py --gpu $(GPU) --version $(VERSION) $(if $(ARCH),--arch $(ARCH)) $(ARGS)

# DRAM bandwidth from gpu-benches' gpu-stream, written as a run JSON for `make microbenchmarks`
# (which also calls this), plus a comparison with the profiles.
microbenchmarks-gpu-benches:
	@test -n "$(GPU)" -a -n "$(VERSION)" || { echo "usage: make microbenchmarks-gpu-benches GPU=<gpu> VERSION=<label> [ARCH=sm_XX] [ARGS='--force']"; exit 1; }
	$(PYTHON) scripts/run/run_gpu_benches.py --gpu $(GPU) --version $(VERSION) $(if $(ARCH),--arch $(ARCH)) $(ARGS)

# Compare a profile's spec fields with this machine's GPU; needs nvcc.
check_spec:
	./scripts/check_gpu_spec.sh $(PROFILE)

# FOR DOCKER
build:
	docker build --platform $(PLATFORM) -t $(IMAGE) .

docker-run:
	@docker run --rm --platform $(PLATFORM) $(DOCKER_FLAGS) \
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
	@echo "make ablation ARGS='...'         - run the ablation from a saved metrics file"
	@echo "make test                        - run tests"
	@echo "make batch-experiment GPU=.. [BMKS='..'] [DRY_RUN=1] [PROFILE=..] [DESC='..'] - e2e accuracy for every benchmarks/*/configs/*.toml"
	@echo "make batch-results BATCH=.. [ARGS='--csv'] - table of a batch's launches, T_GPU and speedups (GASPerf only unless ARGS='--model all')"
	@echo "make microbenchmarks GPU=.. VERSION=.. - run HPMLL, gpu-stream and gasp-bm, merge into one profile"
	@echo "make microbenchmarks-hpmll GPU=.. VERSION=.. - run the HPMLL benchmarks that match profile fields, build a profile"
	@echo "make microbenchmarks-gpu-benches GPU=.. VERSION=.. - DRAM bandwidth from gpu-stream, compare with the profiles"
	@echo "make check_spec PROFILE=..       - compare a profile's spec fields with this machine's GPU"
	@echo "make lint                       - lint Python code"
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
	@echo "  make run ARGS='--gpu b200 --config benchmarks/vecadd/vecadd.toml'"
	@echo "  make run ARGS='--profile hardware/profiles/rtx2080ti.json kernel.cu'"
	@echo "  make run ARGS='--gpu rtx2080ti --run --config benchmarks/vecadd/vecadd.toml --save-metrics vecadd.metrics.json'"
	@echo "  make run ARGS='--gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml --metrics vecadd.metrics.json --override cache=profiled'"
	@echo "  make ablation ARGS='--gpu rtx2080ti --config benchmarks/vecadd/vecadd.toml --metrics vecadd.metrics.json'"
	@echo "  make docker-run DOCKER_FLAGS='--gpus all' ARGS='--gpu b200 --run --config benchmarks/vecadd/vecadd.toml'"
