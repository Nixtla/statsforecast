IMAGE := arima
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init:
	docker build -t ${IMAGE} .

run_module: .require-module
	docker run -i --rm ${DOCKER_PARAMETERS} \
		${IMAGE} ${module}

run_python:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} python

run_R:
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE} R

.PHONY: run_tests_statsforecast
run_tests_statsforecast:
	echo "Running StatsForecast Python experiments..."
	@for group in Daily Hourly Weekly; do \
		echo "Running Python experiment: model=statsforecast, group=$$group"; \
		$(MAKE) run_module module="python -m src.statsforecast --dataset M4 --group $$group"; \
	done

.PHONY: run_tests_pmdarima
run_tests_pmdarima:
	@echo "Running PMDARIMA Python experiments..."
	@for group in Daily Hourly Weekly; do \
		echo "Running Python experiment: model=pmdarima, group=$$group"; \
		$(MAKE) run_module module="python -m src.pmdarima --dataset M4 --group $$group"; \
	done

.PHONY: run_tests_prophet
run_tests_prophet:
	@echo "Running Prophet Python experiments..."
	@for group in Daily Hourly Weekly; do \
		echo "Running Python experiment: model=prophet, group=$$group"; \
		$(MAKE) run_module module="python -m src.prophet --dataset M4 --group $$group"; \
	done

.PHONY: run_tests_r
run_tests_r: data_prep_r
	@echo "\nRunning all R experiments..."
	@for group in Daily Hourly Weekly; do \
		echo "Running R experiment (forecast package): group=$$group"; \
		$(MAKE) run_module module="Rscript src/arima_forecast_r.R --dataset M4 $$group"; \
		echo "Running R experiment (fable package): group=$$group"; \
		$(MAKE) run_module module="Rscript src/arima_fable_r.R --dataset M4 $$group"; \
	done

.PHONY: data_prep_r
data_prep_r:
	@echo "\nPreparing data for R experiments..."
	@for group in Daily Hourly Weekly; do \
		echo "Preparing data for R experiment: group=$$group"; \
		$(MAKE) run_module module="python -m src.data --dataset M4 --group $$group"; \
	done

.PHONY: run_all_tests
run_all_tests: run_tests_statsforecast run_tests_pmdarima run_tests_prophet run_tests_r
	@echo "\nAll experiments completed!"


.PHONY: run_evaluation
run_evaluation:
	$(MAKE) run_module module="python -m src.evaluation";

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

bash_docker: 
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

.require-module:
ifndef module
	$(error module is required)
endif
