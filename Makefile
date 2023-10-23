include dev/.help.mk 

SHELL := /bin/bash 
CURR_DIR := $(CURDIR)

START_COMMAND := jupyter-lab --allow-root
JUPYTER_LIST := jupyter-lab list
PIPINSTALLE := pip install -e .

build: # Command to build Docker file [optional]
	@docker build -t statsforecast -f dev/Dockerfile .

run: build # Run jupyter notebook using Docker image
	@docker run --name statsforecast --rm -d --network host -v $(CURR_DIR):/workdir/ statsforecast $(START_COMMAND)
	@docker exec statsforecast $(PIPINSTALLE)

buildless: # Run jupyter notebook using Docker image without building the image
	@docker run --name statsforecast --rm -d --network host -v $(CURR_DIR):/workdir/ statsforecast $(START_COMMAND)
	@docker exec statsforecast $(PIPINSTALLE)

address: # Show the ipaddress and port of Jupyter Notebook 
	@docker exec statsforecast $(JUPYTER_LIST)

stop: # Stops statsforecast container
	@docker stop statsforecast

remove: # Deletes statsforecast Docker image
	@docker image rm statsforecast
