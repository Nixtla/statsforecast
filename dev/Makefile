include ../dev/.help.mk 

build: # Command to build Docker file [optional]
	@docker build -t statsforecast -f dev/Dockerfile .

run: build # Run jupyter notebook using Docker image
	@docker run --rm statsforecast --net=host -v "$(pwd)":/workdir statsforecast

remove: # Deletes statsforecast Docker image
	@docker image rm statsforecast
