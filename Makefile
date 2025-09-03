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

load_docs_scripts:
	if [ ! -d "docs-scripts" ] ; then \
		git clone -b scripts https://github.com/Nixtla/docs.git docs-scripts --single-branch; \
	fi

api_docs:
	cd python && lazydocs .statsforecast --no-watermark  --output-path ../docs
	python docs/to_mdx.py

examples_docs:
	mkdir -p nbs/_extensions
	cp -r docs-scripts/mintlify/ nbs/_extensions/mintlify
	quarto render nbs/docs --output-dir ../docs/mintlify/
	quarto render nbs/src --output-dir ../docs/mintlify/
	quarto render nbs/blog --output-dir ../docs/mintlify/

format_docs:
	# replace _docs with docs
	sed -i -e 's/_docs/docs/g' ./docs-scripts/docs-final-formatting.bash
	bash ./docs-scripts/docs-final-formatting.bash
	find docs/mintlify -name "*.mdx" -exec sed -i.bak '/^:::/d' {} + && find docs/mintlify -name "*.bak" -delete

# replace <= with \<=
	find docs/mintlify -name "*.mdx" -exec sed -i.bak 's/<=/\\<=/g' {} + && find docs/mintlify -name "*.bak" -delete

preview_docs:
	cd docs/mintlify && mintlify dev

clean:
	rm -f docs/*.md
	find docs/mintlify -name "*.mdx" -exec rm -f {} +

all_docs: load_docs_scripts api_docs examples_docs format_docs
