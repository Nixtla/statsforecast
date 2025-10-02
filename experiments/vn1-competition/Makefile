IMAGE := vn1-competition
ROOT := $(shell dirname $(realpath $(firstword ${MAKEFILE_LIST})))
PARENT_ROOT := $(shell dirname ${ROOT})
PORT := 8888

DOCKER_PARAMETERS := \
	--user $(shell id -u) \
	-v ${ROOT}:/app \
	-w /app \
	-e HOME=/tmp

init: download_data
	docker build -t ${IMAGE} .

jupyter:
	docker run -d --rm ${DOCKER_PARAMETERS} -e HOME=/tmp -p ${PORT}:8888 ${IMAGE} \
		bash -c "jupyter lab --ip=0.0.0.0 --no-browser --NotebookApp.token=''"

bash_docker: 
	docker run -it --rm ${DOCKER_PARAMETERS} ${IMAGE}

download_data:
	@mkdir -p data
	@test -f data/phase_0_sales.csv || curl https://www.datasource.ai/attachments/eyJpZCI6Ijk4NDYxNjE2NmZmZjM0MGRmNmE4MTczOGMyMzI2ZWI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFNhbGVzLmNzdiIsInNpemUiOjEwODA0NjU0LCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ -o data/phase_0_sales.csv
	@test -f data/phase_1_sales.csv || curl https://www.datasource.ai/attachments/eyJpZCI6ImM2OGQxNGNmNTJkZDQ1MTUyZTg0M2FkMDAyMjVlN2NlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTgzOTYsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/phase_1_sales.csv
	@test -f data/phase_2_sales.csv || curl https://www.datasource.ai/attachments/eyJpZCI6IjhlNmJmNmU3ZTlhNWQ4NTcyNGVhNTI4YjAwNTk3OWE1LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMiAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTI0MzcsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/phase_2_sales.csv
	@test -f data/solution_1st_place.csv || curl https://www.datasource.ai/attachments/eyJpZCI6IjI1NDQxYmMyMTQ3MTA0MjJhMDcyYjllODcwZjEyNmY4LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoicGhhc2UgMiBzdWJtaXNzaW9uIGV4YW1pbmUgc21vb3RoZWQgMjAyNDEwMTcgRklOQUwuY3N2Iiwic2l6ZSI6MTk5MzAzNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0 -o data/solution_1st_place.csv
	@test -f data/solution_2nd_place.csv || curl https://www.datasource.ai/attachments/eyJpZCI6IjU3ODhjZTUwYTU3MTg3NjFlYzMzOWU0ZTg3MWUzNjQxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoidm4xX3N1Ym1pc3Npb25fanVzdGluX2Z1cmxvdHRlLmNzdiIsInNpemUiOjM5MDkzNzksIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/solution_2nd_place.csv
	@test -f data/solution_3rd_place.csv || curl https://www.datasource.ai/attachments/eyJpZCI6ImE5NzcwNTZhMzhhMTc2ZWJjODFkMDMwMTM2Y2U2MTdlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiYXJzYW5pa3phZF9zdWIuY3N2Iiwic2l6ZSI6Mzg4OTcyNCwibWltZV90eXBlIjoidGV4dC9jc3YifX0 -o data/solution_3rd_place.csv
	@test -f data/solution_4th_place.csv || curl https://www.datasource.ai/attachments/eyJpZCI6ImVlZmUxYWY2NDFjOWMwM2IxMzRhZTc2MzI1Nzg3NzIxLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiVEZUX3R1bmVkX1YyX3NlZWRfNDIuY3N2Iiwic2l6ZSI6NjA3NDgzLCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ -o data/solution_4th_place.csv
	@test -f data/solution_5th_place.csv || curl https://www.datasource.ai/attachments/eyJpZCI6IjMwMDEwMmY3NTNhMzlhN2YxNTk3ODYxZTI1N2Q2NzRmLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiZGl2aW5lb3B0aW1pemVkd2VpZ2h0c2Vuc2VtYmxlLmNzdiIsInNpemUiOjE3OTU0NzgsIm1pbWVfdHlwZSI6InRleHQvY3N2In19 -o data/solution_5th_place.csv

run:
	docker run -v "$(ROOT)/data:/app/data" -i --rm ${DOCKER_PARAMETERS} ${IMAGE} \
		uv run quarto render src/vn1_competition.ipynb --to gfm --output-dir .
