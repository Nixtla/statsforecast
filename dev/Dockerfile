FROM python:3.11

COPY dev/requirements.txt setup/

RUN pip install --upgrade pip
RUN pip install -r setup/requirements.txt

# Setting the working environment.
# This will server as an empty directory where all the files be dropped to.
WORKDIR /workdir/
