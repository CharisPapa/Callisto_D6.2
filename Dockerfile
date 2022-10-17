# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.0.3-devel-ubuntu20.04
CMD nvidia-smi

#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3.7 python3-pip


WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

#copies the applicaiton from local path to container path
COPY ./ ./

CMD ["python3", "frNER_api.py", "--host=0.0.0.0"]