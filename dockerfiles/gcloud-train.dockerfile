# Base image
FROM python:3.11-slim

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile
# Mounting is More efficient than copying
COPY ml_art/ ml_art/
COPY outputs/2024-01-17/18-49-13/train_set.pt data/processed/train_set.pt
COPY outputs/2024-01-17/18-49-13/test_set.pt data/processed/test_set.pt
# I will mount all the created dir to my host folders
# RUN mkdir ml_art
# RUN mkdir data
# defualt log directory set by HYDRA will be mounted to notebooks on run
RUN mkdir outputs

RUN make cuda_requirements


ENV LOCAL_PATH=/

ENTRYPOINT ["make","train"]
