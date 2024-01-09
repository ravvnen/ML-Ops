# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile
# Mounting is More efficient than copying
# COPY ml_art/ ml_art/
# COPY data/ data/
# I will mount all the created dir to my host folders
RUN mkdir ml_art
RUN mkdir data
# defualt log directory set by HYDRA will be mounted to notebooks on run
RUN mkdir outputs

WORKDIR /
RUN make requirements

ENTRYPOINT ["make","data"]

# Set the default command to run "make train"
CMD ["train"]
