# Base image
FROM  nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirementsCUDA.txt requirements.txt
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

RUN make requirements


ENV LOCAL_PATH=/

ENTRYPOINT ["make","data"]

# Set the default command to run "make train"
CMD ["train"]
