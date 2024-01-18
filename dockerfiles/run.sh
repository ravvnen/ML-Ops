
# # Docker with CPU
# docker run -it -v "${PWD}/ml_art:/ml_art/" -v "${PWD}/data:/data/" -v "${PWD}/outputs:/outputs/" ml_art:v2

# # Docker with GPU
# docker run --gpus all -it -v "${PWD}/ml_art:/ml_art/" -v "${PWD}/data:/data/" -v "${PWD}/outputs:/outputs/"  --entrypoint bash wikiart-gcloud-train:v1

# No Mounts - Pack Everything in Container
docker run --gpus all -it  wikiart-gcloud-train:v1
