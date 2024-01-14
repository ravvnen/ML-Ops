
# Docker with CPU
docker run -it -v "${PWD}/ml_art:/ml_art/" -v "${PWD}/data:/data/" -v "${PWD}/outputs:/outputs/" ml_art:v2

# # Docker with GPU
# docker run --gpus all -it -v "${PWD}/ml_art:/ml_art/" -v "${PWD}/data:/data/" -v "${PWD}/outputs:/outputs/"  ml_art_cuda:v1
