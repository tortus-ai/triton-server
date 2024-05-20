#!/bin/bash

docker run --gpus all -it --rm --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 --env-file ./.env -v ${PWD}/model_repository:/opt/tritonserver/model_repository -v ${PWD}/.hf-cache:/opt/tritonserver/.hf-cache -p 80:8000 triton tritonserver --model-repository=model_repository
