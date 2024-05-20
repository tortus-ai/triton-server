#!/bin/bash

locust -f ${PWD}/load_testing/locustfile.py \
--data ${PWD}/load_testing/example/data_llama_4000.json \
--schema ${PWD}/model_repository/llama3_8b/config.pbtxt \
--host http://34.34.52.244/v2/models/llama3_8b/infer \
--loglevel DEBUG --run-time 60m \
