#!/bin/bash

locust -f ../locustfile.py \
--data /Users/nin/repos/triton-sever/load_testing/example/data_llama_4000.json \
--schema /Users/nin/repos/triton-sever/model_repository/llama3_8b/config.pbtxt \
--host http://34.34.52.244/v2/models/llama3_8b/infer \
--loglevel DEBUG --run-time 60m