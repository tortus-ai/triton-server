#!/bin/bash

locust -f ${PWD}/load_testing/locustfile.py \
    --data ${PWD}/load_testing/data/soap_note.json \
    --schema ${PWD}/model_repository/llama3_8b/config.pbtxt \
    --host http://34.34.52.244/v2/models/llama3_8b/infer \
    --loglevel DEBUG --run-time 60m \
    --starting-users 2 --bulk-ramp 2 --bulk-interval 60
