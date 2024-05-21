# Triton server

<div align="center">
<img width="25%" src="assets/triton_logo.png">
</div>

This repo contains the code required to set up a triton server to host ML models. [Triton servers](https://github.com/triton-inference-server) were set up by Nvidia to optimise load for using ML models in production. They support features like dynamic batching and GPU optimisaitons for Nvidia chips. With a Triton server, we're also able to host multiple models at the same time and routing is handled via the folder structure of the [Model Repository](#Model-Repository).

## Getting started

To get up and running, clone the repo and do the following steps:

### Build docker image

Run the following command:

```
docker build -t triton .
```

### Set up environment variables

We currenlty only need the Huggingface token to get this working so make sure to set this in a `.env` file under `HF_TOKEN`. See the [example][.env.example].

### Run server

Once you've got this container ready, you can start the server with the following command:
```
docker run --gpus all -it --rm --shm-size=1G --ulimit memlock=-1 --ulimit stack=67108864 --env-file ./.env -v ${PWD}/model_repository:/opt/tritonserver/model_repository -p 80:8000 triton tritonserver --model-repository=model_repository
```

If this command scares you as much as it does me, you can run a shell script we made to hide what's happening:

```
./run-server.sh
```

PS Don't forget to `chmod +x ./run_server.sh` if this is the first time you're using it!


### Send Requests

Once the server is set up, you can hit it on localhost or through your machine's IP with the following convention:

```
curl -X POST http://{IP}/v2/models/{MODEL_NAME}/infer -d {"inputs": [{"name":"system_message","datatype":"BYTES","shape":[1,1],"data":[["You are a useful assistant, please respond in Pirate speak"]]}, {"name":"user_message","datatype":"BYTES","shape":[1,1],"data":[["Why is the sky blue?"]]}]}
```

NOTE: The payload you send in the request is dependent on the configuration of the model you're using. The model config explains what the input names it expects are and you'll have to provide this using the following convention:

```json
{
    "inputs": [
        {
            "name": "tensor_1_name",
            "datatype": number,
            "shape": array,
            "data": "insert data"
        },
        ...,
    ]
}
```

## Model Repository 

The [Model Repository](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_repository.md) is the directory where all of the models you'll be hosting will sit. The layout should be the following:
```
  <model-repository-path>/
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    <model-name>/
      [config.pbtxt]
      [<output-labels-file> ...]
      <version>/
        <model-definition-file>
      <version>/
        <model-definition-file>
      ...
    ...
```

The main two files you'll care about for a particular model is the model file and config file

### Model file

This is a python file which uses triton's SDK to implement a class which represents the Model and how it handles requests. See [here](https://github.com/triton-inference-server/tutorials/blob/main/HuggingFace/client.py) for an example using a HuggingFace model.

### Model Config file

This is pbtxt file containing the config for your model. You can set the batch size, interfaces and other variables. See [here](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md) for more info.


## Run single load test

From upper level directory, CLI:

```bash
locust -f load_testing/locustfile.py --csv <filename> --headless -t5m --csv-full-history -u 1 -r 1 --data <path_to_data_json> --schema <path_to_model_pbtxt> --hsot <hostname>
```

Flags:
- `--csv`: `.csv` to save results to
- `--headless`: whether or not to run a UI for visualisation
- `-t`: duration of load test
- `--csv-full-history`: whether or not to save all history of the laod test.
- `-u`: number of users
- `-r`: number of requests/second per user
- `--data`: `.json` file that contains the input data to use for the load test.
- `--schema`: path to the Triton model config `.pbtxt` which matches the host endpoint

An example of a data JSON for the Llama3-8b deployment can be found at `load_testing/example/data_llama.json`