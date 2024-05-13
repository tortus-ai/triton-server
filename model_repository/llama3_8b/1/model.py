import os
import json
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import huggingface_hub
from threading import Thread

os.environ["TRANSFORMERS_CACHE"] = "/opt/tritonserver/model_repository/llama3_8b/hf-cache"

huggingface_hub.login(token=os.environ.get("HF_TOKEN"))  ## Add your HF credentials


class TritonPythonModel:
    def initialize(self, args):
        cur_path = os.path.abspath(__file__)
        hf_model = 'meta-llama/Meta-Llama-3-8B-Instruct'
        self.max_output_length = 100 # TODO change this
        self.tokenizer= AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def generate(self, prompt):
        sequences = self.pipeline(
                prompt,
                do_sample=True,
                top_k = 10,
                num_return_sequences=1,
                eos_token_id = self.tokenizer.eos_token_id,
                max_length = self.max_output_length,
                )
        output_tensors = []
        texts = []

        for i, seq in enumerate(sequences):
            text = seq['generated_text']
            texts.append(text)

        tensor = pb_utils.Tensor('generated_text', np.array(texts,
            dtype=np.object_))

        output_tensors.append(tensor)
        response = pb_utils.InferenceResponse(output_tensors=output_tensors)
        return response
    def _read_tensor(self, request, tensor_name):
            msgs = pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()
            msgs = msgs[0].decode("utf-8")
            return msgs

    def execute(self, requests):
        responses = []
        for request in requests:
            # Decode the Byte Tensor into Text
            sys_msg = self._read_tensor(request, "system_message")
            user_msg = self._read_tensor(request, "user_message")
            prompt = f"{sys_msg} \n{user_msg}"
            response = self.generate(prompt)
            responses.append(response)

        return responses

    def finalize(self):
        print('Cleaning up...')
