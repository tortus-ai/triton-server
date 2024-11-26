# coding=utf-8

"""

"""

import os
import re
from typing import List
import json
import triton_python_backend_utils as pb_utils
import numpy as np
import torch


os.environ["HF_HOME"] = "/opt/tritonserver/.hf-cache"
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
import huggingface_hub

huggingface_hub.login(token=os.environ.get("HF_TOKEN"))  ## Add your HF credentials


class TritonPythonModel:
    def initialize(self, args):
        cur_path = os.path.abspath(__file__)
        self.model_config = json.loads(args["model_config"])
        self.model_params = self.model_config.get("parameters", {})
        self.max_length = int(
            self.model_params.get("max_length", {}).get("string_value", "1024")
        )
        quant_map = {
            "4bit": {"load_in_4bit": True},
            "8bit": {"load_in_8bit": True},
            "full": {},
        }
        self.sys_prompt = self.model_params.get("sys_prompt", {}).get("string_value", "")
        self.user_prompt = self.model_params.get("user_prompt", {}).get("string_value", "")
        if not self.sys_prompt or not self.user_prompt:
            raise ValueError("sys_prompt and user_prompt must be provided")
        logger = pb_utils.Logger
        quant_level = self.model_params.get("quantize", {}).get("string_value", "")
        logger.log_info(f"Quant level: {quant_level}")
        quant_arg = quant_map.get(quant_level, {})
        if quant_arg and quant_level == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            quant_arg = {"quantization_config": bnb_config}
        hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir=os.environ["HF_HOME"],
            **quant_arg,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.pipeline.tokenizer.pad_token_id = self.model.config.eos_token_id

    def generate(self, prompts: List[List[dict]]):
        logger = pb_utils.Logger
        batches = self.pipeline(
            prompts,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            max_length=self.max_length,
            batch_size=len(prompts),
        )
        output_tensors = []

        for i, batch in enumerate(batches):
            texts = []
            for i, seq in enumerate(batch):
                text = seq["generated_text"][-1]["content"]
                tokens = self.tokenizer.encode(text)
                logger.log_info(
                    f"Processed item. Number of output tokens: {len(tokens)}"
                )
                texts.append(text)
            results = self.handle_results(texts)
            tensor = pb_utils.Tensor("precision", np.array(results, dtype=np.object_))
            output_tensors.append(tensor)

        return output_tensors

    def _read_tensor(self, request, tensor_name):
        msgs = pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()
        msgs = msgs[0][0].decode("utf-8")
        return msgs

    def _split_entailment(self, request):
        """
        Logic to split data for entailment verification sub-items
        (precision). We generate sub-setences from the document generated,
        and then we generate entailment prompts for each sub-sentence.
        We also generate hashes that uniquely identify the transcript + document,
        for post-processing.
        :param request: The request object
        :return: A list of tuples containing the system, and user prompts.
        """
        transcript = self._read_tensor(request, "transcript")
        document = self._read_tensor(request, "document")

        # split the document using regex by sentence terminators
        document = re.split(r"[.!?]", document)
        document = [sent.strip() for sent in document if sent.strip()]

        user_prompts = [self.user_prompt.replace("$DOCUMENT", transcript).replace("$SENTENCE", sent) for sent in document]
        sys_prompts = [self.sys_prompt for _ in document]
        paired_prompts = list(zip(sys_prompts, user_prompts))
        return paired_prompts

    def _make_prompt(self, request):
        paired_prompts = self._split_entailment(request)
        prompt_dicts = [[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ] for sys, user in paired_prompts]
        return prompt_dicts

    def _extract_answer(self, response:str):
        """
        Takes a response, and extracts the answer from the response.
        :param response: str, The response from the model
        :return: int, The answer extracted from the response
        """
        pattern = r"Answer:\s*([01])"
        match = re.search(pattern, response)
        if match:
            return int(match.group(1))
        return 0

    def handle_results(self, tensor_results: list):
        """
        Post-processing logic for entailment verification sub-items.
        :param tensor_results: The tensor results from the entailment verification
        :return: A dict containing the results for each hash
        """
        sum = 0
        for tensor in tensor_results:
            result = tensor.as_numpy()[0][0].decode("utf-8")
            answer = self._extract_answer(result)
            sum += answer
        return sum / len(tensor_results)

    def execute(self, requests: List):
        logger = pb_utils.Logger
        logger.log_info("Llama Received request")
        logger.log_info(f"(Llama) Num prompts in batch: {len(requests)}")
        prompts = [self._make_prompt(request) for request in requests]
        logger.log_info(f"Number Prompts: {len(prompts)}")
        tensor_results = self.generate(prompts)
        responses = [
            pb_utils.InferenceResponse(output_tensors=[tensor])
            for tensor in tensor_results
        ]

        return responses

    def finalize(self):
        print("Cleaning up...")
