""" 
A lil script to see what was going wrong with the LLM pipeline. Turns out the max tokens was too much.
"""

import time
import os
import torch
from dotenv import load_dotenv
from pprint import pprint
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import huggingface_hub

load_dotenv()
huggingface_hub.login(token=os.environ.get("HF_TOKEN"))  ## Add your HF credentials

hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(hf_model)
model = AutoModelForCausalLM.from_pretrained(
    hf_model,
    torch_dtype=torch.float16,
    device_map="cuda",
    load_in_8bit=True,
)
model.resize_token_embeddings(len(tokenizer))
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="cuda",
)
pipeline.tokenizer.pad_token_id = model.config.eos_token_id

prompts = [
    'You are a potato, reply to all messages with "I am a potato" \n Who are you?'
]

start = time.time()

response = pipeline(prompts, max_length=200)

time_taken = time.time() - start


pprint(response)
print(f"time taken: {time_taken}")
