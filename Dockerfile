FROM nvcr.io/nvidia/tritonserver:23.10-py3
RUN pip install transformers==4.34.0 protobuf==3.20.3 sentencepiece==0.1.99 accelerate==0.23.0 einops==0.6.1 bitsandbytes
