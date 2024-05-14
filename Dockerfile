FROM nvcr.io/nvidia/tritonserver:23.10-py3
ENV TZ=Europe \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libjpeg-dev \
    zlib1g-dev \
    libxml2-dev \
    libxslt-dev \
    mupdf \
    mupdf-tools \
    libmupdf-dev \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install transformers==4.34.0 protobuf==3.20.3 sentencepiece==0.1.99 accelerate==0.23.0 einops==0.6.1 bitsandbytes
