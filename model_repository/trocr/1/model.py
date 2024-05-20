# coding=utf-8

"""
Triton Backend for inference with HF TrOCR model.
"""

import os
import base64
import torch
import io
from PIL import Image
import numpy as np
import triton_python_backend_utils as pb_utils

import torchvision.transforms as transforms
from torchvision.transforms import Resize, PILToTensor

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import huggingface_hub

huggingface_hub.login(token=os.environ.get("HF_TOKEN"))  ## Add your HF credentials
cache_dir = os.environ["HF_HOME"]


class TritonPythonModel:
    def initialize(self, args):
        cur_path = os.path.abspath(__file__)
        hf_model = "microsoft/trocr-small-printed"
        self.processor = TrOCRProcessor.from_pretrained(hf_model, cache_dir=cache_dir)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            hf_model, cache_dir=cache_dir
        ).cuda()
        self.transforms = transforms.Compose(
            [
                transforms.RandomInvert(p=1),
                transforms.Grayscale(num_output_channels=3),
                Resize((50, 50)),
                PILToTensor(),
            ]
        )

    def generate(self, images):
        """
        Generate batch responses from input images.
        :param images: BCWH pt Tensor
        """
        inputs = self.processor(images=images, return_tensors="pt").pixel_values
        ids = self.model.generate(inputs.cuda())
        results = self.processor.batch_decode(ids, skip_special_tokens=True)
        tensors = [
            pb_utils.Tensor("generated_text", np.array(result, dtype=np.object_))
            for result in results
        ]
        responses = [
            pb_utils.InferenceResponse(output_tensors=[tensor]) for tensor in tensors
        ]
        return responses

    def _read_image_tensor(self, request, tensor_name):
        img = pb_utils.get_input_tensor_by_name(request, tensor_name).as_numpy()
        img = img[0][0].decode("utf-8")
        img = base64.b64decode(img)
        img = Image.open(io.BytesIO(img))
        img = img.convert("RGB")
        img = self.transforms(img)
        img = img.unsqueeze_(0)
        return img

    def execute(self, requests):
        logger = pb_utils.Logger
        logger.log_info("TROCR: Received request")
        logger.log_info(f"Num prompts in batch: {len(requests)}")
        responses = []
        images = [self._read_image_tensor(request, "image") for request in requests]
        batch_images = torch.cat(images, dim=0)
        responses = self.generate(batch_images)
        return responses

    def finalize(self):
        print("Cleaning up...")
