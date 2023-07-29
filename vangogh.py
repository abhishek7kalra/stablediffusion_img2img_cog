# -*- coding: utf-8 -*-

# !nvidia-smi

# !pip install diffusers transformers ftfy
# !pip install -qq "ipywidgets>=7,<8"
# !pip install accelerate

import argparse
import os
parser = argparse.ArgumentParser(description='Image Style Transfer')
parser.add_argument('--input_photo_path', type=str, required=True, help='Path to the input photo')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output styled image')
args = parser.parse_args()

input_photo_path = args.input_photo_path

output_path = os.path.join(args.output_path, "output.png")

from huggingface_hub import notebook_login

notebook_login()

import inspect
import warnings
from typing import List, Optional, Union

import torch
from torch import autocast
from tqdm.auto import tqdm

from diffusers import StableDiffusionImg2ImgPipeline

device = "cuda"
model_id_or_path = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float16)
pipe = pipe.to(device)

import requests
from io import BytesIO
from PIL import Image

#url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

input_photo_path = 'sketch-mountains-input.jpg'

#response = requests.get(url)
init_img = Image.open(input_photo_path)
#init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
init_img

prompt = "Van Gogh style"

images = pipe(prompt=prompt, image=init_img, strength=0.75, guidance_scale=7.5).images

images[0].save(output_path)

