from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from typing import List
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import Dataset
import os
import random


######checking
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")


MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            #float16
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        )
        # .to("mps")
        #remove mps and add cuda
        # .to("cuda")
        self.pipe.enable_attention_slicing()
        #remove slicing

    @torch.inference_mode()
    def predict(
        self,
        input_photo: Path = Input(description="Path to the input photo"),
        #output_path: Path = Input(description="Path to save the output styled image"),
        prompt: str = Input(description="Style prompt", default="Van Gogh style"),
    ) -> Path:
        input_photo_path = input_photo
        #output_image_path = output_path
        print("testing")
        init_img = Image.open(input_photo_path)
        init_img = init_img.resize((768, 512))

        output = self.pipe(prompt=prompt, image=init_img)
        print("saving")
        output_image_path = f"/tmp/out.png"  # Temporarily save the image
        output[0].save(output_image_path)

        return output_image_path
