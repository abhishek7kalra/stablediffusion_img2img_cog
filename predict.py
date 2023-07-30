from PIL import Image
from cog import BasePredictor, Input, Path
import torch
from typing import List
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import Dataset
import os
import random
import io
import base64


MODEL_ID = "runwayml/stable-diffusion-v1-5"
MODEL_CACHE = "diffusers-cache"

class Predictor(BasePredictor):
    def setup(self):
        print("Loading pipeline...")
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")
        
        self.pipe.enable_attention_slicing()

    @torch.inference_mode()
    def predict(
        self,
        input_photo: Path = Input(description="Path to the input photo"),
        prompt: str = Input(description="Style prompt", default="Van Gogh style"),
    ) -> Path:
        
        input_photo_path = input_photo
        
        print("testing")
        init_img = Image.open(input_photo_path)
        init_img = init_img.resize((512, 512))

        output = self.pipe(prompt=prompt, image=init_img, strength=0.75, guidance_scale=7.5).images
        
        print("saving...")
    
        with io.BytesIO() as output_buffer:
            output[0].save(output_buffer, format='PNG')
            output_bytes = output_buffer.getvalue()

        # Create a data URL from the output bytes
        data_url = "data:image/png;base64," + base64.b64encode(output_bytes).decode()
        
        # Saving the image
        print("saved")
        
        # Return the data URL as the output prediction
        return data_url

