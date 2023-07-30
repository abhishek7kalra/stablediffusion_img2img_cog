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
    ) -> List[Path]:
        
        input_photo_path = input_photo
        
        print("testing")
        init_img = Image.open(input_photo_path)
        init_img = init_img.resize((512, 512))

        output = self.pipe(prompt=prompt, image=init_img, strength=0.75, guidance_scale=7.5)
        output_paths = []
        
        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))
        
        print("saved")
        
        return output_paths