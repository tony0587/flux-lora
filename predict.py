# predict.py
from cog import BasePredictor, Input, Path
import torch
from diffusers import FluxPipeline
from huggingface_hub import login
from pathlib import Path as SysPath
import os


class Predictor(BasePredictor):
    def setup(self):
        """Authenticate and load base Flux model with LoRA weights."""
        print("Authenticating with Hugging Face...")
        token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if token:
            login(token=token)
        else:
            print("⚠️ No Hugging Face token found — gated models will fail to load")

        print("Loading base model...")
        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()

        # Load LoRA weights
        lora_path = SysPath("models/penis_worship_flux.safetensors")
        print(f"Loading LoRA weights from: {lora_path}")
        self.pipe.load_lora_weights(lora_path)

        print("✅ Model ready!")

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate image from"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        steps: int = Input(description="Number of diffusion steps", default=30),
        guidance_scale: float = Input(description="Prompt adherence", default=3.5),
        width: int = Input(description="Image width", default=1024),
        height: int = Input(description="Image height", default=1024),
        seed: int = Input(description="Random seed", default=42),
    ) -> Path:
        """Generate an image using the LoRA-boosted Flux model."""
        generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"Generating image for: {prompt}")
        image = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        output_path = Path("output.png")
        image.save(output_path)
        print(f"Image saved to {output_path}")

        return output_path
