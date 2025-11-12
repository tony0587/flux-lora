# predict.py
# Cog prediction interface for a Flux + LoRA image generation model
# https://github.com/replicate/cog

from cog import BasePredictor, Input, Path
import torch
from diffusers import DiffusionPipeline
from pathlib import Path as SysPath


class Predictor(BasePredictor):
    def setup(self):
        """Load base Flux model and apply LoRA weights once when container starts."""
        print("Loading base model...")
        self.pipe = DiffusionPipeline.from_pretrained(
            "black-forest-labs/flux-dev",
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # Load LoRA weights from your model path
        lora_path = SysPath("models/mymodel.safetensors")
        print(f"Loading LoRA weights from: {lora_path}")
        self.pipe.load_lora_weights(lora_path)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")

        print("Model and LoRA loaded successfully!")

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate image from"),
        negative_prompt: str = Input(
            description="Negative prompt to avoid unwanted results", default=""
        ),
        steps: int = Input(
            description="Number of diffusion steps", ge=1, le=50, default=30
        ),
        guidance_scale: float = Input(
            description="How closely to follow the prompt", ge=1, le=15, default=7.5
        ),
        width: int = Input(description="Image width", ge=256, le=1024, default=512),
        height: int = Input(description="Image height", ge=256, le=1024, default=512),
        seed: int = Input(description="Random seed (for reproducibility)", default=42),
    ) -> Path:
        """Run image generation and return the generated image."""
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        print(f"Generating image for prompt: {prompt}")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            generator=generator,
        ).images[0]

        output_path = Path("output.png")
        image.save(output_path)
        print(f"Image saved to {output_path}")

        return output_path
