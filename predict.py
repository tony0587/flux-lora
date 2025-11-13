# predict.py
import os
from cog import BasePredictor, Input, Path, Secret
import torch
from pathlib import Path as SysPath

# Use FluxPipeline (Flux model)
from diffusers import FluxPipeline

class Predictor(BasePredictor):
    def setup(self):
        # Nothing heavy here — actual loading happens in predict so the secret can be provided per-run.
        print("Predictor ready. Model will be loaded on predict() using provided HF token and LoRA file.")

    def predict(
        self,
        prompt: str = Input(description="Text prompt to generate an image from"),
        hf_token: Secret = Input(description="Hugging Face token (for gated base models)"),
        num_inference_steps: int = Input(
            description="Number of diffusion steps", ge=1, le=150, default=30
        ),
        lora_scale: float = Input(
            description="LoRA mixing scale (how strongly to apply LoRA)", ge=0.0, le=5.0, default=1.0
        ),
        width: int = Input(description="Image width", ge=256, le=2048, default=1024),
        height: int = Input(description="Image height", ge=256, le=2048, default=1024),
        seed: int = Input(description="Random seed for reproducibility", default=42),
    ) -> Path:
        """
        Loads base Flux model from Hugging Face using the provided token, applies the LoRA weights
        from models/penis_worship_flux.safetensors, then generates an image and returns the Path to it.
        """

        # Get token value securely
        token = hf_token.get_secret_value()
        if not token:
            raise ValueError("Missing Hugging Face token. Provide hf_token secret input.")

        # Paths
        lora_path = SysPath("models/penis_worship_flux.safetensors")
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found at {lora_path}. Put your LoRA at models/mymodel.safetensors")

        # Load base model from Hugging Face (gated or public); auth via token
        print("Loading base Flux model from Hugging Face (this may take a while)...")
        # Use bfloat16/float16 depending on availability; bfloat16 on some models, float16 otherwise
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() and hasattr(torch, "bfloat16") else torch.float16

        self.pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            token=token,              # pass token so gated model can be downloaded
            torch_dtype=torch_dtype,
        )

        # If available, use CPU offload to save VRAM (optional)
        try:
            self.pipe.enable_model_cpu_offload()
        except Exception:
            # some pipeline versions may not have this or it may fail; ignore
            pass

        # Load LoRA weights
        print(f"Loading LoRA weights from {lora_path} (scale={lora_scale})...")
        try:
            # Many pipelines provide a method like load_lora_weights
            self.pipe.load_lora_weights(str(lora_path))
        except Exception as e:
            # If your LoRA loader API differs, raise a clearer error
            raise RuntimeError(f"Failed to load LoRA weights: {e}")

        # If pipeline supports applying a multiplier/scale for LoRA, set it.
        # Some implementations provide a `.set_lora_scale()` or accept scale in the call.
        # If not available, we keep lora_scale for later use in call params if supported.
        try:
            # Example API — adapt if your pipeline uses a different API
            if hasattr(self.pipe, "set_lora_scale"):
                self.pipe.set_lora_scale(float(lora_scale))
        except Exception:
            # ignore if not supported
            pass

        # Move pipeline to GPU if available
        if torch.cuda.is_available():
            self.pipe.to("cuda")
            device = "cuda"
        else:
            self.pipe.to("cpu")
            device = "cpu"

        # Prepare RNG/generator
        generator = torch.Generator(device=device).manual_seed(seed)

        # Generate the image
        print("Generating image...")
        # If your pipeline accepts a 'lora_scale' parameter at call time, pass it here.
        call_kwargs = dict(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=7.5,  # you can make guidance_scale an input if you want
            width=width,
            height=height,
            generator=generator,
        )
        # Try to include lora_scale if supported by the pipeline
        try:
            # some pipelines accept lora_scale or lora_weight
            call_kwargs["lora_scale"] = float(lora_scale)
        except Exception:
            pass

        out = self.pipe(**call_kwargs)
        image = out.images[0]

        # Save and return output path
        output_path = Path("output.png")
        image.save(output_path)
        print(f"Saved output image to {output_path}")

        return output_path
