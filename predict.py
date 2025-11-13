import os
from cog import BasePredictor, Input, Path, Secret
from huggingface_hub import InferenceClient

class Predictor(BasePredictor):
    def setup(self):
        """Setup runs once when the container starts."""
        print("✅ Predictor initialized")

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate an image from"),
        hf_token: Secret = Input(description="Your Hugging Face access token (required for gated models)"),
        width: int = Input(description="Width of the generated image", default=1024, ge=256, le=2048),
        height: int = Input(description="Height of the generated image", default=1024, ge=256, le=2048),
    ) -> Path:
        """Generate an image from the given text prompt."""

        token = hf_token.get_secret_value()
        print("🔐 Using Hugging Face token to authenticate...")

        # Initialize the Hugging Face client with the token
        client = InferenceClient(
            model="black-forest-labs/FLUX.1-dev",
            token=token,
        )

        print(f"🧠 Generating image for prompt: {prompt}")

        # Run inference
        image = client.text_to_image(
            prompt=prompt,
            width=width,
            height=height
        )

        # Save output
        output_path = Path("output.png")
        image.save(output_path)

        print(f"✅ Image saved to {output_path}")
        return output_path
