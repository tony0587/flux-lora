import os
from cog import BasePredictor, Input, Path
from huggingface_hub import InferenceClient

class Predictor(BasePredictor):
    def setup(self):
        """Setup runs once when the container starts."""
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not token:
            raise ValueError("Missing Hugging Face token! Add HUGGING_FACE_HUB_TOKEN in your secrets.")

        # Initialize Hugging Face Inference client
        self.client = InferenceClient(
            model="black-forest-labs/FLUX.1-dev",
            token=token,
        )
        print("✅ Connected to Hugging Face FLUX.1-dev model")

    def predict(
        self,
        prompt: str = Input(description="Prompt to generate an image from"),
    ) -> Path:
        """Generate an image from the given text prompt."""
        print(f"🧠 Generating image for: {prompt}")

        # Run inference
        image = self.client.text_to_image(prompt)

        # Save the result to output path
        output_path = Path("output.png")
        image.save(output_path)

        print(f"✅ Image saved to {output_path}")
        return output_path
