"""
Multimodal Remix Engine.
Enables bidirectional transformations:
1. Image -> Caption -> Stylized Prompt -> New AI Image (Image Variation / Remix)
2. Caption -> Image -> Multimodal Caption Analysis
"""
import os
import time
from PIL import Image
from caption_to_image import caption_to_image_engine


class MultimodalRemixEngine:
    """Orchestrates bidirectional image and caption remix workflows."""

    def __init__(self):
        self.image_engine = caption_to_image_engine

    def remix_image(
        self,
        base_caption: str,
        user_modification: str = "",
        target_style: str = "photorealistic",
        aspect_ratio: str = "1:1",
        seed: int = None
    ) -> dict:
        """
        Takes an extracted caption from an image, applies user modifications,
        and generates a brand new AI reimagined image.
        """
        # Combine base caption with user alterations
        if user_modification and user_modification.strip():
            composed_prompt = f"{user_modification.strip()}, inspired by {base_caption.strip()}"
        else:
            composed_prompt = base_caption.strip()

        # Generate the new image
        result = self.image_engine.generate(
            prompt=composed_prompt,
            style=target_style,
            aspect_ratio=aspect_ratio,
            seed=seed
        )

        result["original_caption"] = base_caption
        result["user_modification"] = user_modification
        result["final_composed_prompt"] = composed_prompt
        return result


# Global instance
remix_engine = MultimodalRemixEngine()

