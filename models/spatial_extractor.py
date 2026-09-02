"""
Spatial Feature Extractor using DINOv2 (vitb14 - Base).

Wraps the DINOv2 backbone as a frozen feature extractor that produces
a 768-dimensional CLS token embedding per input frame.
"""

import torch
import torch.nn as nn
from torchvision import transforms

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DINOV2_MODEL_NAME, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, FEATURE_DIM


class DINOv2SpatialExtractor(nn.Module):
    """
    DINOv2-based spatial feature extractor.

    Loads the DINOv2 vitb14 backbone via torch.hub, freezes all weights,
    and extracts the CLS token embedding (768-dim) for each input frame.
    """

    def __init__(self, model_name: str = DINOV2_MODEL_NAME, freeze: bool = True):
        super().__init__()
        self.model_name = model_name
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        if freeze:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Preprocessing transform (ImageNet normalization expected by DINOv2)
        self.preprocess = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 3, 224, 224), already normalized.

        Returns:
            CLS token embeddings of shape (batch, 768).
        """
        with torch.no_grad():
            features = self.backbone(x)  # (batch, 768)
        return features

    def extract_from_pil(self, pil_image) -> torch.Tensor:
        """
        Convenience method: takes a PIL image, preprocesses it,
        and returns the feature vector.

        Args:
            pil_image: A PIL.Image in RGB format.

        Returns:
            Feature vector of shape (768,).
        """
        tensor = self.preprocess(pil_image).unsqueeze(0)  # (1, 3, 224, 224)
        device = next(self.backbone.parameters()).device
        tensor = tensor.to(device)
        return self.forward(tensor).squeeze(0)  # (768,)

    def get_preprocess_transform(self) -> transforms.Compose:
        """Returns the preprocessing transform for external use."""
        return self.preprocess


if __name__ == "__main__":
    # Quick verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DINOv2SpatialExtractor()
    model = model.to(device)

    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, FEATURE_DIM), f"Expected (1, {FEATURE_DIM}), got {output.shape}"
    print("[OK] DINOv2 spatial extractor verified successfully!")
