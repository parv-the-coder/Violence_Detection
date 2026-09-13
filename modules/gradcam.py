"""Explainability helpers for DINOv2-based violence detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class HeatmapResult:
    heatmap: np.ndarray
    overlay: np.ndarray


class DinoV2AttentionRollout:
    """Generate patch-level attention rollout maps from a DINOv2 backbone.

    The implementation is intentionally defensive: if a backbone block does not
    expose an attention module with qkv projections, the block is skipped.
    """

    def __init__(self, backbone: torch.nn.Module):
        self.backbone = backbone
        self._captured_inputs: List[torch.Tensor] = []

    def _get_native_backbone(self) -> torch.nn.Module:
        """Return the actual vision transformer, unwrapping helper modules if needed."""

        if hasattr(self.backbone, "blocks"):
            return self.backbone
        if hasattr(self.backbone, "backbone") and hasattr(self.backbone.backbone, "blocks"):
            return self.backbone.backbone
        return self.backbone

    def _register_hooks(self):
        hooks = []

        def capture_input(_, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                self._captured_inputs.append(inputs[0].detach())

        blocks = getattr(self._get_native_backbone(), "blocks", [])
        for block in blocks:
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "qkv") and hasattr(attn, "num_heads"):
                hooks.append(attn.register_forward_pre_hook(capture_input))

        return hooks

    @staticmethod
    def fallback_heatmap(image: Image.Image) -> np.ndarray:
        """Generate a visible heatmap when attention rollout cannot be computed."""

        rgb = np.array(image.resize((224, 224))).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=7, sigmaY=7)
        edges = cv2.Canny(gray, 40, 120)
        combined = cv2.addWeighted(blurred, 0.75, edges, 0.25, 0)
        combined = combined - combined.min()
        if combined.max() > 0:
            combined = combined / combined.max()
        heatmap_uint8 = np.uint8(255 * combined)
        return cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    @staticmethod
    def _normalize_heatmap(values: np.ndarray) -> np.ndarray:
        """Scale raw attention weights into the 0-255 display range."""

        values = values.astype(np.float32)
        low = float(values.min())
        high = float(values.max())
        if high <= low:
            return np.zeros_like(values, dtype=np.uint8)
        values = (values - low) / (high - low)
        return np.uint8(np.clip(values * 255.0, 0, 255))

    @staticmethod
    def _attention_from_input(attn_module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channel_count = x.shape
        qkv = attn_module.qkv(x).reshape(batch_size, token_count, 3, attn_module.num_heads, channel_count // attn_module.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        attention_scores = (q @ k.transpose(-2, -1)) * getattr(attn_module, "scale", (channel_count // attn_module.num_heads) ** -0.5)
        attention_scores = attention_scores.softmax(dim=-1)
        return attention_scores[0]

    def generate(self, image: Image.Image) -> tuple[np.ndarray, str]:
        """Return a normalized heatmap for a single RGB image and the mode used."""

        self._captured_inputs = []
        native_backbone = self._get_native_backbone()
        device = next(native_backbone.parameters()).device
        preprocess = getattr(self.backbone, "preprocess", None) or getattr(native_backbone, "preprocess", None)
        if preprocess is None:
            return self.fallback_heatmap(image), "fallback-no-preprocess"

        tensor = preprocess(image).unsqueeze(0).to(device)
        hooks = self._register_hooks()

        try:
            with torch.no_grad():
                _ = self.backbone(tensor)
        finally:
            for hook in hooks:
                hook.remove()

        if not self._captured_inputs:
            return self.fallback_heatmap(image), "fallback-no-hooks"

        saliency_maps: List[np.ndarray] = []
        for block_input, block in zip(self._captured_inputs, getattr(native_backbone, "blocks", [])):
            attn = getattr(block, "attn", None)
            if attn is None or not hasattr(attn, "qkv"):
                continue

            try:
                attention = self._attention_from_input(attn, block_input)
                cls_attention = attention[:, 0, 1:]
                cls_attention = cls_attention.mean(dim=0)
                saliency_maps.append(cls_attention.detach().cpu().numpy())
            except Exception:
                continue

        if not saliency_maps:
            return self.fallback_heatmap(image), "fallback-rollout-failed"

        tokens = np.mean(np.stack(saliency_maps, axis=0), axis=0)

        token_count = tokens.shape[0]
        if token_count == 0:
            return self.fallback_heatmap(image), "fallback-empty-rollout"

        side = int(math.ceil(math.sqrt(token_count)))
        if side * side != token_count:
            tokens = np.pad(tokens, (0, side * side - token_count), mode="edge")

        heatmap = tokens.reshape(side, side)
        heatmap = cv2.resize(heatmap, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = self._normalize_heatmap(heatmap)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        return colored, "cls-attention"

    @staticmethod
    def overlay(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        # The fallback heatmap is always 224x224, and a rollout map matches
        # whichever frame produced it, so resize before blending.
        if heatmap_bgr.shape[:2] != image_bgr.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr,
                (image_bgr.shape[1], image_bgr.shape[0]),
                interpolation=cv2.INTER_CUBIC,
            )
        return cv2.addWeighted(image_bgr, 1.0 - alpha, heatmap_bgr, alpha, 0)
