"""
Temporal Sequence Classifier using Transformer Encoder (PyTorch).

Takes a sequence of DINOv2 spatial embeddings (SEQUENCE_LENGTH frames × 768-dim)
and classifies the segment as Violence or Non-Violence.
"""

import torch
import torch.nn as nn

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FEATURE_DIM, SEQUENCE_LENGTH, NUM_HEADS, NUM_TRANSFORMER_LAYERS, DROPOUT, NUM_CLASSES


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer Encoder block with Post-LayerNorm architecture:
      MultiHeadAttention → Residual → LayerNorm
      FFN(ReLU6) → Residual → LayerNorm
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU6(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            Output tensor of same shape.
        """
        # Self-attention with residual, then norm
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm_1(x + attn_output)

        # FFN with residual, then norm
        ffn_output = self.ffn(x)
        x = self.layernorm_2(x + ffn_output)
        return x


class TemporalTransformer(nn.Module):
    """
    Temporal sequence classifier using Transformer Encoder.

    Architecture:
        1. Prepend a learnable CLS token to the sequence
        2. Add learnable positional embeddings
        3. Pass through N Transformer Encoder blocks
        4. Extract CLS token output
        5. Classify via Dense → Sigmoid

    Input:  (batch, seq_len, 768) — seq_len frames × 768-dim DINOv2 features
    Output: (batch, 2) — [P(violence), P(non-violence)]
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        seq_len: int = SEQUENCE_LENGTH,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_TRANSFORMER_LAYERS,
        dropout: float = DROPOUT,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(seq_len + 1, feature_dim)  # +1 for CLS

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(feature_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, feature_dim) — sequence of frame embeddings

        Returns:
            (batch, num_classes) — classification probabilities
        """
        batch_size = x.size(0)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, feature_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, feature_dim)

        # Add positional embeddings
        positions = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_embedding(positions)

        # Transformer encoder blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Extract CLS token output
        cls_output = x[:, 0, :]  # (batch, feature_dim)

        # Classification
        logits = self.classifier(cls_output)  # (batch, num_classes)
        output = self.sigmoid(logits)
        return output


if __name__ == "__main__":
    # Quick verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TemporalTransformer()
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    dummy_input = torch.randn(1, SEQUENCE_LENGTH, FEATURE_DIM).to(device)
    output = model(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {output.shape}"
    print("[OK] Temporal Transformer verified successfully!")
