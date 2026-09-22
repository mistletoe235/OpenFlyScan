"""Inject precomputed per-image DINO tokens into a frozen Pi3X forward."""

from __future__ import annotations

import torch


class CachedDinoEncoder(torch.nn.Module):
    def __init__(self, tokens: torch.Tensor, output_dtype: torch.dtype | None = None):
        super().__init__()
        if tokens.ndim != 3 or tokens.shape[-1] != 1024:
            raise ValueError(f"expected [B*N, patches, 1024], got {tuple(tokens.shape)}")
        self.register_buffer("tokens", tokens.detach(), persistent=False)
        self.output_dtype = output_dtype

    def forward(self, images: torch.Tensor, *, is_training: bool = True) -> dict[str, torch.Tensor]:
        del is_training
        if images.shape[0] != self.tokens.shape[0]:
            raise RuntimeError(f"cached DINO batch mismatch: {images.shape[0]} vs {self.tokens.shape[0]}")
        dtype = self.output_dtype if self.output_dtype is not None else self.tokens.dtype
        return {"x_norm_patchtokens": self.tokens.to(device=images.device, dtype=dtype)}
