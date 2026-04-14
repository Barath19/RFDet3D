"""Simple depth encoder for converting raw depth maps to latent tokens.

When LingBot-Depth is not available, this module converts GT depth
(e.g., ARKit LiDAR) into depth latent tokens that Det3DHead can
cross-attend to. Much simpler than the full LingBot pipeline but
provides real depth information to the 3D head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DepthMapEncoder(nn.Module):
    """Encode a raw depth map into latent tokens for Det3DHead.

    Takes (B, 1, H, W) depth in meters, downsamples, and projects
    to (B, H'*W', latent_dim) tokens.

    The encoding includes:
    - Log depth (handles large range)
    - Depth validity mask
    - Spatial position (normalized x, y coordinates)
    - Learnable projection to latent_dim
    """

    def __init__(
        self,
        latent_dim: int = 128,
        downsample: int = 16,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.downsample = downsample

        # Input: [log_depth, valid_mask, norm_x, norm_y] = 4 channels
        self.proj = nn.Sequential(
            nn.Linear(4, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(
        self,
        depth: Tensor,
        image_hw: tuple[int, int],
    ) -> Tensor:
        """Encode depth map to latent tokens.

        Args:
            depth: (B, 1, H, W) depth in meters. 0 = invalid.
            image_hw: (H, W) original image size.

        Returns:
            depth_latents: (B, H'*W', latent_dim) tokens for Det3DHead.
        """
        B = depth.shape[0]
        H, W = image_hw
        h_down = H // self.downsample
        w_down = W // self.downsample

        # Downsample depth to match ray embedding resolution
        depth_down = F.adaptive_avg_pool2d(depth, (h_down, w_down))  # (B, 1, h', w')

        # Log depth (safe for zeros)
        valid = (depth_down > 0.01).float()
        log_depth = torch.log(depth_down.clamp(min=0.01)) * valid  # (B, 1, h', w')

        # Normalized spatial coordinates
        ys = torch.linspace(0, 1, h_down, device=depth.device)
        xs = torch.linspace(0, 1, w_down, device=depth.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 1, h', w')
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        # Stack features: [log_depth, valid, x, y]
        features = torch.cat([log_depth, valid, grid_x, grid_y], dim=1)  # (B, 4, h', w')

        # Flatten to tokens
        features = features.flatten(2).permute(0, 2, 1)  # (B, h'*w', 4)

        # Project to latent dim
        latents = self.proj(features)  # (B, h'*w', latent_dim)

        return latents
