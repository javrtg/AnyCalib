import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anycalib.manifolds import Unit3


def cvx_upsample(x: Tensor, mask: Tensor, up_factor: int = 7) -> Tensor:
    """Upsample [H/k, W/k, C] -> [H, W, C] using convex combination of 3x3 patches.

    Code adapted from RAFT (Teed and Deng, 2020):
    https://github.com/princeton-vl/RAFT/blob/3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02/core/raft.py#L72

    Args:
        x: (N, C, H, W) input tensor
        mask: (N, 9, 1, 1, H, W) already softmaxed mask tensor
        up_factor: upsample factor
    """
    N, C, H, W = x.shape
    up_x = F.unfold(x, (3, 3), padding=1)
    up_x = up_x.view(N, C, 9, 1, 1, H, W)
    up_x = torch.sum(mask * up_x, dim=2)
    up_x = up_x.permute(0, 1, 4, 2, 5, 3)
    return up_x.reshape(N, C, up_factor * H, up_factor * W)


class ConvexTangentDecoder(nn.Module):
    """Convex Tangent Coordinates Decoder.

    This decoder predicts 2D coordinates in the tangent space of the unit sphere at the
    optical axis: z_1 = [0, 0, 1]. These coordinates are subsequently mapped to unit
    rays using the exponential map.

    Args:
        in_channels: number of input channels
        up_factor: upsampling factor
    """

    def __init__(self, in_channels: int = 256, up_factor: int = 7):
        super().__init__()
        self.in_channels = in_channels
        self.up_factor = up_factor

        # tangent head
        self.tangent_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # tangent coords (2)
            nn.Conv2d(in_channels // 2, 2, 1),
        )
        # weights head for convex upsampling to input resolution
        self.upsampling_weights_head = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, up_factor**2 * 9, 1, padding=0),
            nn.Unflatten(1, (1, 9, up_factor, up_factor)),
            nn.Softmax(dim=2),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | float]:
        # head
        tangent_pred = self.tangent_head(x)  # (B, 5, H/7, W/7)
        weights = self.upsampling_weights_head(x)
        # upsample
        tangent_pred = cvx_upsample(tangent_pred, weights, self.up_factor)
        # postprocess
        tangent_coords = tangent_pred[:, :2]
        rays = Unit3.expmap_at_z1(tangent_coords.permute(0, 2, 3, 1)).permute(
            0, 3, 1, 2
        )
        out: dict[str, Tensor | float] = {
            "rays": rays,
            "tangent_coords": tangent_coords,
        }
        return out
