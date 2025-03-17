import torch
from torch import Tensor

from anycalib.cameras import CameraFactory

PARAMS = {
    "pinhole": torch.tensor([800.0, 700, 320, 240]),  # fx, fy, cx, cy
}


def get_2d3d_correspondences(
    cam_id: str,
    params: Tensor,
    n: int = 10,
    outlier_ratio: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Generate 2D-3D correspondences."""
    rng = torch.Generator().manual_seed(0)
    unit_bearings = torch.rand((n, 3), generator=rng) + 1e-7  # ensure FoV < 180 deg
    unit_bearings /= unit_bearings.norm(dim=-1, keepdim=True)
    im_coords = CameraFactory.project(cam_id, params, unit_bearings)
    return im_coords, unit_bearings
