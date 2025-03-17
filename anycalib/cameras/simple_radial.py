import torch
from torch import Tensor

from anycalib.cameras.radial import Radial


class SimpleRadial(Radial):
    """Simple pinhole camera model with polynomial radial distortion.

    Projection:
        x = f * (X / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cx
        y = f * (Y / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cy
    The (ordered) intrinsic parameters are fx, fy, cx, cy, k1, k2, ...,
        - f [pixels] is the focal length,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    Args:
        max_fov: Threshold in degrees for masking out bearings/rays whose incidence
            angles correspond to fovs above this admissible field of view.
        num_k: number of radial distortion coefficients. Default is 1.
        undist_iters: number of Newton iterations for undistorting radii.
        undist_tol: threshold for checking convergence of the Newton algorithm.
    """

    NAME = "simple_radial"
    # number of focal lengths
    NUM_F = 1
    PARAMS_IDX = {
        "f": 0,
        "cx": 1,
        "cy": 2,
        "k1": 3,
        "k2": 4,
        "k3": 5,
        "k4": 6,
    }

    def __init__(
        self,
        max_fov: float = 170,
        num_k: int = 1,
        undist_iters: int = 25,
        undist_tol: float = 1e-5,
    ):
        if not (0 < max_fov < 180):
            raise ValueError(f"`max_fov` must be in (0, 180) but got: {max_fov}.")
        if num_k <= 0 or not isinstance(num_k, int):
            raise ValueError(f"`num_k` must be a positive integer but got: {num_k}.")
        self.max_fov = max_fov
        self.num_k = num_k
        self.undist_iters = undist_iters
        self.undist_tol = undist_tol

    def _form_batched_system(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Form the 2D equations for each 2D-3D correspondence.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            As: (..., N, 3 + num_k) design matrices (without stacking).
            bs: (..., N, 2) observations.
        """
        eps = torch.finfo(bearings.dtype).eps
        num_k = self.num_k
        # perspective projection
        proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)  # (..., N, 2)
        # form linear system
        if cxcy is None:
            As = proj.new_zeros((*proj.shape, 3 + num_k))
            As[..., 0] = im_coords
            As[..., 0, 1] = As[..., 1, 2] = -1
            offset = 3
        else:
            As = proj.new_zeros((*proj.shape, 1 + num_k))
            As[..., 0] = im_coords - cxcy[..., None, :]
            offset = 1
        # distortion terms
        radii_u2 = (proj * proj).sum(-1, keepdim=True)  # (..., N, 1)
        proj_radii = -proj * radii_u2  # (..., N, 2)
        As[..., offset] = proj_radii
        for i in range(1, num_k):
            proj_radii = proj_radii * radii_u2
            As[..., offset + i] = proj_radii
        return As, proj
