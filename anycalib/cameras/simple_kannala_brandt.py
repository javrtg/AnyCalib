import torch
from torch import Tensor

from anycalib.cameras.kannala_brandt import KannalaBrandt


class SimpleKannalaBrandt(KannalaBrandt):
    """Kannala-Brandt camera model [1].

    We use the common [2, 3, 4] slight variation of the original model [1] which sets
    k1 in [1, eq. 6] to 1.0. Thus the radial projection is defined as:
        r(θ) = θ + k1 * θ^3 + k2 * θ^5 + k3 * θ^7 + k4 * θ^9,
    where θ is the incidence angle of the incoming ray, computed as
        θ = atan2(sqrt(x^2 + y^2), z).
    for a 3D point with coordinates (x, y, z).
    Additionally, this implementation allows the use of a variable number of coefficients.

    [1] A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and
        Fish-Eye Lenses, J. Kannala, S. Brandt, PAMI 2006.
    [2] The Double Sphere Camera Model, V. Usenko et al., 3DV 2018.
    [3] BabelCalib: A Universal Approach to Calibrating Central Cameras, Y. Lochman et
        al., ICCV 2021.
    [4] Project Aria (KB3 camera model), Meta Reality Labs Research, 2023.

    The (ordered) intrinsic parameters are f, cx, cy, k1, k2, ...,
        - f [pixels] is the focal length,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    Args:
        num_k: number of radial distortion coefficients. Default is 4.
        newton_iters: number of Newton iterations for mapping sensor radii to polar angles (θ)
        newton_tol: threshold for checking convergence of the Newton algorithm.
    """

    NAME = "simple_kb"
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

    def _form_batched_system(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Form the 2D equations for each 2D-3D correspondence.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            As: (..., N, 2, {1 .. 4} + num_k) design matrices (without stacking).
            bs: (..., N, 2) observations.
        """
        num_k = self.num_k
        # ray radii and polar angles
        ray_radii = torch.linalg.norm(bearings[..., :2], dim=-1, keepdim=True)
        theta = torch.atan2(ray_radii, bearings[..., 2:])  # (..., N, 1)
        # form eqs corresponding to focal length(s) and principal point
        if cxcy is None:
            As = bearings.new_zeros((*theta.shape[:-1], 2, 3 + num_k))
            As[..., 0] = im_coords
            As[..., 0, 1] = As[..., 1, 2] = -1
            offset = 3
        else:
            As = bearings.new_zeros((*theta.shape[:-1], 2, 1 + num_k))
            As[..., 0] = im_coords - cxcy[..., None, :]
            offset = 1
        # form RHS (..., N, 2)
        bs = (
            theta
            * bearings[..., :2]
            / ray_radii.clamp(torch.finfo(ray_radii.dtype).eps)
        )
        # eqs corresponding to distortion terms
        theta_2 = theta**2
        coeff_i = -theta_2 * bs
        As[..., offset] = coeff_i
        for i in range(1, num_k):
            coeff_i = coeff_i * theta_2
            As[..., offset + i] = coeff_i
        return As, bs
