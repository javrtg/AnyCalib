import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib.cameras.base import BaseCamera


class FOV(BaseCamera):
    """Implementation of the "FOV" camera model [1].

    NOTE: This projection model is not linear on the parameters. Because of this,
    this class just implements the projection and unprojection methods, but not
    the fitting methods.

    The (ordered) intrinsic parameters are fx, fy, cx, cy, ω:
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - ω represents the field of view of the ideal corresponding fisheye-lens.


    [1] Straight lines have to be straight: automatic calibration and removal of
        distortion from scenes of structured enviroments. Deverney and Faugeras, 2001.
    """

    NAME = "fov"
    # number of focal lengths
    NUM_F = 2
    PARAMS_IDX = {
        "fx": 0,
        "fy": 1,
        "cx": 2,
        "cy": 3,
        "k1": 4,  # ω
    }
    num_k = 1  # ω

    def __init__(self, safe_optim: bool = True, omega_min: float = 0.15):
        self.safe_optim = safe_optim
        self.omega_min = omega_min

    def parse_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Parse parameters into focal lengths, principal points, and distortion.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., 2) focal lengths, (..., 2) principal points, (..., num_k) distortion.
        """
        self.validate_params(params)
        f = params[..., : self.NUM_F]
        c = params[..., self.NUM_F : self.NUM_F + 2]
        k = params[..., self.NUM_F + 2 :]
        return f, c, k

    @classmethod
    def create_from_params(cls, params: Tensor | None = None) -> BaseCamera:
        return cls()

    @classmethod
    def create_from_id(cls, id_) -> BaseCamera:
        assert id_ == cls.NAME, f"Expected id='{cls.NAME}', but got: {id_}."
        return cls()

    @property
    def id(self) -> str:
        return self.NAME

    def validate_params(self, params: Tensor):
        if params.shape[-1] != self.NUM_F + 3:
            num_f = self.NUM_F
            total = num_f + 3
            params_str = ("fx, fy" if self.NUM_F == 2 else "f") + ", cx, cy, k1"
            raise ValueError(
                f"Expected (..., {total}) parameters as input, representing (...) "
                f"cameras and {total} intrinsic params per cameras: {params_str}, i.e. "
                f"with {num_f} focal(s) and 1 distortion coefficient, but got: "
                f"{params.shape[-1]=}."
            )

    def get_min_sample_size(self, with_cxcy: bool) -> int:
        """Minimal number of 2D-3D samples needed to fit the intrinsics."""
        # total_params = (self.NUM_F if with_cxcy else self.NUM_F + 2) + 1
        # return ceil(0.5 * total_params)
        return 2 if self.NUM_F == 1 else 3

    def project(self, params: Tensor, points_3d: Tensor) -> tuple[Tensor, None]:
        """Project 3D points in the reference of the camera to image coordinates.

        Args:
            points_3d: (..., N, 3) 3D points in the reference of each camera.
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., N, 2) image coordinates.
            valid: None, to indicate that all projections are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
                Actually, the FOV model is undefined for the 3D point (0, 0, 0). This is
                not currently flagged (maybe a TODO). Due to Pytorch's convention of
                atan2(0, 0)=0, this point is projected to the principal point (cx, cy).
        """
        self.validate_params(params)
        w = params[..., -1:]  # (..., 1)
        R = torch.linalg.norm(points_3d[..., :2], dim=-1)  # (..., N)
        r = torch.arctan2(2 * R * torch.tan(0.5 * w), points_3d[..., 2]) / w  # (..., N)
        m = (
            r.unsqueeze(-1)
            * points_3d[..., :2]
            / R.unsqueeze(-1).clamp(torch.finfo(R.dtype).eps)
        )
        im_coords = params[..., None, :-3] * m + params[..., None, -3:-1]
        return im_coords, None

    def unproject(self, params: Tensor, points_2d: Tensor) -> tuple[Tensor, None]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3) unit bearing vectors in the camera frame.
            valid: None, to indicate that all unprojections are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
        """
        self.validate_params(params)
        w = params[..., None, -1:]  # (..., 1, 1)
        m = (points_2d - params[..., None, -3:-1]) / params[..., None, :-3]
        r = torch.linalg.norm(m, dim=-1, keepdim=True)  # (..., N, 1)
        xy_factor = torch.sin(r * w) / (2 * r * torch.tan(0.5 * w))
        unit_bearings = torch.cat((xy_factor * m, torch.cos(r * w)), dim=-1)
        return F.normalize(unit_bearings, dim=-1), None

    def fit_minimal(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> Tensor:
        """Fit instrinsics with a minimal set of 2D-bearing correspondences.

        Args:
            im_coords: (..., MIN_SAMPLE_SIZE, 2) image coordinates.
            bearings: (..., MIN_SAMPLE_SIZE, 3) unit bearing vectors.

        Returns:
            (..., D) intrinsic parameters.
        """
        raise NotImplementedError

    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        params0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fit instrinsics with a set of 2D-bearing correspondences.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances expressed in the
                tangent space of the input bearings. Currently not supported.
            params0: (..., D) approximate estimation of the intrinsic parameters to
                propagate the covariances from the tangent space to the error space.
                Currently not supported.

        Returns:
            (..., D) intrinsic parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found and/or the system is singular.
        """
        raise NotImplementedError

    def jac_bearings_wrt_params(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings wrt the intrinsic parameters.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, D) Jacobians.
        """
        raise NotImplementedError

    def jac_bearings_wrt_imcoords(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings w.r.t. the image coordinates.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, 2) Jacobians.
        """
        raise NotImplementedError
