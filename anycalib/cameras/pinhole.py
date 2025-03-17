from math import pi, tan

import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3

DEG2RAD = pi / 180.0


def check_within_fov(bearings: Tensor, max_fov: float) -> Tensor:
    """Check which bearings are within the admissible field of view.

    Perspective projection becomes unstable for points/bearings whose incidence angle
    is typically above 60 degrees (at FoV of 120 deg). This function masks out bearings
    whose incidence angle is above the admissible field of view `max_fov`.

    Args:
        bearings: (..., 3) unit bearing vectors in the camera frame.
        max_fov: Threshold in degrees for masking out bearings/rays whose incidence
            angles are above this admissible field of view.

    Returns:
        (...,) boolean tensor.
    """
    max_fov = min(max_fov, 179.0)
    radii = torch.linalg.norm(bearings[..., :2], dim=-1)
    valid = radii < bearings[..., 2] * tan(0.5 * max_fov * DEG2RAD)  # (..., N)
    return valid


def propagate_tangent_covs(bearings: Tensor, proj: Tensor, covs: Tensor) -> Tensor:
    """Propagate covariances expressed in the tangent space of the bearings to the
    space where errors are linear w.r.t. the intrinsic parameters.

    The propagation is done to the reparameterized pinhole error space:
        e_x = X/Z - a_x*u + b_x,
        e_y = Y/Z - a_y*v + b_y,
    where a = 1/f, and b = c/f.

    Args:
        bearings: (..., N, 3) unit bearing vectors in the camera frame.
        proj: (..., N, 2) perspective projection of the bearings (to save computation).
        covs: (..., N, 2) diagonal elements of the covariances expressed in the tangent
            space of the input bearings.

    Returns:
        (..., N, 2, 2) error covariances.
    """
    # Jacobian of the error w.r.t. point in unit sphere (..., N, 2, 3)
    derr_dp = covs.new_zeros((*covs.shape, 3))
    derr_dxy = 1 / bearings[..., 2:].clamp(torch.finfo(bearings.dtype).eps)
    derr_dz = -proj * derr_dxy
    derr_dp[..., 0, 0] = derr_dp[..., 1, 1] = derr_dxy[..., 0]
    derr_dp[..., 0, 2] = derr_dz[..., 0]
    derr_dp[..., 1, 2] = derr_dz[..., 1]
    # derr_dp[..., [0, 1], [0, 1]] = derr_dxy
    # derr_dp[..., [0, 1], [2, 2]] = derr_dz
    # Jacobian of the error w.r.t. the coordinates in the tangent plane
    derr_dbasis = derr_dp @ Unit3.get_tangent_basis(bearings)
    error_covs = ut.fast_small_matmul(
        derr_dbasis * covs[..., None, :], derr_dbasis.transpose(-1, -2)
    )
    return error_covs


class Pinhole(BaseCamera):
    """Pinhole camera model.

    This class uses as intrinsic parameters (in order) fx, fy, cx, cy, where:
        - (fx, fy) [pixels] are the the focal lengths,
        - (cx, cy) [pixels] is the principal points.

    Args:
        max_fov: Threshold in degrees for masking out bearings/rays whose incidence
            angles correspond to fovs above this admissible field of view.
    """

    NAME = "pinhole"
    NUM_F = 2
    PARAMS_IDX = {"fx": 0, "fy": 1, "cx": 2, "cy": 3}
    num_k = 0  # no distortion parameters

    def __init__(self, max_fov: float = 170):
        if not (0 < max_fov < 180):
            raise ValueError(f"'max_fov' must be in (0, 180) but got: {max_fov}.")
        self.max_fov = max_fov

    @classmethod
    def create_from_params(cls, params: Tensor | None = None) -> BaseCamera:
        return cls()

    @classmethod
    def create_from_id(cls, id_: str | None = None) -> BaseCamera:
        return cls()

    @property
    def id(self) -> str:
        return self.NAME

    def validate_params(self, params: Tensor):
        if params.shape[-1] != self.NUM_F + 2:
            total = self.NUM_F + 2
            params_str = ("fx, fy" if self.NUM_F == 2 else "f") + ", cx, cy"
            raise ValueError(
                f"'params' must have shape (..., {total}), representing (...) cameras "
                f"and {total} intrinsic parameters per camera: {params_str}. However, "
                f"got: {params.shape=}."
            )

    @staticmethod
    def get_min_sample_size(with_cxcy: bool) -> int:
        """Minimal number of 2D-3D samples needed to fit the intrinsics."""
        return 1 if with_cxcy else 2

    def project(self, params: Tensor, points_3d: Tensor) -> tuple[Tensor, Tensor]:
        """Project 3D points in the reference of the camera to image coordinates.

        Args:
            params: (..., 4) intrinsic parameters (fx, fy, cx, cy) for (...) cameras.
            points_3d: (..., N, 3) 3D points in the reference of each camera.

        Returns:
            (..., N, 2) image coordinates.
            (..., N) boolean tensor indicating valid projections.
        """
        self.validate_params(params)
        # perspective projection
        proj = points_3d[..., :2] / points_3d[..., 2:].clamp(torch.finfo(points_3d.dtype).eps)  # fmt: skip
        # image coordinates (in pixels)
        num_f = self.NUM_F
        im_coords = params[..., None, :num_f] * proj + params[..., None, num_f:]
        return im_coords, points_3d[..., 2] > 0

    def unproject(self, params: Tensor, points_2d: Tensor) -> tuple[Tensor, None]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., 4) intrinsic parameters (fx, fy, cx, cy) for (...) cameras.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            unit_bearings: (..., N, 3) unit bearing vectors in the camera frame.
            valid: None, to indicate that all unprojections are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
        """
        self.validate_params(params)
        nf = self.NUM_F
        # normalize image coordinates
        unproj = (points_2d - params[..., None, nf:]) / params[..., None, :nf]
        # bearing vectors
        bearings = torch.cat((unproj, unproj.new_ones((*unproj.shape[:-1], 1))), dim=-1)
        return F.normalize(bearings, dim=-1), None

    def fit_minimal(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> Tensor:
        """Fit instrinsics with a minimal set of 2D-bearing correspondences.

        The pinhole projection in the x-image direction is:
            u = fx * (X/Z) + cx
        Thus, we can recover the focal length fx and principal point (cx) by
            fx = (u0 - u1) / (X0/Z0 - X1/Z1)
            cx = u0 - fx*X0/Z0
        The same applies for the y-image direction.
        When cxcy is given, the equations are simplified further but care should be
        taken with its input shape: In RANSAC, tipically, im_coords and bearings will
        have a shape of (..., N_samples, 1, 2) and (..., N_samples, 1, 3), respectively.
        Thus, cxcy should have a shape of
            a) (..., 1, 2) to allow broadcasting across all samples, or
            b) should be expanded: (..., N_samples, 2) for the same purpose.

        Args:
            im_coords: (..., MIN_SIZE, 2) image coordinates.
            bearings: (..., MIN_SIZE, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            (..., 4) fitted intrinsic parameters: fx, fy, cx, cy.
        """
        eps = torch.finfo(bearings.dtype).eps
        assert (
            self.get_min_sample_size(cxcy is not None)
            == im_coords.shape[-2]
            == bearings.shape[-2]
        ), "Input sizes do not match the minimal sample size."

        if cxcy is None:
            proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)
            proj_diff = proj.diff(dim=-2).abs().clamp(eps)
            f = (im_coords.diff(dim=-2) / proj_diff).squeeze(-2).abs()
            c = im_coords[..., 0, :] - f * proj[..., 0, :]
            return torch.cat((f, c), dim=-1)

        iproj = bearings[..., 2:] / bearings[..., :2].abs().clamp(eps)  # (Z/XY)
        f = ((im_coords - cxcy[..., None, :]) * iproj).squeeze(-2).abs()  # (..., 2)
        return torch.cat((f, cxcy.expand_as(f)), dim=-1)

    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        *ignored_args,
        **ignored_kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Fit instrinsics with a set of 2D-bearing correspondences.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances expressed in the
                tangent space of the input bearings.

        Returns:
            (..., 4) fitted intrinsic parameters: fx, fy, cx, cy.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found (<0) or the system is singular (>0).
        """
        eps = torch.finfo(bearings.dtype).eps
        # perspective projection
        proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)  # (..., N, 2)
        valid = check_within_fov(bearings, self.max_fov)  # (..., N)
        bs = proj  # (..., N, 2) observations

        # propagate covariances if present (..., N, 2, 2)
        if covs is not None:
            e_covs = propagate_tangent_covs(
                bearings, proj, covs.clamp(torch.finfo(covs.dtype).eps)
            )
            Ws, info = torch.linalg.inv_ex(e_covs)
            valid = valid & (info == 0)
        else:
            Ws = None

        if cxcy is None:
            # form linear system
            As = proj.new_zeros((*proj.shape, 4))
            # normalize first two columns for improved conditioning number
            norm_factor = im_coords.amax(-2, keepdim=True)  # (..., 1, 2)
            As[..., 0, 0] = im_coords[..., 0] / norm_factor[..., 0]
            As[..., 1, 1] = im_coords[..., 1] / norm_factor[..., 1]
            As[..., 0, 2] = As[..., 1, 3] = -1
            intrinsics, info = ut.solve_2dweighted_lstsq(As, bs, Ws, valid)
            f = norm_factor.squeeze(-2) * intrinsics[..., :2].reciprocal()
            c = intrinsics[..., 2:] * f
            return torch.cat((f, c), dim=-1), info

        As = torch.diag_embed(im_coords - cxcy[..., None, :])  # (..., N, 2, 2)
        invf, info = ut.solve_2dweighted_lstsq(As, bs, Ws, valid)  # (..., 2)
        return torch.cat((invf.reciprocal(), cxcy), dim=-1), info

    def jac_bearings_wrt_params(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Jacobian of the bearings w.r.t. the intrinsic parameters.

        Args:
            params: (..., {3, 4}) intrinsic parameters ({f, (fx, fy)}, cx, cy).
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, {3, 4}) Jacobians.
        """
        self.validate_params(params)
        eps = torch.finfo(bearings.dtype).eps
        x = bearings / bearings[..., 2:].clamp(eps)
        proj = x[..., :2]  # (..., N, 2)
        x_inorm = 1 / torch.linalg.norm(x, dim=-1, keepdim=True)[..., None]
        db_dmxy = x_inorm * (
            x.new_zeros((3, 2)).fill_diagonal_(1)
            - x_inorm**2 * x[..., None] * proj[..., None, :]
        )  # (..., N, 3, 2)

        num_f = self.NUM_F
        i_nf = -params[..., None, :num_f].reciprocal()  # (..., 1, {1, 2})
        dmxy_dfc = torch.cat(
            (
                (proj * i_nf).diag_embed() if num_f == 2 else (proj * i_nf)[..., None],
                i_nf.expand_as(proj).diag_embed(),
            ),
            dim=-1,
        )  # (..., N, 2, 4)
        return ut.fast_small_matmul(db_dmxy, dmxy_dfc)

    def jac_bearings_wrt_imcoords(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Jacobian of the bearings w.r.t. the intrinsic image coordinates.

        Args:
            params: (..., {3, 4}) intrinsic parameters ({f, (fx, fy)}, cx, cy).
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, 2) Jacobians.
        """
        self.validate_params(params)
        eps = torch.finfo(bearings.dtype).eps
        x = bearings / bearings[..., 2:].clamp(eps)
        proj = x[..., :2]  # (..., N, 2)
        x_inorm = 1 / torch.linalg.norm(x, dim=-1, keepdim=True)[..., None]
        db_dmxy = x_inorm * (
            x.new_zeros((3, 2)).fill_diagonal_(1)
            - x_inorm**2 * x[..., None] * proj[..., None, :]
        )  # (..., N, 3, 2)
        db_dim = db_dmxy / params[..., None, None, : self.NUM_F]
        return db_dim
