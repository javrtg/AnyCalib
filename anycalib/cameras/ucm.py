import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera


class UCM(BaseCamera):
    """Implementation of the Unified Camera Model (UCM) [1, Sec. II].

    The (ordered) intrinsic parameters are fx, fy, cx, cy, ξ:
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - ξ represents the distance from the center of projection to the center of the
            sphere and controls the magnitude of radial distortion present in the image.


    [1] Single View Point Omnidirectional Camera Calibration from Planar Grids.
        C Mei, P Rives, ICRA 2007.
    """

    NAME = "ucm"
    # number of focal lengths
    NUM_F = 2
    PARAMS_IDX = {
        "fx": 0,
        "fy": 1,
        "cx": 2,
        "cy": 3,
        "k1": 4,  # ξ
    }
    num_k = 1  # ξ

    def __init__(self, safe_optim: bool = True):
        self.safe_optim = safe_optim  # whether to ensure ξ >= 0 during nonlin optim.

    def parse_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Parse parameters into focal lengths, principal points, and distortion.

        This method is useful for creating a unified API between Radial and SimpleRadial
        camera models.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., 2) focal lengths, (..., 2) principal points, (..., num_k) distortion.
        """
        self.validate_params(params)
        f = params[..., : self.NUM_F]
        c = params[..., self.NUM_F : self.NUM_F + 2]
        k = params[..., -1:]
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

    def project(self, params: Tensor, points_3d: Tensor) -> tuple[Tensor, Tensor]:
        """Project 3D points in the reference of the camera to image coordinates.

        Args:
            points_3d: (..., N, 3) 3D points in the reference of each camera.
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., N, 2) image coordinates.
            (..., N) boolean tensor indicating valid projections.
        """
        self.validate_params(params)
        xi = params[..., -1:]  # (..., 1)
        b = F.normalize(points_3d, dim=-1)
        im_coords = (
            params[..., None, :-3] * b[..., :2] / (xi.unsqueeze(-1) + b[..., 2:])
            + params[..., None, -3:-1]
        )
        # 3d points that are in a range with unique (valid) projection
        valid = b[..., 2] > -torch.minimum(
            xi, xi.clamp(torch.finfo(xi.dtype).eps).reciprocal()
        )
        return im_coords, valid

    def unproject(self, params: Tensor, points_2d: Tensor) -> tuple[Tensor, Tensor]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3) unit bearing vectors in the camera frame.
            (..., N) boolean tensor indicating valid unprojections.
        """
        self.validate_params(params)
        xi = params[..., None, -1:]  # (..., 1, 1)
        m = (points_2d - params[..., None, -3:-1]) / params[..., None, :-3]
        r2 = (m**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        tmp = 1 + (1 - xi**2) * r2
        valid = tmp.squeeze(-1) >= 0
        fac = (xi + tmp.clamp(0).sqrt()) / (1 + r2)
        # NOTE: invalid bearings generally will not have a unit norm
        unit_bearings = torch.cat((fac * m, fac - xi), dim=-1)
        return unit_bearings, valid

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
        is_simple = self.NUM_F == 1
        if is_simple:
            cxcy_, info0 = ut.cxcy_from_rays(im_coords, bearings)
        else:
            cxcy_, pix_ar, info0 = ut.cxcy_and_pix_ar_from_rays(im_coords, bearings)
        cxcy_ = cxcy_ if cxcy is None else cxcy
        # form system. The two constraints of one correspondence are linearly dependent
        # so we use instead the first constraint of two different correspondences
        imx_cx = im_coords[..., :2, 0] - cxcy_[..., :1]  # (..., 2)
        A = torch.stack((bearings[..., :2, 0], -imx_cx), dim=-1)  # (..., 2, 2)
        b = bearings[..., :2, 2] * imx_cx  # (..., 2)
        sol, info = torch.linalg.solve_ex(A, b)  # (..., 2)
        sol[(info != 0) | (info0 != 0)] = 1.1  # set bogus value for invalid solutions
        fx = sol[..., :1]
        if is_simple:
            return torch.cat((fx, cxcy_.expand_as(sol), sol[..., 1:]), dim=-1)
        return torch.cat(
            (fx, pix_ar.unsqueeze(-1) * fx, cxcy_.expand_as(sol), sol[..., 1:]), dim=-1
        )

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
        if covs is not None:
            raise NotImplementedError
        is_simple = self.NUM_F == 1
        if is_simple:
            cxcy_, info1 = ut.cxcy_from_rays(im_coords, bearings)
            pix_ar = im_coords.new_ones(())
        else:
            cxcy_, pix_ar, info1 = ut.cxcy_and_pix_ar_from_rays(im_coords, bearings)
            pix_ar = pix_ar.unsqueeze(-1)
        assert (cxcy_ > 0).all() and (pix_ar > 0).all()
        cxcy_ = cxcy_ if cxcy is None else cxcy  # dirty...

        # form batched system.
        rc = torch.linalg.norm(im_coords - cxcy_[..., None, :], dim=-1)  # (..., N)
        Ra = torch.linalg.norm(
            torch.stack((bearings[..., 0], pix_ar * bearings[..., 1]), dim=-1), dim=-1
        )  # (..., N)
        d = torch.linalg.norm(bearings, dim=-1)  # (..., N)
        A = torch.stack((Ra, -rc * d), dim=-1)  # (..., N, 2)
        b = rc * bearings[..., 2]  # (..., N)
        # solve
        sol, info2 = torch.linalg.solve_ex(
            A.transpose(-1, -2) @ A, (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
        )
        info = torch.where((info1 == 0) & (info2 == 0), 0, -1)

        # check bound ξ >= 0
        on_bounds = sol[..., 1:] >= 0
        if on_bounds.all():
            fx = sol[..., :1]
            if is_simple:
                return torch.cat((fx, cxcy_.expand_as(sol), sol[..., 1:]), dim=-1), info
            params = torch.cat(
                (fx, pix_ar * fx, cxcy_.expand_as(sol), sol[..., 1:]), dim=-1
            )
            return params, info

        # activate inequality (bound) ξ=0 (pinhole) and re-solve
        eps = torch.finfo(Ra.dtype).eps
        valid = (Ra > eps) & (bearings[..., 2] > eps)
        fx = (rc * bearings[..., 2] / Ra.clamp(eps) * valid).sum(-1) / valid.sum(-1).clamp(1)  # fmt: skip

        fx = torch.where(on_bounds, sol[..., :1], fx.unsqueeze(-1))
        if is_simple:
            return (
                torch.cat((fx, cxcy_.expand_as(sol), sol[..., 1:].clamp(0)), dim=-1),
                info,
            )
        params = torch.cat(
            (fx, pix_ar * fx, cxcy_.expand_as(sol), sol[..., 1:].clamp(0)), dim=-1
        )
        return params, info

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
        self.validate_params(params)
        num_f = self.NUM_F
        is_simple = num_f == 1
        if_ = params[..., None, :-3].reciprocal()  # (..., 1, {1, 2})
        c = params[..., None, -3:-1]  # (..., 1, 2)
        xi = params[..., None, -1:]  # (..., 1, 1)

        # intermediate variables
        m = (im_coords - c) * if_  # (..., N, 2)
        r2 = (m**2).sum(dim=-1, keepdim=True)
        ir2p1 = (r2 + 1).reciprocal()
        tmp = torch.sqrt((1 + (1 - xi**2) * r2).clamp(0))  # TODO: flag negative values
        fac = (xi + tmp) * ir2p1  # (..., N, 1)

        dfac_dr2 = (ir2p1 * (0.5 * (1 - xi**2) / tmp - fac)).unsqueeze(-1)  # (.N, 1, 1)
        dr2_dc = -2 * m * if_  # (..., N, 2)
        dr2_dfc = torch.cat(
            ((m * dr2_dc).sum(-1, keepdim=True) if is_simple else m * dr2_dc, dr2_dc),
            dim=-1,
        )  # (..., N, {3, 4})
        db_dfc = dfac_dr2 * torch.cat(
            (m[..., None] * dr2_dfc[..., None, :], dr2_dfc.unsqueeze(-2)),
            dim=-2,
        )  # (..., N, 3, {3, 4})
        # remaining chain rule term
        db_dfc[..., :2, :num_f] += (
            (0.5 * fac * dr2_dc).unsqueeze(-1)
            if is_simple
            else (0.5 * fac * dr2_dc).diag_embed()
        )
        db_dfc[..., :2, num_f:] -= (fac * if_).expand_as(im_coords).diag_embed()

        dfac_dxi = (tmp - r2 * xi) * ir2p1 / tmp  # (..., N, 1)
        db_dxi = torch.cat((m * dfac_dxi, dfac_dxi - 1), dim=-1)

        db_dparams = torch.cat((db_dfc, db_dxi.unsqueeze(-1)), dim=-1)
        return db_dparams

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
        self.validate_params(params)
        if_ = params[..., None, :-3].reciprocal()  # (..., 1, {1, 2})
        c = params[..., None, -3:-1]  # (..., 1, 2)
        xi = params[..., None, -1:]  # (..., 1, 1)

        # intermediate variables
        m = (im_coords - c) * if_  # (..., N, 2)
        r2 = (m**2).sum(dim=-1, keepdim=True)
        ir2p1 = (r2 + 1).reciprocal()
        tmp = torch.sqrt((1 + (1 - xi**2) * r2).clamp(0))  # TODO: flag negative values
        fac = (xi + tmp) * ir2p1  # (..., N, 1)

        dfac_dr2 = (ir2p1 * (0.5 * (1 - xi**2) / tmp - fac)).unsqueeze(-1)  # (.N, 1, 1)
        dr2_dim = 2 * m * if_  # (..., N, 2)
        dr2_dim = dfac_dr2 * torch.cat(
            (m[..., None] * dr2_dim[..., None, :], dr2_dim.unsqueeze(-2)),
            dim=-2,
        )
        dr2_dim[..., :2, :] += (fac * if_).expand_as(im_coords).diag_embed()
        return dr2_dim

    def fit_dist_from_radii(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, None]:
        """Fit distortion parameters (ξ) from sensor and ray radii.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays.

        Returns:
            (..., 1) distortion parameter (ξ).
            None, to indicate that all estimations are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
        """
        valid = r >= torch.finfo(r.dtype).eps
        r = torch.where(valid, r, 1)  # avoid division by zero
        xi = ((R / r - Z) * valid).sum(-1) / valid.sum(-1).clamp(1)  # masked mean
        return xi.unsqueeze(-1).clamp(0), None

    def get_optim_update(self, params: Tensor, delta: Tensor) -> Tensor:
        """Undo reparameterization of ξ within delta and apply the update.

        During optimization the following should be satisfied:
            * ξ ∈ [0, ∞)
        To ensure this, we optimize for
            * ξ_hat in ξ = ξ_hat^2
        Because of this, we need to undo this reparameterization before applying the update.

        Args:
            params: (..., D) intrinsic parameters.
            delta: (..., D) update.

        Returns:
            (..., D) updated intrinsic parameters.
        """
        if not self.safe_optim:
            return params + delta

        d_xihat = delta[..., -1:]
        xi = params[..., -1:]
        return torch.cat(
            (
                params[..., :-1] + delta[..., :-1],
                xi + 2 * xi.sqrt() * d_xihat + d_xihat**2,
            ),
            dim=-1,
        )

    def get_optim_jac(self, jac: Tensor, params: Tensor) -> Tensor:
        """If needed, apply chain rule for reparameterized parameters during optimization.

        During optimization the following should be satisfied:
            * ξ ∈ [0, ∞)
        To ensure this, we optimize for
            * ξ_hat in ξ = ξ_hat^2
        Because of this, we apply the chain rule to get the Jacobian w.r.t. the
        *reparameterized* parameter.

        Args:
            jac: (..., N, D) Jacobian.
            params: (..., D) intrinsic parameters.

        Returns:
            (..., N, D) updated Jacobian.
        """
        if not self.safe_optim:
            return jac

        return torch.cat(
            (jac[..., :-1], jac[..., -1:] * (2 * params[..., -1:, None].sqrt())),
            dim=-1,
        )

        # # NOTE: dσ(x)/dx = σ(x)*(1 - σ(x))
        # a = params[..., -2, None, None]  # alpha
        # b = params[..., -1, None, None]  # beta
        # db_dbhat = (b - self.beta_min) * (self.beta_max - b) / self.beta_ptp
        # return torch.cat(
        #     (jac[..., :-2], a * (1 - a) * jac[..., -2, None], db_dbhat * jac[..., -1:]),
        #     dim=-1,
        # )

    def fit_from_radii_reparametrized(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, None]:
        """Fit distortion parameters (ξ) from sensor and ray radii.

        This class estimates a reparametrized version of the distortion parameter (ξ),
        such that the focal length is closer in magnitude to that of other camera models.
        In this version, the projection funcion (mapping from ray radii to sensor radii)
        is given by: r = R (ξ + 1) / (ξ + Z). This method assumes that R^2 + Z^2 = 1.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays. It is assumed that R^2 + Z^2 = 1.

        Returns:
            (..., 1) distortion parameter (ξ).
            None, to indicate that all estimations are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
        """
        num = R - r * Z
        den = r - R
        valid = den.abs() >= torch.finfo(den.dtype).eps
        den = torch.where(valid, den, 1)  # avoid division by zero
        # masked mean
        xi = ((num / den) * valid).sum(-1) / valid.sum(-1).clamp(1)
        # return xi.unsqueeze(-1).clamp(0), None
        return xi.unsqueeze(-1), None
