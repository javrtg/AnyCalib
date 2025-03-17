import importlib
from math import ceil

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.special import logit

import anycalib.utils as ut
from anycalib.cameras.base import BaseCamera

# from anycalib.cameras.factory import CameraFactory


class EUCM(BaseCamera):
    """Implementation of the Enhanced Unified Camera Model (EUCM) [1, Sec. II].

    The (ordered) intrinsic parameters are fx, fy, cx, cy, xi:
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - α
        - β


    [1] An Enhanced Unified Camera Model. B. Khomutenko et al., RA-L 2015.
    """

    NAME = "eucm"
    # number of focal lengths
    NUM_F = 2
    PARAMS_IDX = {
        "fx": 0,
        "fy": 1,
        "cx": 2,
        "cy": 3,
        "k1": 4,  # alpha
        "k2": 5,  # beta
    }
    num_k = 2

    def __init__(
        self,
        proxy_cam_id: str = "kb:3",
        proxy_cam_id_sac: str = "kb:2",
        safe_optim: bool = True,
        beta_optim_min: float = 1e-6,
        beta_optim_max: float = 1e2,
    ):
        assert "simple" not in proxy_cam_id and "simple" not in proxy_cam_id_sac
        # FIXME: ugly import to avoid circular imports
        CameraFactory = importlib.import_module("anycalib.cameras.factory").CameraFactory  # fmt: skip
        # Intermediate camera model used during linear fitting
        self.proxy_cam: BaseCamera = CameraFactory.create_from_id(proxy_cam_id)
        self.proxy_cam_sac: BaseCamera = CameraFactory.create_from_id(proxy_cam_id_sac)
        self.safe_optim = safe_optim
        # bounds for β during optimization (ignored if safe_optim=False)
        assert beta_optim_max >= beta_optim_min > 0, "β_max >= β_min > 0 not satisfied."
        self.beta_min = beta_optim_min
        self.beta_max = beta_optim_max
        self.beta_ptp = beta_optim_max - beta_optim_min

    def parse_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Parse parameters into focal lengths, principal points, and distortion.

        This method is useful for creating a unified API between Radial and SimpleRadial
        camera models.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., 2) focal lengths, (..., 2) principal points, (..., 2) alpha and beta.
        """
        self.validate_params(params)
        f = params[..., :-4]
        c = params[..., -4:-2]
        k = params[..., -2:]
        return f, c, k

    @classmethod
    def create_from_params(cls, params: Tensor | None = None) -> BaseCamera:
        return cls()

    @classmethod
    def create_from_id(cls, id_) -> BaseCamera:
        assert id_ == "eucm", f"Expected id='eucm', but got: {id_}."
        return cls()

    @property
    def id(self) -> str:
        return "eucm"

    def validate_params(self, params: Tensor):
        if params.shape[-1] != self.NUM_F + 4:
            num_f = self.NUM_F
            total = num_f + 4
            params_str = (
                "fx, fy" if self.NUM_F == 2 else "f"
            ) + ", cx, cy, \u03B1, \u03B2"
            raise ValueError(
                f"Expected (..., {total}) parameters as input, representing (...) "
                f"cameras and {total} intrinsic params per cameras: {params_str}, i.e. "
                f"with {num_f} focal(s) and 2 distortion coefficients, but got: "
                f"{params.shape[-1]=}."
            )

    def get_min_sample_size(self, with_cxcy: bool) -> int:
        """Minimal number of 2D-3D samples needed to fit the intrinsics."""
        num_f = self.NUM_F
        total_params = (num_f if with_cxcy else num_f + 2) + max(
            2, self.proxy_cam_sac.num_k
        )
        return ceil(0.5 * total_params)

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
        eps = torch.finfo(params.dtype).eps
        xy = points_3d[..., :2]
        z = points_3d[..., 2:]
        d = torch.sqrt(
            (params[..., None, -1:] * (xy**2).sum(dim=-1, keepdim=True) + z**2).clamp(eps)  # fmt: skip
        )
        alpha = params[..., None, -2:-1]  # (..., 1, 1)
        im_coords = (
            params[..., None, :-4] * xy / (alpha * d + (1 - alpha) * z)
            + params[..., None, -4:-2]
        )
        # 3d points that are in a range with unique (valid) projection
        eps = torch.finfo(params.dtype).eps
        alpha = alpha.clamp(eps, 1 - eps)
        xi = alpha / (1 - alpha)
        valid = z > -torch.minimum(xi, xi.reciprocal()) * d
        return im_coords, valid.squeeze(-1)

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
        eps = torch.finfo(params.dtype).eps
        alpha = params[..., None, -2:-1]  # (..., 1, 1)
        beta = params[..., None, -1:]
        m = (points_2d - params[..., None, -4:-2]) / params[..., None, :-4]
        r2 = (m**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        tmp = 1 - (2 * alpha - 1) * beta * r2
        valid = tmp.squeeze(-1) >= 0
        mz = (1 - beta * alpha**2 * r2) / (
            alpha * torch.sqrt(tmp.clamp(eps)) + (1 - alpha)
        )
        bearings = F.normalize(torch.cat((m, mz), dim=-1), dim=-1)
        return bearings, valid

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
        # use intermediate camera model to approximate the radial profile that will be
        # used to fit the EUCM model
        ms_kb = self.proxy_cam_sac.get_min_sample_size(cxcy is not None)
        params0_ = self.proxy_cam_sac.fit_minimal(
            im_coords[..., :ms_kb, :], bearings[..., :ms_kb, :], cxcy
        )
        # intermediate variables
        num_f = self.NUM_F
        f_x = params0_[..., None, :1] * bearings[..., :2, :1]
        imx_cx = im_coords[..., :2, :1] - params0_[..., None, num_f : num_f + 1]
        z_imcx = bearings[..., :2, 2:] * imx_cx  # (..., 2, 1)
        # form system. The two constraints of one correspondence are linearly dependent
        # so we use instead the first constraint of two different correspondences
        A = torch.cat(
            (
                (bearings[..., :2, :2] ** 2).sum(dim=-1, keepdim=True) * imx_cx**2,
                2 * z_imcx * (z_imcx - f_x),
            ),
            dim=-1,
        )  # (..., 2, 2)
        b = (f_x**2 - 2 * f_x * z_imcx + z_imcx**2).squeeze(-1)  # (..., 2)
        sol, info = torch.linalg.solve_ex(A, b)
        sol[info != 0] = 0.5  # set bogus value for invalid solutions
        # remaining intrinsics
        alpha = sol[..., -1:].clamp(0, 1)
        beta = sol[..., :1] / (alpha**2).clamp(torch.finfo(alpha.dtype).eps)
        params = torch.cat((params0_[..., : num_f + 2], alpha, beta), dim=-1)
        return params

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
        num_f = self.NUM_F
        # use intermediate camera model to approximate the radial profile that will be
        # used to fit the EUCM model
        params0_, info_0 = self.proxy_cam.fit(im_coords, bearings, cxcy)
        # sensor and ray radii
        m = (im_coords - params0_[..., None, num_f: num_f + 2]) / params0_[..., None, :num_f]  # fmt: skip
        r = torch.linalg.norm(m, dim=-1)
        R = torch.linalg.norm(bearings[..., :2], dim=-1)
        coeffs, info = self.fit_dist_from_radii(r, R, bearings[..., 2])
        info = torch.where((info_0 == 0) & (info == 0), 0, -1)
        params = torch.cat((params0_[..., : num_f + 2], coeffs), dim=-1)
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
        is_simple = self.NUM_F == 1
        nif = -params[..., None, :-4].reciprocal()  # (..., 1, {1, 2})
        alpha = params[..., None, -2:-1]  # (..., 1, 1)
        beta = params[..., None, -1:]

        # intermediate variables
        ab = alpha * beta
        aab = alpha * ab
        mxy = (params[..., None, -4:-2] - im_coords) * nif
        r2 = (mxy**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        mz_root = torch.sqrt((1 - (2 * alpha - 1) * beta * r2).clamp(0))  # TODO: flag negative values # fmt: skip
        mz_den = alpha * mz_root + (1 - alpha)
        mz = (1 - aab * r2) / mz_den
        m_inorm_sq = (r2 + mz**2).reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        m = torch.cat((mxy, mz), dim=-1)  # (..., N, 3)
        mxy_nif = mxy * nif  # (..., N, 2)

        # Jacobian of bearing w.r.t. m
        db_dm = torch.sqrt(m_inorm_sq) * (
            torch.eye(3, device=params.device, dtype=params.dtype)
            + ut.fast_small_matmul(-m_inorm_sq * m.unsqueeze(-1), m.unsqueeze(-2))
        )  # (..., N, 3, 3)

        # chain rule for Jacobian w.r.t. focal lengths and principal point
        db_dfc_mxy = torch.cat(  # db_dmxy * dmxy_dfc
            (
                (
                    (db_dm[..., :2] * mxy_nif.unsqueeze(-2)).sum(-1, keepdim=True)
                    if is_simple
                    else db_dm[..., :2] * mxy_nif.unsqueeze(-2)
                ),
                db_dm[..., :2] * nif.unsqueeze(-2),
            ),
            dim=-1,
        )  # (..., N, 3, {3, 4})
        dmz_dr2 = -aab / mz_den + mz * ab * (alpha - 0.5) / (mz_den * mz_root)  # (..., N, 1) # fmt: skip
        dr2_dfc = 2 * torch.cat(
            (
                (mxy * mxy_nif).sum(-1, keepdim=True) if is_simple else mxy * mxy_nif,
                mxy_nif,
            ),
            dim=-1,
        ).unsqueeze(-2)  # (..., N, 1, {3, 4}) # fmt: skip
        db_dfc = db_dfc_mxy + db_dm[..., 2:] * dmz_dr2[..., None] * dr2_dfc

        # chain rule for Jacobian w.r.t. alpha and beta
        dmz_dalpha = (-2 * ab * r2 + mz * (ab * r2 / mz_root - mz_root + 1)) / mz_den
        dmz_dbeta = alpha * r2 * (mz / mz_root * (alpha - 0.5) - alpha) / mz_den

        db_dparams = torch.cat(
            (
                db_dfc,
                db_dm[..., 2:] * dmz_dalpha[..., None],
                db_dm[..., 2:] * dmz_dbeta[..., None],
            ),
            dim=-1,
        )
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
        if_ = params[..., None, :-4].reciprocal()  # (..., 1, {1, 2})
        alpha = params[..., None, -2:-1]  # (..., 1, 1)
        beta = params[..., None, -1:]

        # intermediate variables
        ab = alpha * beta
        aab = alpha * ab
        mxy = (im_coords - params[..., None, -4:-2]) * if_
        r2 = (mxy**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        mz_root = torch.sqrt((1 - (2 * alpha - 1) * beta * r2).clamp(0))  # TODO: flag negative values # fmt: skip
        mz_den = alpha * mz_root + (1 - alpha)
        mz = (1 - aab * r2) / mz_den
        m_inorm_sq = (r2 + mz**2).reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        m = torch.cat((mxy, mz), dim=-1)  # (..., N, 3)

        # Jacobian of bearing w.r.t. m
        db_dm = torch.sqrt(m_inorm_sq) * (
            torch.eye(3, device=params.device, dtype=params.dtype)
            + ut.fast_small_matmul(-m_inorm_sq * m.unsqueeze(-1), m.unsqueeze(-2))
        )  # (..., N, 3, 3)

        # chain rule for Jacobian w.r.t. image coordinates
        db_dim_mxy = db_dm[..., :2] * if_.unsqueeze(-2)  # (..., N, 3, 2)
        dmz_dr2 = -aab / mz_den + mz * ab * (alpha - 0.5) / (mz_den * mz_root)  # (..., N, 1) # fmt: skip
        dr2_dim = (2 * mxy * if_).unsqueeze(-2)  # (..., N, 1, 2)
        db_dim = db_dim_mxy + db_dm[..., 2:] * dmz_dr2[..., None] * dr2_dim
        return db_dim

    def fit_dist_from_radii(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fit distortion parameters from sensor and ray radii.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays.

        Returns:
            (..., D_k) distortion parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found and/or the system is singular. None if all
                estimates are valid.
        """
        eps = torch.finfo(r.dtype).eps
        Zr = Z * r
        Zr_R = Zr - R
        Aa = (R * r) ** 2
        Ab = 2 * Zr * Zr_R
        A = torch.stack((Aa, Ab), dim=-1)  # (..., N, 2)
        b = Zr_R**2  # (..., N)
        sol, info = torch.linalg.solve_ex(
            A.transpose(-1, -2) @ A, (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
        )

        alpha = sol[..., 1]  # (...,)
        a2b = sol[..., 0]
        # undo reparametrization of β and clamp it for perspective cameras since β is
        # undefined for them (the projection model becomes independent of β).
        beta = torch.where(alpha < 0.025, 1, a2b / (alpha**2).clamp(eps))
        params = torch.stack((alpha, beta), dim=-1)
        # check bounds
        bmax = self.beta_max
        on_bounds = (alpha >= 0) & (alpha <= 1) & (a2b >= 0) & (a2b <= bmax)
        if on_bounds.all():  # bounds satisfied, i.e. already valid solution
            return params, info

        # activate inequalities (bounds) and re-solve
        # 1) α=1
        r2, R2, Z2 = r**2, R**2, Z**2
        valid = (r2 > eps) & (R2 > eps)
        a2b = ((1 / r2.clamp(eps) - Z2 / R2.clamp(eps)) * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(1)  # fmt: skip
        err1 = torch.where(
            (a2b >= 0) & (a2b <= bmax),
            ((Aa * a2b[..., None] + Ab - b) ** 2).sum(dim=-1),
            torch.inf,
        )
        # 2) α^2β=bmax
        den1 = 2 * Zr
        den2 = 2 * Z * Zr_R
        valid = (den1.abs() > eps) & (den2.abs() > eps)
        den1 = torch.where(valid, den1, 1)
        den2 = torch.where(valid, den2, 1)
        alpha = ((Zr_R / den1 - bmax * r * R2 / den2) * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(1)  # fmt: skip
        err2 = torch.where(
            (alpha >= 0) & (alpha <= 1),
            ((Aa * bmax + Ab * alpha[..., None] - b) ** 2).sum(dim=-1),
            torch.inf,
        )
        # 3) α=1, α^2β=bmax
        err3 = ((Aa * bmax + Ab - b) ** 2).sum(dim=-1)
        # 4) α=0, α^2β=0
        err4 = (b**2).sum(dim=-1)

        # select best solution
        one = torch.ones_like(err1)
        zero = torch.zeros_like(err1)
        alpha_cand = torch.stack((one, alpha, one, zero), dim=-1)
        a2b_cand = torch.stack((a2b, bmax * one, bmax * one, zero), dim=-1)  # (..., 4)
        best_idx = torch.stack((err1, err2, err3, err4), dim=-1).argmin(dim=-1, keepdim=True)  # fmt: skip
        best_alpha = torch.take_along_dim(alpha_cand, best_idx, dim=-1)  # (..., 1)
        best_a2b = torch.take_along_dim(a2b_cand, best_idx, dim=-1)  # (..., 1)

        # undo reparametrization of β and clamp it for perspective cameras since β is
        # undefined for them (the projection model becomes independent of β).
        beta = torch.where(best_alpha < 0.025, 1, best_a2b / (best_alpha**2).clamp(eps))
        params = torch.where(
            on_bounds[..., None], params, torch.cat((best_alpha, beta), dim=-1)
        )
        return params, info
        # return torch.stack((best_alpha, best_a2b), dim=-1), info

    def get_optim_update(self, params: Tensor, delta: Tensor) -> Tensor:
        """Undo reparameterization of α and β within delta and apply the update.

        During optimization the following should be satisfied:
            * α ∈ [0, 1]
            * β ∈ (0, ∞], actually we use [β_min, β_max] for more stable optimization.
        To ensure this, we optimize for
            * α_hat in α = sigmoid(α_hat) and
            * β_hat in β = sigmoid(β_hat) * β_ptp + β_min.
        Because of this, we need to undo this reparameterization before applying the update.

        Args:
            params: (..., D) intrinsic parameters.
            delta: (..., D) update.

        Returns:
            (..., D) updated intrinsic parameters.
        """
        if not self.safe_optim:
            return params + delta

        b_hat = logit((params[..., -1:] - self.beta_min) / self.beta_ptp)
        return torch.cat(
            (
                params[..., :-2] + delta[..., :-2],
                torch.sigmoid(logit(params[..., -2]) + delta[..., -2]).unsqueeze(-1),
                torch.sigmoid(b_hat + delta[..., -1:]) * self.beta_ptp + self.beta_min,
            ),
            dim=-1,
        )

    def get_optim_jac(self, jac: Tensor, params: Tensor) -> Tensor:
        """If needed, apply chain rule for reparameterized parameters during optimization.

        During optimization the following should be satisfied:
            * α ∈ [0, 1]
            * β ∈ (0, ∞], actually we use [β_min, β_max] for more stable optimization.
        To ensure this, we optimize for
            * α_hat in α = sigmoid(α_hat) and
            * β_hat in β = sigmoid(β_hat) * β_ptp + β_min.
        Because of this, we apply the chain rule to get the Jacobian w.r.t. the
        *reparameterized* parameters.

        Args:
            jac: (..., N, D) Jacobian.
            params: (..., D) intrinsic parameters.

        Returns:
            (..., N, D) updated Jacobian.
        """
        if not self.safe_optim:
            return jac

        # NOTE: dσ(x)/dx = σ(x)*(1 - σ(x))
        a = params[..., -2, None, None]  # alpha
        b = params[..., -1, None, None]  # beta
        db_dbhat = (b - self.beta_min) * (self.beta_max - b) / self.beta_ptp
        return torch.cat(
            (jac[..., :-2], a * (1 - a) * jac[..., -2, None], db_dbhat * jac[..., -1:]),
            dim=-1,
        )

    def fit_from_radii_unsafe(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fit distortion parameters from sensor and ray radii.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays.

        Returns:
            (..., D_k) distortion parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found and/or the system is singular. None if all
                estimates are valid.
        """
        A = torch.stack(((R * r) ** 2, 2 * Z * r * (Z * r - R)), dim=-1)  # (..., N, 2)
        b = (R - Z * r) ** 2
        sol, info = torch.linalg.solve_ex(
            A.transpose(-1, -2) @ A, (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
        )
        # print(torch.linalg.svdvals(A.transpose(-1, -2) @ A))
        alpha = sol[..., -1:]
        beta = sol[..., :1] / (alpha**2).clamp(torch.finfo(alpha.dtype).eps)
        dist_params = torch.cat((alpha, beta), dim=-1)
        return dist_params, info
        # return sol.flip(-1), info

    def fit_from_radii_scipy(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Fit distortion parameters from sensor and ray radii.

        Args:
            r: (..., N) radii (from center of distortion) in the sensor plane.
            R: (..., N) radii of the rays: R = sqrt(X^2 + Y^2)
            Z: (..., N) Z coordinates of the rays.

        Returns:
            (..., D_k) distortion parameters.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found and/or the system is singular. None if all
                estimates are valid.
        """
        from scipy.optimize import lsq_linear

        A = torch.stack(((R * r) ** 2, 2 * Z * r * (Z * r - R)), dim=-1)  # (..., N, 2)
        b = (R - Z * r) ** 2
        res = lsq_linear(
            A.cpu().numpy(),
            b.cpu().numpy(),
            bounds=((0, 0), (1e3, 1)),
            method="bvls",
        )
        sol, info = torch.from_numpy(res.x).flip(-1), res.success
        sol[1] = 1 if sol[0] < 0.025 else sol[1] / (sol[0] ** 2)
        return sol, torch.tensor(info == 0)
