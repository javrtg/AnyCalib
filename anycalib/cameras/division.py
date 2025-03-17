import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera


def radii_via_companion(
    R: Tensor, Z: Tensor, k: Tensor, cplex_tol: float = 1e-4
) -> tuple[Tensor, Tensor]:
    """Get radii in retinal plane using the companion matrix of the polynomial.

    TODO: Implement backward pass via implicit function theorem.

    Args:
        R: (..., N) radii of the 3D points: sqrt(X^2 + Y^2).
        Z: (..., N) z-coordinates of the 3D points.
        k: (..., num_k) distortion coefficients.

    Returns:
        (..., N) radii in the retinal plane.
        (..., N) boolean tensor indicating valid radii.
    """
    eps = torch.finfo(R.dtype).eps
    # points close to the optical axis lead to numerical issues. Mask them and project
    # them later to the principal point.
    close_to_ppoint = R < eps
    iR = torch.where(close_to_ppoint, 0.1, R).reciprocal()
    # form companion matrix (..., N, 2*num_k, 2*num_k)
    k_lead = k[..., -1:]  # (..., 1)
    k_lead = torch.where(k_lead == 0, eps, k_lead)
    ik_lead = torch.where(k_lead.abs() < eps, k_lead.sign() * eps, k_lead).reciprocal()
    companion = R.new_ones((*R.shape, 2 * k.shape[-1] - 1)).diag_embed(-1)
    companion[..., 0, -1] = -ik_lead
    companion[..., 0, -2] = Z * iR * ik_lead
    companion[..., 0, 1:-2:2] = (-k[..., :-1].flip(-1) * ik_lead).unsqueeze(-2)
    # get smallest positive real root (if exists) (..., N, 2*num_k)
    roots = torch.linalg.eigvals(companion).masked_fill(close_to_ppoint[..., None], 0)
    valid = (roots.real >= 0) & (roots.imag.abs() < cplex_tol * roots.real.abs())
    sol = torch.where(valid, roots.real, torch.inf).amin(dim=-1)
    valid = sol < torch.inf
    sol = torch.where(valid, sol, 0.1)  # set bogus value
    return sol, valid


def unproject_z(xy: Tensor, k: Tensor) -> Tensor:
    """Back-projection/unprojection function for the Division model.

    Args:
        xy: (..., N, 2) image coordinates in the retinal/sensor plane.
        k: (..., num_k) distortion coefficients.

    Returns:
        (..., N) backprojected z-coordinate.
    """
    r2 = ri = (xy**2).sum(-1)  # (..., N)
    z = 1 + k[..., :1] * r2
    for i in range(1, k.shape[-1]):
        ri = ri * r2
        z = z + k[..., i, None] * ri
    return z


class Division(BaseCamera):
    """Implementation of the Division Camera Model [1].

    This class implements the slight variation [2, 3] of the original model [1] which
    defines the back-projection (or unprojection) function as:
        x = (u - cx)/fx
        y = (v - cy)/fy
        z =  1 + k1*r^2 + k2*r^4 + ...
    where r is the radius of the retinal point, defined as: r = sqrt(x^2 + y^2). The
    unprojected point is subsequently normalized to have unit norm. This implementation
    supports a variable number (up to 4) of distortion coefficients, controlled by the
    variable/attribute num_k.
    The (ordered) intrinsic parameters are fx, fy, cx, cy, k1, k2, ...
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    [1] Simultaneous Linear Estimation of Multiple View Geometry and Lens Distortion.
        A.W. Fitzgibbon, CVPR 2001.
    [2] Revisiting Radial Distortion Absolute Pose. V. Larsson et al., ICCV 2019.
    [3] Babelcalib: A Universal Approach to Calibrating Central Cameras.
        Y. Lochman et al., ICCV 2021.
    """

    NAME = "division"
    # number of focal lengths
    NUM_F = 2
    PARAMS_IDX = {
        "fx": 0,
        "fy": 1,
        "cx": 2,
        "cy": 3,
        "k1": 4,
        "k2": 5,
        "k3": 6,
        "k4": 7,
    }

    def __init__(self, num_k: int = 1, complex_tol: float = 1e-4):
        if num_k <= 0 or not isinstance(num_k, int):
            raise ValueError(f"`num_k` must be a positive integer but got: {num_k}.")
        self.num_k = num_k
        self.cplex_tol = complex_tol

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
        k = params[..., self.NUM_F + 2 :]
        return f, c, k

    @classmethod
    def create_from_params(cls, params: Tensor) -> BaseCamera:
        return cls(num_k=params.shape[-1] - cls.NUM_F - 2)

    @classmethod
    def create_from_id(cls, id_: str) -> BaseCamera:
        num_k_str = id_.partition(":")[2]  # expected format: name:num_k
        assert 0 <= len(num_k_str) <= 1, f"Invalid id format: {id_}"
        if len(num_k_str) == 0:
            return cls()  # default
        return cls(num_k=int(num_k_str))

    @property
    def id(self) -> str:
        return f"{self.NAME}:{self.num_k}"

    def validate_params(self, params: Tensor):
        if params.shape[-1] != self.NUM_F + 2 + self.num_k:
            num_f, num_k = self.NUM_F, self.num_k
            total = num_f + 2 + num_k
            params_str = ("fx, fy" if self.NUM_F == 2 else "f") + ", cx, cy, k1, ..."
            raise ValueError(
                f"Expected (..., {total}) parameters as input, representing (...) "
                f"cameras and {total} intrinsic params per cameras: {params_str}, i.e. "
                f"with {num_f} focal(s) and {num_k} distortion coeffs., but got: "
                f"{params.shape[-1]=}."
            )

    def get_min_sample_size(self, with_cxcy: bool) -> int:
        """Minimal number of 2D-3D samples needed to fit the intrinsics."""
        # total_params = (self.NUM_F if with_cxcy else self.NUM_F + 2) + 1
        # return ceil(0.5 * total_params)
        return 1 + self.num_k if self.NUM_F == 1 else max(3, 1 + self.num_k)

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
        f, c, k = self.parse_params(params)

        if self.num_k == 1:  # closed-form solution
            z = points_3d[..., 2]  # (..., N)
            disc = z**2 - 4 * k * (points_3d[..., :2] ** 2).sum(-1)
            den = z + torch.sqrt(disc.clamp(0))  # use + to get smallest root
            valid = (disc > -eps) & (den > eps)
            im_coords = 2 * points_3d[..., :2] / den.clamp(eps).unsqueeze(-1)
            im_coords = f.unsqueeze(-2) * im_coords + c.unsqueeze(-2)
            return im_coords, valid

        R = torch.linalg.norm(points_3d[..., :2], dim=-1)  # (..., N)
        r, valid = radii_via_companion(R, points_3d[..., 2], k, self.cplex_tol)
        im_coords = (
            r.unsqueeze(-1)
            * points_3d[..., :2]
            / torch.where(R < eps, 1, R).unsqueeze(-1)
        )
        im_coords = f.unsqueeze(-2) * im_coords + c.unsqueeze(-2)
        return im_coords, valid

    def unproject(self, params: Tensor, points_2d: Tensor) -> tuple[Tensor, None]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3) unit bearing vectors in the camera frame.
            None, to indicate that all unprojections are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
        """
        self.validate_params(params)
        f, c, k = self.parse_params(params)
        xy = (points_2d - c[..., None, :]) / f[..., None, :]  # (..., N, 2)
        z = unproject_z(xy, k)  # (..., N)
        unit_bearings = F.normalize(torch.cat((xy, z.unsqueeze(-1)), dim=-1), dim=-1)
        return unit_bearings, None

    def fit_minimal(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> Tensor:
        """Fit instrinsics with a minimal set of 2D-bearing correspondences.

        Args:
            im_coords: (..., min_n, 2) image coordinates.
            bearings: (..., min_n, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            (..., D) fitted intrinsic parameters: fx, fy, cx, cy, k1, ...
        """
        is_simple = self.NUM_F == 1
        num_k = self.num_k
        num_kp1 = num_k + 1
        if is_simple:
            cxcy_, info0 = ut.cxcy_from_rays(im_coords, bearings)
            pix_ar = 1
        else:
            cxcy_, pix_ar, info0 = ut.cxcy_and_pix_ar_from_rays(im_coords, bearings)
            pix_ar = pix_ar.unsqueeze(-1)
        cxcy_ = cxcy_ if cxcy is None else cxcy
        # form batched system
        Ra = torch.linalg.norm(
            torch.stack(
                (bearings[..., :num_kp1, 0], pix_ar * bearings[..., :num_kp1, 1]),
                dim=-1,
            ),
            dim=-1,
            keepdim=True,
        )  # (..., N, 1)
        xc = im_coords[..., :num_kp1, :] - cxcy_.unsqueeze(-2)
        rc = torch.linalg.norm(xc, dim=-1)  # (..., N)
        rca2 = (xc[..., 0] ** 2 + xc[..., 1] ** 2 / pix_ar**2).unsqueeze(-1)
        A = torch.cat(  # (..., N, 1 + num_k)
            (
                Ra,
                Ra * rca2,
                Ra
                * rca2.pow(
                    torch.arange(2, num_k + 1, device=Ra.device, dtype=Ra.dtype)
                ),
            ),
            dim=-1,
        )
        b = bearings[..., :num_kp1, 2] * rc  # (..., N)
        # solve
        sol, info = torch.linalg.solve_ex(A, b)  # (..., 2)
        sol[(info != 0) | (info0 != 0)] = 1.1  # set bogus value for invalid solutions
        # undo the reparametrization
        fx = sol[..., :1]
        coeffs = torch.cat(
            (
                sol[..., 1:2] * fx,
                sol[..., 2:]
                * fx.pow(
                    torch.arange(3, 2 * num_k + 1, 2, device=fx.device, dtype=fx.dtype)
                ),
            ),
            dim=-1,
        )
        c_sh = (*sol.shape[:-1], 2)
        if is_simple:
            return torch.cat((fx, cxcy_.expand(*c_sh), coeffs), dim=-1)
        params = torch.cat((fx, pix_ar * fx, cxcy_.expand(*c_sh), coeffs), dim=-1)
        return params

    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        params0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fit instrinsics with a non-minimal set of 2D-bearing correspondences

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
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
        num_k = self.num_k
        # recover pixel aspect ratio and principal point
        if is_simple:
            cxcy_, info1 = ut.cxcy_from_rays(im_coords, bearings)
            pix_ar = 1
        else:
            cxcy_, pix_ar, info1 = ut.cxcy_and_pix_ar_from_rays(im_coords, bearings)
            pix_ar = pix_ar.unsqueeze(-1)
        cxcy_ = cxcy_ if cxcy is None else cxcy  # dirty...

        # form batched system
        Ra = torch.linalg.norm(
            torch.stack((bearings[..., 0], pix_ar * bearings[..., 1]), dim=-1),
            dim=-1,
            keepdim=True,
        )  # (..., N, 1)
        xc = im_coords - cxcy_.unsqueeze(-2)
        rc = torch.linalg.norm(xc, dim=-1)  # (..., N)
        rca2 = (xc[..., 0] ** 2 + xc[..., 1] ** 2 / pix_ar**2).unsqueeze(-1)
        A = torch.cat(  # (..., N, 1 + num_k)
            (
                Ra,
                Ra * rca2,
                Ra
                * rca2.pow(
                    torch.arange(2, num_k + 1, device=Ra.device, dtype=Ra.dtype)
                ),
            ),
            dim=-1,
        )
        b = bearings[..., 2] * rc  # (..., N)
        # solve
        sol, info2 = torch.linalg.solve_ex(
            A.transpose(-1, -2) @ A, (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
        )
        info = torch.where((info1 == 0) & (info2 == 0), 0, -1)
        # undo the reparametrization
        fx = sol[..., :1]
        coeffs = torch.cat(
            (
                sol[..., 1:2] * fx,
                sol[..., 2:]
                * fx.pow(
                    torch.arange(3, 2 * num_k + 1, 2, device=fx.device, dtype=fx.dtype)
                ),
            ),
            dim=-1,
        )
        c_sh = (*sol.shape[:-1], 2)
        if is_simple:
            return torch.cat((fx, cxcy_.expand(*c_sh), coeffs), dim=-1), info
        params = torch.cat((fx, pix_ar * fx, cxcy_.expand(*c_sh), coeffs), dim=-1)
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
        num_k = self.num_k
        is_simple = num_f == 1
        nif = -params[..., None, :num_f].reciprocal()  # (..., 1, {1, 2})
        k = params[..., None, num_f + 2 :]  # (..., 1, num_k)

        # intermediate variables
        mxy = (params[..., None, num_f : num_f + 2] - im_coords) * nif
        mxy_nif = mxy * nif  # (..., N, 2)
        r2 = r2i = (mxy**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        mz = 1 + k[..., :1] * r2
        dmz_dr2 = k[..., :1].expand_as(r2)  # Jacobian of mz w.r.t. r^2 (..., N, 1)
        for i in range(1, num_k):
            dmz_dr2 = dmz_dr2 + (i + 1) * k[..., i, None] * r2i
            r2i = r2i * r2
            mz = mz + k[..., i, None] * r2i
        m_inorm_sq = (r2 + mz**2).reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        m = torch.cat((mxy, mz), dim=-1)  # (..., N, 3)

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
        dr2_dfc = 2 * torch.cat(
            (
                (mxy * mxy_nif).sum(-1, keepdim=True) if is_simple else mxy * mxy_nif,
                mxy_nif,
            ),
            dim=-1,
        ).unsqueeze(-2)  # (..., N, 1, {3, 4}) # fmt: skip
        db_dfc = db_dfc_mxy + db_dm[..., 2:] * dmz_dr2[..., None] * dr2_dfc
        # chain rule for Jacobian w.r.t. distortion coefficients
        dmz_dk = torch.cat(
            (r2, r2.pow(torch.arange(2, num_k + 1, device=r2.device))), dim=-1
        )

        db_dparams = torch.cat(
            (db_dfc, db_dm[..., 2:] * dmz_dk.unsqueeze(-2)),
            dim=-1,
        )
        return db_dparams

    def jac_bearings_wrt_imcoords(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings wrt the image coordinates.

        NOTE: This method assumes that all input bearings have unit norm.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, 2) Jacobians.
        """
        # intermediate variables
        self.validate_params(params)
        num_f = self.NUM_F
        num_k = self.num_k
        if_ = params[..., None, :num_f].reciprocal()  # (..., 1, {1, 2})
        k = params[..., None, num_f + 2 :]  # (..., 1, num_k)

        # intermediate variables
        mxy = (im_coords - params[..., None, num_f : num_f + 2]) * if_
        r2 = r2i = (mxy**2).sum(dim=-1, keepdim=True)  # (..., N, 1)
        mz = 1 + k[..., :1] * r2
        dmz_dr2 = k[..., :1].expand_as(r2)  # Jacobian of mz w.r.t. r^2 (..., N, 1)
        for i in range(1, num_k):
            dmz_dr2 = dmz_dr2 + (i + 1) * k[..., i, None] * r2i
            r2i = r2i * r2
            mz = mz + k[..., i, None] * r2i
        m_inorm_sq = (r2 + mz**2).reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        m = torch.cat((mxy, mz), dim=-1)  # (..., N, 3)

        # Jacobian of bearing w.r.t. m
        db_dm = torch.sqrt(m_inorm_sq) * (
            torch.eye(3, device=params.device, dtype=params.dtype)
            + ut.fast_small_matmul(-m_inorm_sq * m.unsqueeze(-1), m.unsqueeze(-2))
        )  # (..., N, 3, 3)

        # chain rule for Jacobian w.r.t. image coordinates
        db_dim_mxy = db_dm[..., :2] * if_.unsqueeze(-2)  # (..., N, 3, 2)
        dr2_dim = 2 * (mxy * if_).unsqueeze(-2)  # (..., N, 1, 2)
        db_dim = db_dim_mxy + db_dm[..., 2:] * dmz_dr2[..., None] * dr2_dim
        return db_dim

    def fit_dist_from_radii(
        self, r: Tensor, R: Tensor, Z: Tensor
    ) -> tuple[Tensor, Tensor | None]:
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
        if self.num_k == 1:  # use closed-form solution
            num = Z * r - R
            den = R * r**2
            # masked mean
            valid = den >= torch.finfo(den.dtype).eps
            den = torch.where(valid, den, 1)
            k = (num / den * valid).sum(dim=-1) / valid.sum(dim=-1).clamp(1)
            # # check monotonicity
            # r_sort, _ = r.sort(dim=-1)
            # z = 1 + k[..., None] * r_sort**2
            # z = z / torch.sqrt(r_sort**2 + z**2)
            # print(f"{k}\n{Z}\n{z}")
            # assert torch.allclose(
            #     z.sort(dim=-1, descending=True).values, z
            # ), f"{k}\n{r}\n{R}"
            return k.unsqueeze(-1), torch.where(valid.sum(dim=-1) > 0, 0, -1)

        # form system
        r2 = (r**2).unsqueeze(-1)
        A = R.unsqueeze(-1) * torch.cat(  # (..., N, num_k)
            (
                r2,
                r2.pow(torch.arange(2, self.num_k + 1, device=r.device, dtype=r.dtype)),
            ),
            dim=-1,
        )
        b = Z * r - R  # (..., N)
        # solve
        sol, info = torch.linalg.solve_ex(
            A.transpose(-1, -2) @ A, (A.transpose(-1, -2) @ b.unsqueeze(-1)).squeeze(-1)
        )
        return sol, info
