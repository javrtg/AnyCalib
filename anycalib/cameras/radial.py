from math import ceil, sqrt

import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera
from anycalib.cameras.pinhole import check_within_fov
from anycalib.manifolds import Unit3


def cbrt(x: Tensor) -> Tensor:
    """Cube root of a tensor."""
    return torch.sign(x) * torch.pow(x.abs().clamp(torch.finfo(x.dtype).eps), 1 / 3)


class NewtonUndistortRadii(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, k: Tensor, radii_d: Tensor, n_iters: int = 25, undist_tol: float = 1e-5
    ) -> tuple[Tensor, Tensor]:
        """Newton method for undistorting radii.

        Solve for the undistorted radius, r_u, in:
            f := -r_d + r_u + k1*r_u^3 + k2*r_u^5 + ... = 0
        given the distorted radius, r_d.

        Args:
            k: (..., num_k) radial distortion coefficients.
            radii_d: (..., N) distorted radii.
            n_iters: number of Newton iterations.
            undist_tol: threshold to check convergence of the Newton algorithm.

        Returns:
            radii_u: (..., N) undistorted radii.
            converged: (..., N) boolean tensor indicating convergence.
        """
        num_k = k.shape[-1]
        # initialize to solution without distortion
        radii_u = radii_d
        for _ in range(n_iters):
            # initialize polynomial and its derivative
            radii_ui = radii_ui2 = radii_u * radii_u
            dist = 1 + k[..., :1] * radii_ui2
            f_grad = 1 + 3 * k[..., :1] * radii_ui2
            # remaining terms/coefficients
            for j in range(1, num_k):
                radii_ui = radii_ui2 * radii_ui  # radii_ui = r_u^(2 + 2i)
                dist = dist + k[..., j, None] * radii_ui
                f_grad = f_grad + (3 + 2 * j) * k[..., j, None] * radii_ui
            f = radii_u * dist - radii_d
            radii_u = radii_u - (f / f_grad)
        # check convergence
        converged = f.abs() <= undist_tol
        # tensors needed for backward
        ctx.save_for_backward(k, radii_u)
        ctx.mark_non_differentiable(converged)
        return radii_u, converged

    @staticmethod
    def backward(
        ctx, dloss_dradiiu: Tensor, dloss_dconverged: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        """Backward pass for the Newton method using the implicit function theorem.

        Args:
            dloss_dradiiu: (..., N) gradient w.r.t. the undistorted radii.
            dloss_dconverged: (..., N) gradient w.r.t. the convergence flag. Ignored
                as it is not differentiable.

        Returns:
            dloss_dk: (..., num_k) gradient w.r.t. the radial distortion coefficients.
            dloss_radiid: (..., N) gradient w.r.t. the distorted radii.
        """
        k, radii_u = ctx.saved_tensors
        dloss_dk = dloss_radiid = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            num_k = k.shape[-1]
            powers = torch.arange(3, 3 + 2 * num_k, 2, device=radii_u.device)  # 3, 5...
            # f_grad := 1 + 3*k1*r_u^2 + 5*k2*r_u^4 + ...
            powers_g = torch.arange(2, 2 + 2 * num_k, 2, device=radii_u.device)
            f_grad = 1 + (
                powers * k[..., None, :] * torch.pow(radii_u[..., None], powers_g)
            ).sum(-1)  # (..., N) # fmt: skip

        if ctx.needs_input_grad[0]:
            monomials = torch.pow(radii_u[..., None], powers)  # r_u^3, r_u^5, ...
            dradiiu_dk = -monomials / f_grad[..., None]  # (..., N, num_k)
            dloss_dk = (dloss_dradiiu[..., None] * dradiiu_dk).sum(-2)  # (..., num_k)

        if ctx.needs_input_grad[1]:
            dloss_radiid = dloss_dradiiu / f_grad

        return dloss_dk, dloss_radiid, None, None


def newton_undistort_radii(
    k: Tensor, radii_d: Tensor, n_iters: int = 25, undist_tol: float = 1e-5
) -> tuple[Tensor, Tensor]:
    """Newton method for undistorting radii.

    Solve for the undistorted radius, r_u, in:
        f := -r_d + k1*r_u^3 + k2*r_u^5 + ... = 0
    given the distorted radius, r_d.

    Args:
        k: (..., num_k) radial distortion coefficients.
        radii_d: (..., N) distorted radii.
        n_iters: number of Newton iterations.
        undist_tol: absolute tolerance for simple convergence check.

    Returns:
        radii_u: (..., N) undistorted radii.
        converged: (..., N) boolean tensor indicating convergence.
    """
    return NewtonUndistortRadii.apply(k, radii_d, n_iters, undist_tol)  # type: ignore


def compute_distortion(radii_u2: Tensor, k: Tensor) -> Tensor:
    """Compute the radial distortion factor: 1 + k1*r^2 + k2*r^4 + ...

    Args:
        radii_u2: (..., N) squared undistorted radii.
        k: (..., num_k) radial distortion coefficients.

    Returns:
        (..., N) radial distortion factor.
    """
    radii_ui = radii_u2  # (..., N)
    dist = 1 + k[..., :1] * radii_u2
    for i in range(1, k.shape[-1]):
        radii_ui = radii_u2 * radii_ui  # radii_ui = r_u^(2 + 2i)
        dist = dist + k[..., i, None] * radii_ui
    return dist


def propagate_tangent_covs(
    bearings: Tensor, proj: Tensor, covs: Tensor, k: Tensor
) -> Tensor:
    """Propagate covariances expressed in the tangent space of the bearings to the
    space where errors are linear w.r.t. the intrinsic parameters.

    The propagation is done to the error space:
        e_x = a_x*u - b_x - (X/Z)*(1 + k1*r^2 + k2*r^4 + ...)
        e_y = a_y*v - b_y - (Y/Z)*(1 + k1*r^2 + k2*r^4 + ...)
    where a = 1/f, and b = c/f, (u, v) are the pixel image coordinates, and (X, Y, Z)
    are the 3D point/bearing coordinates.

    Args:
        bearings: (..., N, 3) unit bearing vectors in the camera frame.
        proj: (..., N, 2) perspective projection of the bearings (to save computation).
        covs: (..., N, 2) diagonal elements of the covariances expressed in the tangent
            space of the input bearings.
        k: (..., num_k) approximate distortion coeffs to propagate the covariances.

    Returns:
        (..., N, 2, 2) error covariances.
    """
    dist = compute_distortion((proj * proj).sum(-1), k)  # (..., N)
    # Jacobian of the error w.r.t. point in unit sphere (..., 2, 3)
    derr_dp = covs.new_zeros((*covs.shape, 3))
    derr_dxy = dist / bearings[..., 2].clamp(torch.finfo(bearings.dtype).eps)  # (...,N)
    derr_dp[..., 0, 0] = derr_dp[..., 1, 1] = derr_dxy
    derr_dp[..., 2] = -proj * derr_dxy[..., None]
    # Jacobian of the error w.r.t. the coordinates in the tangent plane
    derr_dbasis = derr_dp @ Unit3.get_tangent_basis(bearings)
    error_covs = ut.fast_small_matmul(
        derr_dbasis * covs[..., None, :], derr_dbasis.transpose(-1, -2)
    )
    return error_covs


class Radial(BaseCamera):
    """Pinhole camera model with polynomial radial distortion.

    Projection:
        x = fx * (X / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cx
        y = fy * (Y / Z) * (1 + k1 * r^2 + k2 * r^4 + ...) + cy
    The (ordered) intrinsic parameters are fx, fy, cx, cy, k1, k2, ...,
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    Args:
        max_fov: Threshold in degrees for masking out bearings/rays whose incidence
            angles correspond to fovs above this admissible field of view.
        num_k: number of radial distortion coefficients. Default is 2.
        undist_iters: number of Newton iterations for undistorting radii.
        undist_tol: threshold for checking convergence of the Newton algorithm.
    """

    NAME = "radial"
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

    def __init__(
        self,
        max_fov: float = 170,
        num_k: int = 2,
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

    def parse_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Parse parameters into focal lengths, principal points, and distortion.

        This method is useful for creating a unified API between Radial and SimpleRadial
        camera models.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., 2) focal lengths, (..., 2) principal points, (..., num_k) distortion.
        """
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
        total_params = (self.NUM_F if with_cxcy else self.NUM_F + 2) + self.num_k
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
        f, c, k = self.parse_params(params)
        # perspective projection
        proj = points_3d[..., :2] / points_3d[..., 2:].clamp(torch.finfo(points_3d.dtype).eps)  # fmt: skip
        dist = compute_distortion((proj * proj).sum(-1), k)  # (..., N)
        proj_d = proj * dist[..., None]
        # image coordinates (in pixels)
        im_coords = f[..., None, :] * proj_d + c[..., None, :]
        return im_coords, points_3d[..., 2] > 0

    def unproject(
        self, params: Tensor, points_2d: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        """Unproject image coordinates to unit bearing vectors in the camera frame.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.
            points_2d: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3) unit bearing vectors in the camera frame.
            (..., N) boolean tensor indicating convergence.
        """
        self.validate_params(params)
        f, c, k = self.parse_params(params)
        # normalize image coordinates
        unproj_d = (points_2d - c[..., None, :]) / f[..., None, :]
        # solve for the undistorted radii
        rd = torch.linalg.norm(unproj_d, dim=-1)
        if k.shape[-1] == 1:  # closed-form Cardano's method
            # adapted from https://github.com/stanfordnmbl/opencap-core/blob/main/utilsCameraPy3.py
            eps = torch.finfo(k.dtype).eps
            k = torch.where(k >= 0, k.clamp(eps), k.clamp(None, -eps))  # abs(k) > 0
            Q = k.reciprocal() / 3
            R = 0.5 * rd / k
            disc = Q**3 + R**2
            disc_sqrt = torch.sqrt(disc.abs())
            # positive-discriminant solution
            sol_p = cbrt(R + disc_sqrt) + cbrt(R - disc_sqrt)
            # negative-discriminant solution
            S = cbrt(torch.sqrt(R**2 + disc.abs()))
            T = torch.arctan2(disc_sqrt, R) / 3
            sol_n = S * (-torch.cos(T) + sqrt(3) * torch.sin(T))
            # select solution
            ru = torch.where(disc >= 0, sol_p, sol_n)
            valid = None
        else:
            ru, valid = newton_undistort_radii(
                k, rd, self.undist_iters, self.undist_tol
            )
        # undistort image coordinates: p_u = p_d * r_u / r_d = p_d / dist
        eps = torch.finfo(rd.dtype).eps
        unproj = unproj_d * (ru / rd.clamp(eps))[..., None]
        # normalize to unit bearing vectors
        bearings = torch.cat((unproj, unproj.new_ones((*unproj.shape[:-1], 1))), dim=-1)
        return F.normalize(bearings, dim=-1), valid

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
        assert (
            self.get_min_sample_size(cxcy is not None)
            == im_coords.shape[-2]
            == bearings.shape[-2]
        ), "Input sizes do not match the minimal sample size"

        As, bs = self._form_batched_system(im_coords, bearings, cxcy)
        A = As.flatten(-3, -2)[..., : As.shape[-1], :]  # (..., D, D)
        b = bs.flatten(-2, -1)[..., : As.shape[-1]]  # (..., D)

        sol, info = torch.linalg.solve_ex(A, b)  # (..., 4 + num_k)
        sol = sol.squeeze(-1)
        sol[info != 0] = 1.1  # set bogus value for invalid solutions
        # sol = torch.linalg.lstsq(A, b).solution.squeeze(-1)

        nf = self.NUM_F  # number of focal lengths
        f = sol[..., :nf].reciprocal()
        if cxcy is None:
            return torch.cat((f, sol[..., nf : nf + 2] * f, sol[..., nf + 2 :]), dim=-1)
        return torch.cat((f, cxcy.expand(*f.shape[:-1], 2), sol[..., nf:]), dim=-1)

    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        params0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Fit reparametrized instrinsics with a non-minimal set of 2D-bearing
        correspondences and bearing covariances.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances
                expressed in the tangent space of the input bearings.
            params0: (..., D) approximate estimation of the intrinsic parameters to
                propagate the covariances from the tangent space to the error space.

        Returns:
            (..., D) fitted intrinsic parameters: fx, fy, cx, cy, k1, ...
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found (<0) or the system is singular (>0).
        """
        valid = check_within_fov(bearings, self.max_fov)

        # normalize for better condition number
        nf = self.NUM_F  # number of focal lengths
        fac = im_coords.amax(-2 if nf == 2 else (-2, -1), keepdim=True)  # (..,1, {2,1})

        As, bs = self._form_batched_system(
            im_coords / fac, bearings, None if cxcy is None else cxcy / fac.squeeze(-2)
        )

        if covs is not None:
            assert (
                params0 is not None
            ), "Approximate parameters are required when covs are provided."
            # NOTE: bs = proj
            k0 = params0[..., self.NUM_F + 2 :]
            eps = torch.finfo(covs.dtype).eps
            e_covs = propagate_tangent_covs(bearings, bs, covs.clamp(eps), k0)
            Ws, info = torch.linalg.inv_ex(e_covs)
            valid = valid & (info == 0)
        else:
            Ws = None

        sol, info = ut.solve_2dweighted_lstsq(As, bs, Ws, valid)
        f = fac.squeeze(-2) * sol[..., :nf].reciprocal()
        if cxcy is None:
            c = sol[..., nf : nf + 2] * f
            return torch.cat((f, c, sol[..., nf + 2 :]), dim=-1), info
        return torch.cat((f, cxcy, sol[..., nf:]), dim=-1), info

    def _form_batched_system(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Form the 2D equations for each 2D-3D correspondence.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            As: (..., N, 4 + num_k) design matrices (without stacking).
            bs: (..., N, 2) observations.
        """
        eps = torch.finfo(bearings.dtype).eps
        num_k = self.num_k
        # perspective projection
        proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)  # (..., N, 2)
        # form linear system
        if cxcy is None:
            As = proj.new_zeros((*proj.shape, 4 + num_k))
            As[..., 0, 0] = im_coords[..., 0]
            As[..., 1, 1] = im_coords[..., 1]
            As[..., 0, 2] = As[..., 1, 3] = -1
            offset = 4
        else:
            As = proj.new_zeros((*proj.shape, 2 + num_k))
            As[..., 0, 0] = im_coords[..., 0] - cxcy[..., 0, None]
            As[..., 1, 1] = im_coords[..., 1] - cxcy[..., 1, None]
            offset = 2
        # distortion terms
        radii_u2 = (proj * proj).sum(-1, keepdim=True)  # (..., N, 1)
        proj_radii = -proj * radii_u2  # (..., N, 2)
        As[..., offset] = proj_radii
        for i in range(1, num_k):
            proj_radii = proj_radii * radii_u2
            As[..., offset + i] = proj_radii
        return As, proj

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
        f, c, k = self.parse_params(params)
        num_k = self.num_k
        eps = torch.finfo(bearings.dtype).eps
        iz = 1 / bearings[..., 2:].clamp(eps)
        proj_u = bearings[..., :2] * iz  # (..., N, 2)

        # 1) projection Jacobian dprojd/dbearing as dprojd/dproju * dprojud/dbearing
        # 2) dprojd/dk (retinal/sensor plane projection w.r.t. distort coeffs k)
        powers = proj_u.new_empty((*iz.shape, num_k))  # r^2, r^4 .. (..., N, 1, num_k)
        ru_2 = ru_i = powers[..., 0, 0] = (proj_u * proj_u).sum(-1)  # (..., N)
        k = k[..., None, :]  # (..., 1, num_k)
        agg1 = 2 * k[..., 0]
        agg2 = k[..., 0] * ru_2
        for i in range(1, num_k):
            agg1 = agg1 + (2 + 2 * i) * k[..., i] * ru_i
            ru_i = powers[..., 0, i] = ru_i * ru_2
            agg2 = agg2 + k[..., i] * ru_i
        dprojd_dproju = (
            (1 + agg2)[..., None].expand(*agg2.shape, 2).diag_embed()
            + agg1[..., None, None] * proj_u[..., None] * proj_u[..., None, :]
        )  # fmt:skip # (..., N, 2, 2)
        dproju_dbearing = torch.cat(
            (iz.expand_as(proj_u).diag_embed(), (-iz * proj_u)[..., None]),
            dim=-1,
        )  # (..., N, 2, 3)
        dprojd_dk = proj_u[..., None] * powers  # (..., N, 2, num_k)

        # apply inverse function theorem to obtain dbearing/dprojd
        dbearing_dprojd = pinv_2d_jac_wrt_bearing(
            ut.fast_small_matmul(dprojd_dproju, dproju_dbearing), bearings
        )  # (..., N, 3, 2)

        in_f = -1 / f.clamp(torch.finfo(f.dtype).eps)[..., None, :]
        proj_d = (c[..., None, :] - im_coords) * in_f
        dprojd_df = (
            (proj_d * in_f)[..., None]
            if self.NUM_F == 1
            else (proj_d * in_f).diag_embed()
        )  # (..., N, 2, {1, 2})
        dprojd_dc = in_f.expand_as(proj_d).diag_embed()  # (..., N, 2, 2)

        # triple product rule: db/dk = - db/dprojd * dprojd/dk
        dprojd_dparams = torch.cat((dprojd_df, dprojd_dc, -dprojd_dk), dim=-1)
        return ut.fast_small_matmul(dbearing_dprojd, dprojd_dparams)

    def jac_bearings_wrt_imcoords(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Jacobian of the bearings wrt image coordinates.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, 2) Jacobians.
        """
        self.validate_params(params)
        k = params[..., self.NUM_F + 2 :]
        num_k = self.num_k
        eps = torch.finfo(bearings.dtype).eps
        iz = 1 / bearings[..., 2:].clamp(eps)
        proj_u = bearings[..., :2] * iz  # (..., N, 2)

        # 1) projection Jacobian dprojd/dbearing as dprojd/dproju * dprojud/dbearing
        powers = proj_u.new_empty((*iz.shape, num_k))  # r^2, r^4 .. (..., N, 1, num_k)
        ru_2 = ru_i = powers[..., 0, 0] = (proj_u * proj_u).sum(-1)  # (..., N)
        k = k[..., None, :]  # (..., 1, num_k)
        agg1 = 2 * k[..., 0]
        agg2 = k[..., 0] * ru_2
        for i in range(1, num_k):
            agg1 = agg1 + (2 + 2 * i) * k[..., i] * ru_i
            ru_i = powers[..., 0, i] = ru_i * ru_2
            agg2 = agg2 + k[..., i] * ru_i
        dprojd_dproju = (
            (1 + agg2)[..., None].expand(*agg2.shape, 2).diag_embed()
            + agg1[..., None, None] * proj_u[..., None] * proj_u[..., None, :]
        )  # fmt:skip # (..., N, 2, 2)
        dproju_dbearing = torch.cat(
            (iz.expand_as(proj_u).diag_embed(), (-iz * proj_u)[..., None]),
            dim=-1,
        )  # (..., N, 2, 3)

        # apply inverse function theorem to obtain dbearing/dprojd
        dbearing_dprojd = pinv_2d_jac_wrt_bearing(
            ut.fast_small_matmul(dprojd_dproju, dproju_dbearing), bearings
        )  # (..., N, 3, 2)
        dbearing_dim = dbearing_dprojd / params[..., None, None, : self.NUM_F]
        return dbearing_dim


def pinv_2d_jac_wrt_bearing(jac: Tensor, unit_bearings: Tensor) -> Tensor:
    """Compute the pseudo-inverse of a 2D Jacobian corresponding to a function whose
    input (independent variable) is a (unit) bearing vector.

    Given The Jacobian ∂f(b)/∂b, where b is a unit bearing vector and f: R^3 -> R^2,
    this function computes ∂b/∂y, where y=f(b). To do this, we use the result from
    Lemma 1 of "Tangent sampson error: Fast approximate two-view reprojection error for
    central camera models", Terekhov and Larsson.

    Args:
        jac: (..., 2, 3) Jacobian w.r.t. the bearing vector.
        unit_bearings: (..., 3) unit bearing vectors.

    Returns:
        (..., 3, 2) pseudo-inverse of the Jacobian.
    """
    gyxd_dxgx = torch.stack(
        (
            torch.linalg.cross(jac[..., 1, :], unit_bearings),
            torch.linalg.cross(unit_bearings, jac[..., 0, :]),
        ),
        dim=-1,
    )
    return gyxd_dxgx / (jac[..., 0, :] * gyxd_dxgx[..., 0]).sum(-1)[..., None, None]
