from math import ceil

import torch
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera


class NewtonThetaFromRadii(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, k: Tensor, sen_radii: Tensor, newton_iters: int, newton_tol: float
    ) -> tuple[Tensor, Tensor]:
        """Newton method for mapping sensor radii to polar angles (θ).

        Solve for θ in:
            f := -sen_radius + θ + k1*θ^3 + k2*θ^5 + ... = 0
        As initialization we follow the one used in Aria [1], which is appropriate for
        monotonically decreasing polynomials (barrel distortion).
        TODO: support alternative initializations.
        [1] Project aria: A New Tool for Egocentric Multi-modal AI Research, Engel et al., 2023.

        Args:
            k: (..., num_k) radial distortion coefficients.
            sen_radii: (..., N) radii in the sensor plane.
            newton_iters: number of Newton iterations.
            newton_tol: threshold for checking convergence.

        Returns:
            theta: (..., N) polar angles.
            converged: (..., N) boolean tensor indicating convergence.
        """
        num_k = k.shape[-1]
        # initialization based on Aria's Project (see docstring)
        theta = torch.sqrt(sen_radii)  # (..., N)
        for _ in range(newton_iters):
            # initialize polynomial and its derivative
            theta_2 = theta_i = theta**2
            factor = 1 + k[..., :1] * theta_2  # 1 + k1*θ^2 + k2*θ^4...
            f_grad = 1 + 3 * k[..., :1] * theta_2
            # remaining terms/coefficients
            for j in range(1, num_k):
                theta_i = theta_i * theta_2
                factor = factor + k[..., j, None] * theta_i
                f_grad = f_grad + (3 + 2 * j) * k[..., j, None] * theta_i
            f = theta * factor - sen_radii
            theta = theta - f / f_grad
        # check convergence
        converged = f.abs() <= newton_tol
        # tensors needed for backward
        ctx.save_for_backward(k, theta)
        ctx.mark_non_differentiable(converged)
        return theta, converged

    @staticmethod
    def backward(
        ctx, dloss_dtheta: Tensor, dloss_dconverged: Tensor
    ) -> tuple[Tensor | None, Tensor | None, None, None]:
        """Backward pass for the Newton method using the implicit function theorem.

        Args:
            dloss_dtheta: (..., N) gradient w.r.t. the undistorted theta.
            dloss_dconverged: (..., N) gradient w.r.t. the convergence flag. Ignored
                as it is not differentiable.

        Returns:
            dloss_dk: (..., num_k) gradient w.r.t. the radial distortion coefficients.
            dloss_sen_radiid: (..., N) gradient w.r.t. the sensor plane radii.
        """
        k, theta = ctx.saved_tensors
        dloss_dk = dloss_sen_radii = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            num_k = k.shape[-1]
            powers = torch.arange(3, 3 + 2 * num_k, 2, device=theta.device)  # 3, 5...
            # f_grad := 1 + 3*k1*θ^2 + 5*k2*θ^4 + ...
            powers_g = torch.arange(2, 2 + 2 * num_k, 2, device=theta.device)
            f_grad = 1 + (
                powers * k[..., None, :] * torch.pow(theta[..., None], powers_g)
            ).sum(-1)  # (..., N) # fmt: skip

        if ctx.needs_input_grad[0]:
            monomials = torch.pow(theta[..., None], powers)  # θ^3, θ^5, ...
            dtheta_dk = -monomials / f_grad[..., None]  # (..., N, num_k)
            dloss_dk = (dloss_dtheta[..., None] * dtheta_dk).sum(-2)  # (..., num_k)

        if ctx.needs_input_grad[1]:
            dloss_sen_radii = dloss_dtheta / f_grad

        return dloss_dk, dloss_sen_radii, None, None


def newton_theta_from_radii(
    k: Tensor, sen_radii: Tensor, newton_iters: int, newton_tol: float
) -> tuple[Tensor, Tensor]:
    """Newton method for mapping sensor radii to polar angles (θ)."""
    return NewtonThetaFromRadii.apply(k, sen_radii, newton_iters, newton_tol)  # type: ignore


def propagate_tangent_covs(bearings: Tensor, bs, covs: Tensor, k0: Tensor) -> Tensor:
    raise NotImplementedError


class KannalaBrandt(BaseCamera):
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

    The (ordered) intrinsic parameters are fx, fy, cx, cy, k1, k2, ...,
        - (fx, fy) [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - (k1, k2, ...) are the radial distortion coefficients.

    Args:
        num_k: number of radial distortion coefficients. Default is 4.
        newton_iters: number of Newton iterations for mapping sensor radii to polar angles (θ)
        newton_tol: threshold for checking convergence of the Newton algorithm.
    """

    NAME = "kb"
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
        num_k: int = 4,
        newton_iters: int = 25,
        newton_tol: float = 1e-5,
    ):
        if num_k < 1 or num_k > 4 or not isinstance(num_k, int):
            raise ValueError(
                f"`num_k` must be a positive integer in [1, 4] but got: {num_k=}."
            )
        self.num_k = num_k
        self.newton_iters = newton_iters
        self.newton_tol = newton_tol

    def parse_params(self, params: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Parse parameters into focal lengths, principal points, and distortion.

        Args:
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., 2) focal lengths, (..., 2) principal points, (..., num_k) distortion.
        """
        nf = self.NUM_F
        f = params[..., :nf]
        c = params[..., nf : nf + 2]
        k = params[..., nf + 2 :]
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

    @staticmethod
    def _radii_from_rays(ray_radii: Tensor, z: Tensor, k: Tensor) -> Tensor:
        """Compute radii in the sensor plane: r = θ + k1 * θ^3 + k2 * θ^5 + ...

        Args:
            ray_radii: (..., N) radii of the rays in the sensor plane.
            z: (..., N) z-coordinates of the rays.
            k: (..., num_k) radial distortion coefficients.
        """
        theta = torch.atan2(ray_radii, z)  # polar angles
        theta_2 = theta_i = theta**2
        radii = 1 + k[..., :1] * theta_2  # θ is factored out (pending to be multiplied)
        for i in range(1, k.shape[-1]):
            theta_i = theta_i * theta_2
            radii = radii + k[..., i, None] * theta_i
        return theta * radii

    def project(self, params: Tensor, points_3d: Tensor) -> tuple[Tensor, None]:
        """Project 3D points in the reference of the camera to image coordinates.

        Args:
            points_3d: (..., N, 3) 3D points in the reference of each camera.
            params: (..., D) intrinsic parameters for (...) cameras.

        Returns:
            (..., N, 2) image coordinates.
            valid: None, to indicate that all projections are valid. This is not the
                case for other camera models, and a boolean mask is returned instead.
                Actually, the KB model is undefined for the 3D point (0, 0, 0). This is
                not currently flagged (maybe a TODO). Due to Pytorch's convention of
                atan2(0, 0)=0, this point is projected to the principal point (cx, cy).
        """
        self.validate_params(params)
        f, c, k = self.parse_params(params)
        # ray and sensor radii (..., N)
        ray_radii = torch.linalg.norm(points_3d[..., :2], dim=-1)
        sen_radii = self._radii_from_rays(ray_radii, points_3d[..., 2], k)
        # image coordinates (..., 2)
        sen_coords = (
            sen_radii.unsqueeze(-1)
            * points_3d[..., :2]
            / ray_radii.unsqueeze(-1).clamp(torch.finfo(ray_radii.dtype).eps)
        )
        im_coords = f.unsqueeze(-2) * sen_coords + c.unsqueeze(-2)
        return im_coords, None

    def unproject(self, params: Tensor, points_2d: Tensor) -> tuple[Tensor, Tensor]:
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
        sen_coords = (points_2d - c.unsqueeze(-2)) / f.unsqueeze(-2)
        sen_radii = torch.linalg.norm(sen_coords, dim=-1)
        theta, valid = newton_theta_from_radii(
            k, sen_radii, self.newton_iters, self.newton_tol
        )
        bearings = torch.cat(
            (
                torch.sin(theta).unsqueeze(-1)
                * sen_coords
                / sen_radii.unsqueeze(-1).clamp(torch.finfo(sen_radii.dtype).eps),
                torch.cos(theta).unsqueeze(-1),
            ),
            dim=-1,
        )
        return bearings, valid

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
        ), f"Input sizes do not match the minimal sample size: {im_coords.shape[-2]=}, {bearings.shape[-2]=}."
        As, bs = self._form_batched_system(im_coords, bearings, cxcy)
        A = As.flatten(-3, -2)[..., : As.shape[-1], :]  # (..., D, D)
        b = bs.flatten(-2, -1)[..., : As.shape[-1]]  # (..., D)

        sol, info = torch.linalg.solve_ex(A, b)  # (..., D)
        sol[info != 0] = 1.1  # set bogus value for invalid solutions

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
        """Fit instrinsics with a non-minimal set of 2D-bearing correspondences

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances
                expressed in the tangent space of the input bearings.
            params0: (..., D) approximate estimation of the intrinsic parameters to
                propagate the covariances from the tangent space to the error space.
                Currently ignored.

        Returns:
            (..., D) fitted intrinsic parameters: fx, fy, cx, cy, k1, ...
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found (<0) or the system is singular (>0).
        """
        if covs is not None:
            raise NotImplementedError
        nf = self.NUM_F  # number of focal lengths
        As, bs = self._form_batched_system(
            im_coords, bearings, None if cxcy is None else cxcy
        )
        sol, info = ut.solve_2dweighted_lstsq_qr(As, bs)
        f = sol[..., :nf].reciprocal()
        if cxcy is None:
            c = sol[..., nf : nf + 2] * f
            return torch.cat((f, c, sol[..., nf + 2 :]), dim=-1), info
        return torch.cat((f, cxcy, sol[..., nf:]), dim=-1), info

    def jac_bearings_wrt_params(
        self, params: Tensor, bearings: Tensor, im_coords: Tensor
    ) -> Tensor:
        """Compute the Jacobian of the bearings wrt the intrinsic parameters.

        NOTE: This method assumes that all input bearings have unit norm.
        Given the unprojection function for the Kannala-Brandt model:
                                   | X |   | sin(θ) * m_x / r |
            b = π^{-1}([u, v]^T) = | Y | = | sin(θ) * m_y / r |
                                   | Z |   | cos(θ)           |
        where:
            - θ = atan2(R, Z) is the polar angle, with R = sqrt(X^2 + Y^2),
            - m_x=(u - c_x)/f_x, m_y=(v - c_y)/f_y are the sensor plane coordinates,
            - r = sqrt(m_x^2 + m_y^2) is the radius in the sensor plane,
        then, using the chain-rule, we compute this Jacobian as:
            * db/dk = db/dθ * dθ/dk
            * db/d(f, c) = db/dθ * dθ/dr * dr/d(f, c)
                           + db/d([m_x, m_y]/r) * d([m_x, m_y]/r)/d(f, c)
        where dθ/dk and dθ/dr are computed using the implicit function theorem.

        Args:
            params: (..., D) intrinsic parameters.
            bearings: (..., N, 3) unit bearing vectors.
            im_coords: (..., N, 2) image coordinates.

        Returns:
            (..., N, 3, D) Jacobians.
        """
        # intermediate variables
        self.validate_params(params)
        f, c, k = self.parse_params(params)
        if_ = f.reciprocal().unsqueeze(-2)  # (..., 1, num_f)
        k = k.unsqueeze(-2)  # (..., 1, num_k)
        num_k = self.num_k
        two_f = self.NUM_F == 2
        Z = bearings[..., 2:]
        R = torch.linalg.norm(bearings[..., :2], dim=-1, keepdim=True)  # (..., N, 1)
        theta = torch.atan2(R, Z)
        sens_coords = (im_coords - c.unsqueeze(-2)) * if_  # (..., N, 2)
        sens_iradii = (
            torch.linalg.norm(sens_coords, dim=-1, keepdim=True)
            .clamp(torch.finfo(sens_coords.dtype).eps)
            .reciprocal()
        )
        sens_cs: Tensor = sens_coords * sens_iradii

        # Jacobian of bearing w.r.t. polar angle (theta)
        db_dtheta = torch.cat((Z * sens_cs, -R), dim=-1).unsqueeze(-1)  # (..., N, 3, 1)
        # Jacobian of sensor radii w.r.t. theta and theta w.r.t. distortion coeffs
        theta_2 = theta_i = theta**2
        dsradii_dtheta = 1 + 3 * k[..., :1] * theta_2
        dtheta_dk = theta.new_empty((*theta.shape, num_k))  # (..., N, 1, num_k)
        dtheta_dk[..., 0] = theta_2 * theta
        for i in range(1, num_k):
            theta_i = theta_i * theta_2
            dsradii_dtheta = dsradii_dtheta + (3 + 2 * i) * k[..., i, None] * theta_i
            dtheta_dk[..., i] = theta_i * theta
        dtheta_dk = -dtheta_dk / dsradii_dtheta.unsqueeze(-1)  # inv. function theorem
        # Jacobian of theta w.r.t. radii - inverse function theorem
        dtheta_dsradii = dsradii_dtheta.reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        # Jacobian of sensor radii w.r.t. focal length(s) and principal point
        dsradii_dc = -if_ * sens_cs  # (..., N, 2)
        dsradii_df = (  # (..., N, num_f)
            dsradii_dc * sens_coords
            if two_f
            else (dsradii_dc * sens_coords).sum(-1, keepdim=True)
        )
        # Jacobian of sensor_cs w.r.t. focal and principal point
        dcs_dc = (if_ * sens_iradii).unsqueeze(-2) * (  # (..., N, 2, 2)
            sens_cs[..., None] * sens_cs.unsqueeze(-2)
            - torch.eye(2, device=sens_cs.device, dtype=sens_cs.dtype)
        )
        dcs_df = (
            sens_coords.unsqueeze(-2) * dcs_dc
            if two_f
            else (sens_coords.unsqueeze(-2) * dcs_dc).sum(-1, keepdim=True)
        )
        # Jacobian of bearings w.r.t. focal and principal point (..., N, 3, num_f + 2)
        db_dfc = (
            db_dtheta
            * dtheta_dsradii
            * torch.cat((dsradii_df, dsradii_dc), dim=-1).unsqueeze(-2)
        )
        db_dfc[..., :2, :] += R[..., None] * torch.cat((dcs_df, dcs_dc), dim=-1)
        # Jacobian of bearings w.r.t. distortion coeffs (..., N, 3, num_k)
        db_dk = db_dtheta * dtheta_dk  # (..., N, 3, num_k)
        return torch.cat((db_dfc, db_dk), dim=-1)

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
        f, c, k = self.parse_params(params)
        if_ = f.reciprocal().unsqueeze(-2)  # (..., 1, num_f)
        k = k.unsqueeze(-2)  # (..., 1, num_k)
        num_k = self.num_k
        Z = bearings[..., 2:]
        R = torch.linalg.norm(bearings[..., :2], dim=-1, keepdim=True)  # (..., N, 1)
        theta = torch.atan2(R, Z)
        sens_coords = (im_coords - c.unsqueeze(-2)) * if_  # (..., N, 2)
        sens_iradii = (
            torch.linalg.norm(sens_coords, dim=-1, keepdim=True)
            .clamp(torch.finfo(sens_coords.dtype).eps)
            .reciprocal()
        )
        sens_cs: Tensor = sens_coords * sens_iradii

        # Jacobian of bearing w.r.t. polar angle (theta)
        db_dtheta = torch.cat((Z * sens_cs, -R), dim=-1).unsqueeze(-1)  # (..., N, 3, 1)
        # Jacobian of sensor radii w.r.t. theta
        theta_2 = theta_i = theta**2
        dsradii_dtheta = 1 + 3 * k[..., :1] * theta_2
        for i in range(1, num_k):
            theta_i = theta_i * theta_2
            dsradii_dtheta = dsradii_dtheta + (3 + 2 * i) * k[..., i, None] * theta_i
        # Jacobian of theta w.r.t. radii - inverse function theorem
        dtheta_dsradii = dsradii_dtheta.reciprocal().unsqueeze(-1)  # (..., N, 1, 1)
        # Jacobian of sensor radii w.r.t. image coordinates
        dsradii_dim = if_ * sens_cs  # (..., N, 2)
        # Jacobian of sensor_cs w.r.t. image coordinates
        dcs_dim = (-if_ * sens_iradii).unsqueeze(-2) * (  # (..., N, 2, 2)
            sens_cs[..., None] * sens_cs.unsqueeze(-2)
            - torch.eye(2, device=sens_cs.device, dtype=sens_cs.dtype)
        )
        # Jacobian of bearings w.r.t. image coordinates
        db_dim = db_dtheta * dtheta_dsradii * dsradii_dim.unsqueeze(-2)
        db_dim[..., :2, :] += R[..., None] * dcs_dim
        return db_dim

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
        eps = torch.finfo(ray_radii.dtype).eps
        # form eqs corresponding to focal length(s) and principal point
        if cxcy is None:
            As = bearings.new_zeros((*theta.shape[:-1], 2, 4 + num_k))
            As[..., 0, 0] = im_coords[..., 0]
            As[..., 1, 1] = im_coords[..., 1]
            As[..., 0, 2] = As[..., 1, 3] = -1
            offset = 4
        else:
            As = bearings.new_zeros((*theta.shape[:-1], 2, 2 + num_k))
            As[..., 0, 0] = im_coords[..., 0] - cxcy[..., 0, None]
            As[..., 1, 1] = im_coords[..., 1] - cxcy[..., 1, None]
            offset = 2
        # form RHS
        bs = theta * bearings[..., :2] / ray_radii.clamp(eps)  # (..., N, 2)
        # eqs corresponding to distortion terms
        theta_2 = theta**2
        coeff_i = -theta_2 * bs
        As[..., offset] = coeff_i
        for i in range(1, num_k):
            coeff_i = coeff_i * theta_2
            As[..., offset + i] = coeff_i
        return As, bs
