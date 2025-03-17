import torch
import torch.nn.functional as F
from torch import Tensor


class Unit3:
    """Class utility for the S^2 manifold (unit sphere in R^3)."""

    # rays whose relative angle θ satifies cos(θ)>=1-EPS_PARALLEL are considered parallel
    EPS_PARALLEL = 1e-4  # ~ 0.81°

    @staticmethod
    def get_tangent_basis(unit_vecs: Tensor) -> Tensor:
        """Compute the tangent basis at the given unit vectors.

        We define the basis vectors of the tangent plane at a point p in the unit sphere
        as the derivatives of the unit vectors with respect to the spherical coordinates,
        θ and φ, i.e. along the direction of the parallel and meridian, respectively, at p:
            basis_vector_0 = ∂p/∂θ,
            basis_vector_1 = ∂p/∂φ,
        Since we define the mapping from spherical to cartesian coordinates as:
            x = sin(θ) * cos(φ),
            y = sin(φ),
            z = cos(θ) * cos(φ),
        the tangent basis vectors are then given by:
            basis_vector_0 = (cos(θ) * cos(φ), 0, -sin(θ) * cos(φ))^T,
            basis_vector_1 = (-sin(θ) * sin(φ), cos(φ), -cos(θ) * sin(φ))^T.

        Args:
            unit_vecs: (..., 3) unit vectors.

        Returns:
            (..., 3, 2) tangent basis vectors.
        """
        # nonzero elements of the first basis vector
        ctheta_cphi = unit_vecs[..., 2]
        stheta_cphi = unit_vecs[..., 0]
        # at poles, the parallel is independent of theta, so we hardcode it to (1, 0, 0)
        sphi = unit_vecs[..., 1]
        is_pole = sphi.abs() >= (1 - torch.finfo(sphi.dtype).eps)
        ctheta_cphi = torch.where(is_pole, 1, ctheta_cphi)
        stheta_cphi = torch.where(is_pole, 0, stheta_cphi)
        # form first basis vector
        parallel_vecs = F.normalize(torch.stack(
            (ctheta_cphi, torch.zeros_like(ctheta_cphi), -stheta_cphi), dim=-1
        ), dim=-1)  # (..., 3) # fmt:skip
        # form second basis vector
        meridian_vecs = torch.linalg.cross(unit_vecs, parallel_vecs, dim=-1)
        tangent_bases = torch.stack((parallel_vecs, meridian_vecs), dim=-1)  # (..,3, 2)
        return tangent_bases

    @staticmethod
    def expmap(unit_vecs: Tensor, tangent_coords: Tensor) -> Tensor:
        """Exponential map on the unit sphere [1, Example 10.21].

        Exponential map of S^2 at (x, v) in T_xS^2, i.e., at the tangent plane of the
        sphere at point x:
            exp_x(v) = cos(ǁvǁ) * x + sin(ǁvǁ)/ǁvǁ * v,
            with   v = Basis_x * tangent_coords,
        where:
            - x is the point on the sphere,
            - v is the tangent vector at x, defined in the same coordinate system as x.
            - Basis_x is the tangent-space basis at x, computed in `get_tangent_basis`.

        [1] An introduction to optimization on smooth manifolds, N. Boumal, 2023.

        Args:
            unit_vecs: (..., 3) Unit vectors in S^2.
            tangent_coords: (..., 2) Coordinates of the tangent vectors, expressed in
                the tangent space spanned by the bases defined with the function
                `get_tangent_basis`.

        Returns:
            (..., 3) Exponential map of v at `unit_vecs`.
        """
        assert unit_vecs.shape[-1] == 3 and tangent_coords.shape[-1] == 2
        # express tangent coords in the same coordinate system as the unit vectors
        basis = Unit3.get_tangent_basis(unit_vecs)  # (..., 3, 2)
        v = (basis * tangent_coords[..., None, :]).sum(-1)  # (..., 3)
        # geodesic (angular) distance between the unit vectors
        # NOTE: this angle is also equal to the norm of v
        theta = torch.linalg.norm(tangent_coords, dim=-1, keepdim=True)  # (..., 1)
        small = theta <= torch.finfo(theta.dtype).eps  # (..., 1)
        theta_guard = torch.where(small, 1, theta)  # guard for 0 div in θ/sin(θ)
        exp = torch.cos(theta) * unit_vecs + torch.where(
            small, v, torch.sin(theta_guard) * v / theta_guard
        )
        return exp

    @staticmethod
    def expmap_at_z1(tangent_coords: Tensor) -> Tensor:
        """Particularization of the exponential map using the tangent plane at (0, 0, 1).

        Args:
            tangent_coords: (..., 2) Coordinates of the tangent vectors, expressed in
                the tangent space spanned by the bases defined with the function
                `get_tangent_basis` at the point (0, 0, 1).

        Returns:
            (..., 3) Exponential map of v at (0, 0, 1).
        """
        assert tangent_coords.shape[-1] == 2
        v_xy = tangent_coords[..., :2]
        # geodesic (angular) distance between vectors to be exponentiated and (0, 0, 1)
        theta = torch.linalg.norm(tangent_coords, dim=-1, keepdim=True)
        small = theta <= torch.finfo(theta.dtype).eps  # (..., 1)
        theta_guard = torch.where(small, 1, theta)  # guard for 0 div in θ/sin(θ)
        scaled_v = torch.where(small, v_xy, torch.sin(theta_guard) * v_xy / theta_guard)
        exp = torch.cat((scaled_v, torch.cos(theta)), dim=-1)
        return exp

    @staticmethod
    def logmap(ref_vecs: Tensor, vecs: Tensor) -> Tensor:
        """Logarithmic map on the unit sphere [1, Example 10.21].

        This function computes the vector v such that Exp_{ref_vec}(v) = vec. v is
        defined in the tangent space of each unit vector, i.e., in the space spanned by
        the basis defined with the function `get_tangent_basis`. The Logarithm map is
        given by:
            Log_x(y) = Basis_x^T (acos(x^T y) (y - (x^T y) x) / ǁy - (x^T y) xǁ),
                     = Basis_x^T θ/sin(θ) * (y - cos(θ) * x),
        where:
            * x := ref_vec, y := vec,
            * θ := acos(x^T y) is the geodesic (angular) distance between x and y, and
            * Basis_x is the tangent-space basis at x, computed in `get_tangent_basis`.
        Finally, since Basis_x is perpendicular to x, we can simplify the logarithm map as:
            Log_x(y) = θ/sin(θ) Basis_x^T * y.

        [1] An introduction to optimization on smooth manifolds, N. Boumal, 2023.

        Args:
            ref_vecs: (..., 3) Reference unit vectors.
            vecs: (..., 3) Unit vectors to map to the tangent space at `ref_vecs`.

        Returns:
            (..., 2) Coordinates of the tangent vectors, expressed in the tangent space
            spanned by the bases defined with the function `get_tangent_basis`
        """
        assert ref_vecs.shape[-1] == vecs.shape[-1] == 3
        cos_theta = (ref_vecs * vecs).sum(-1, keepdim=True)  # (..., 1)
        not_parallel = cos_theta < 1 - Unit3.EPS_PARALLEL
        # guard for 0 div in theta/sin(theta)
        theta = torch.where(not_parallel, Unit3.distance(ref_vecs, vecs)[..., None], 1)
        # correct scale for non-parallel vectors
        scaled_y = torch.where(not_parallel, theta / torch.sin(theta) * vecs, vecs)
        # project to local coords
        tangent_bases = Unit3.get_tangent_basis(ref_vecs)  # (..., 3, 2)
        local_coords = (tangent_bases * scaled_y[..., None]).sum(-2)
        return local_coords

    @staticmethod
    def logmap_at_z1(vecs: Tensor) -> Tensor:
        """Particularization of the logarithmic map using the tangent plane at (0, 0, 1).

        Since Basis_x^T at (0, 0, 1) represents the XY axes as computed in
        `get_tangent_basis`, and:
            Log_x(y) = Basis_x^T θ/sin(θ) * (y - cos(θ) * x),
        this method computes the logarithmic map at the point (0, 0, 1) as:
            Log_x(y) = θ/sin(θ) * (y_x, y_y)^T,

        Args:
            vecs: (..., 3) Unit vectors to map to the tangent space at (0, 0, 1).

        Returns:
            (..., 2) Coordinates of the tangent vectors, expressed in the tangent space
            spanned by the bases defined with the function `get_tangent_basis`.
        """
        assert vecs.shape[-1] == 3
        z1 = torch.tensor([0.0, 0.0, 1.0], device=vecs.device, dtype=vecs.dtype)
        # cosθ = z
        not_parallel = vecs[..., 2:] < 1 - Unit3.EPS_PARALLEL
        # guard for 0 div in theta/sin(theta)
        theta = torch.where(not_parallel, Unit3.distance(z1, vecs)[..., None], 1)
        # logmap
        local_coords = torch.where(
            not_parallel, theta / torch.sin(theta) * vecs[..., :2], vecs[..., :2]
        )
        return local_coords

    @staticmethod
    def distance(x: Tensor, y: Tensor) -> Tensor:
        """Compute the geodesic distance between pairs of points on the unit sphere.

        The geodesic distance between two points x and y on the unit sphere is given by
            d(x, y) = acos(x^T y),      i.e. the angle between the vectors x and y,
        However, acos becomes numerically unstable as the angle approaches 0 or π, and
        its gradient is highly unstable in this regime, reaching infty at an angle 0.
        Thereby we use the equivalent and more stable:
            d(x, y) = 2*asin(0.5 * ǁx - yǁ).
        which stems from: 1) sin(θ/2) = sqrt(0.5(1-cosθ)) and 2) ǁx - yǁ^2 = 2 - 2cosθ.
        However, this alternative is less accurate for big angles, but they are expected
        to be less frequent in practice.
        TODO: for big relative angles (points close to be antipodal), their distance
        can be more accurately computed as: d(x, y) = pi - 2*asin(0.5 * ǁx + yǁ).
        Since asin(0.5 ǁx - yǁ) = asin(0.5 x^T y) and asin(-a) = pi - asin(a).

        Args:
            x: (..., 3) Unit vectors.
            y: (..., 3) Unit vectors.

        Returns:
            (...) Geodesic distance between x and y in radians [0, π].
        """
        chordal_dist = torch.linalg.norm(x - y, dim=-1)
        eps = torch.finfo(chordal_dist.dtype).eps
        geodesic_dist = 2 * torch.asin((0.5 * chordal_dist).clamp(0, 1 - eps))
        return geodesic_dist

    @staticmethod
    def jac_logmap_wrt_vecs(ref_vecs: Tensor, vecs: Tensor) -> Tensor:
        """Compute the Jacobian of the logarithm map w.r.t. vecs.

        Since the Logarithm map is given by (see docstrings of `logmap` and `distance`):
            Log_x(y) = θ/sin(θ) Basis_x^T * y,
        where we compute θ and cos(θ) as:
            θ = 2*asin(0.5 * ǁx - yǁ),  cos(θ) = x^T y,
        The Jacobian dLog_x(y) /dy can thus be computed via the product and chain rules:
            dLog_x(y)/dy = Basis_x^T * y * d(θ/sin(θ))/dθ * dθ/dy + θ/sin(θ) * Basis_x^T,
        where:
            - d(θ/sin(θ))/dθ = (1 - θ*cos(θ)/sin(θ)) / sin(θ),
            - dθ/dy = -(x-y)^T / sin(θ),
        However, care is needed as θ -> 0, to avoid the singularity note that if θ -> 0:
            - θ/sin(θ) -> 1 + O(θ^2),
            - (1- θ*cos(θ)/sin(θ)) / sin^2(θ) -> 1/3 + O(θ^2),
        thereby, we can approximate the Jacobian in this regime as:
            dLog_x(y)/dy = -(1/3) * Basis_x^T * y * (x-y)^T + Basis_x^T.

        Args:
            ref_vecs: (..., 3) Reference unit vectors.
            vecs: (..., 3) Unit vectors that are mapped via the Logarithm to the tangent
                space at `ref_vecs`.

        Returns:
            (..., 2, 3) Jacobian of the logarithm map w.r.t. vecs.
        """
        assert ref_vecs.shape[-1] == vecs.shape[-1] == 3
        x = ref_vecs.view(-1, 3)
        y = vecs.view(-1, 3)
        # local_coords = basis^T * 3d_tangent_vecs -> dlog_dtvecs = basis^T
        basis = Unit3.get_tangent_basis(x).transpose(-2, -1)  # (n, 2, 3)
        by_xyt = (basis * y[:, None]).sum(-1, keepdim=True) * (x - y)[:, None]  # (n, 2, 3) # fmt:skip

        cos_theta = (x * y).sum(-1, keepdim=True)[:, None]  # (n, 1, 1)
        not_parallel = cos_theta < 1 - Unit3.EPS_PARALLEL
        cos_theta = torch.where(not_parallel, cos_theta, 0)  # guard for 0 div in cosec2

        cosec2 = 1 / (1 - cos_theta**2)
        theta_cosec = Unit3.distance(x, y)[:, None, None] * torch.sqrt(cosec2)  # θ/sinθ

        # t1 = (1- θ*cos(θ)/sin(θ)) / sin^2(θ)
        t1 = torch.where(not_parallel, cosec2 * (1 - cos_theta * theta_cosec), 1 / 3)
        dlog_dy = -t1 * by_xyt + torch.where(not_parallel, theta_cosec, 1) * basis
        return dlog_dy.view(*torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]), 2, 3)

    @staticmethod
    def jac_logmap_wrt_vecs_at_z1(vecs: Tensor) -> Tensor:
        """Particularization of the Jacobian of the logarithm map w.r.t. vecs at (0, 0, 1).

        Args:
            vecs: (..., 3) Unit vectors that are mapped via the Logarithm to the tangent
                space at (0, 0, 1).

        Returns:
            (..., 2, 3) Jacobian of the logarithm map w.r.t. vecs.
        """
        assert vecs.shape[-1] == 3
        z1 = torch.tensor([0.0, 0.0, 1.0], device=vecs.device, dtype=vecs.dtype)
        y = vecs.view(-1, 3)
        # local_coords = basis^T * 3d_tangent_vecs -> dlog_dtvecs = basis^T
        by_xyt = y[:, :2, None] * (z1 - y)[:, None]  # (n, 2, 3)

        cos_theta = y[:, 2:, None]  # (n, 1, 1)
        not_parallel = cos_theta < 1 - Unit3.EPS_PARALLEL
        cos_theta = torch.where(not_parallel, cos_theta, 0)  # guard for 0 div in cosec2

        cosec2 = 1 / (1 - cos_theta**2)
        theta_cosec = Unit3.distance(z1, y)[:, None, None] * torch.sqrt(cosec2)  # θ/sinθ # fmt:skip

        # t1 = (1- θ*cos(θ)/sin(θ)) / sin^2(θ)
        t1 = torch.where(not_parallel, cosec2 * (1 - cos_theta * theta_cosec), 1 / 3)
        dlog_dy = -t1 * by_xyt + torch.where(not_parallel, theta_cosec, 1) * torch.eye(
            2, 3, device=vecs.device, dtype=vecs.dtype
        )
        return dlog_dy.view(*vecs.shape[:-1], 2, 3)

    @staticmethod
    def jac_logmap_wrt_refvecs(ref_vecs: Tensor, vecs: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the Jacobian of the logarithm map w.r.t. ref_vecs (x).

        Since the Logarithm map is given by (see docstrings of `logmap` and `distance`):
            Log_x(y) = θ/sin(θ) Basis_x^T * y,
        where we compute θ and cos(θ) as:
            θ = 2*asin(0.5 * ǁx - yǁ),  cos(θ) = x^T y,
        The Jacobian dLog_x(y) /dx can thus be computed via the product and chain rules:
            dLog_x(y)/dx = Basis_x^T * y * d(θ/sin(θ))/dθ * dθ/dx
                           + θ/sin(θ) * d(Basis_x^T * y)/dx,
        where:
            - d(θ/sin(θ))/dθ = (1 - θ*cos(θ)/sin(θ)) / sin(θ),
            - dθ/dx = (x-y)^T / sin(θ),
            - d(Basis_x^T * y)/dx depends on the basis definition at `get_tangent_basis`.
        However, care is needed as θ -> 0, to avoid the singularity note that if θ -> 0:
            - θ/sin(θ) -> 1 + O(θ^2),
            - (1- θ*cos(θ)/sin(θ)) / sin^2(θ) -> 1/3 + O(θ^2),
        thereby, we can approximate the Jacobian in this regime as:
            dLog_x(y)/dx = (1/3) * Basis_x^T * y + d(Basis_x^T * y)/dx.

        Args:
            ref_vecs: (..., 3) Reference unit vectors.
            vecs: (..., 3) Unit vectors that are mapped via the Logarithm to the tangent
                space at `ref_vecs`.

        Returns:
            dlog_dx: (..., 2, 3) Jacobian of the logarithm map w.r.t. ref_vecs.
            valid: (...,) boolean mask for valid jacobians. This maks flags ref_vecs (x)
                near the poles-φ~90-, since the local basis is not differentiable w.r.t.
                ref_vecs at these two extreme points.
        """
        assert ref_vecs.shape[-1] == vecs.shape[-1] == 3
        x = ref_vecs.view(-1, 3)
        y = vecs.view(-1, 3)
        # local_coords = basis^T * 3d_tangent_vecs
        basis = Unit3.get_tangent_basis(x).transpose(-2, -1)  # (n, 2, 3)

        # left term of dlog_dx
        by_xyt = (basis * y[:, None]).sum(-1, keepdim=True) * (x - y)[:, None]  # (n, 2, 3) # fmt:skip
        cos_theta = (x * y).sum(-1, keepdim=True)[:, None]  # (n, 1, 1)
        not_parallel = cos_theta < 1 - Unit3.EPS_PARALLEL
        cos_theta = torch.where(not_parallel, cos_theta, 0)  # guard for 0 div in cosec2
        cosec2 = 1 / (1 - cos_theta**2)
        theta_cosec = Unit3.distance(x, y)[:, None, None] * torch.sqrt(cosec2)  # θ/sinθ
        # t1 = (1- θ*cos(θ)/sin(θ)) / sin^2(θ)
        t1 = torch.where(not_parallel, cosec2 * (1 - cos_theta * theta_cosec), 1 / 3)
        dlog_dx_term1 = t1 * by_xyt

        # right term of dlog_dx
        eps = torch.finfo(basis.dtype).eps
        x0, x1, x2 = x.T
        y0, y1, y2 = y.T
        # the following is sqrt(x0^2 + x2^2)^3 (n,) which is singular at poles
        inorm_x0x2_cub = basis[:, 1, 1].pow(3).clamp(eps).reciprocal()
        valid = inorm_x0x2_cub < (1 / eps)
        # derivative of (Basis_x^T * y) w.r.t. x
        dlog_dx_term2 = torch.zeros_like(basis)  # (n, 2, 3)
        # common terms
        xy0_xy2 = x0 * y0 + x2 * y2
        x012 = x0 * x1 * x2
        # fill
        dlog_dx_term2[:, 0, 0] = -x2 * xy0_xy2 * inorm_x0x2_cub
        dlog_dx_term2[:, 0, 2] = x0 * xy0_xy2 * inorm_x0x2_cub
        dlog_dx_term2[:, 1, 0] = (
            x0**3 * y1 + x012 * y2 + x0 * x2**2 * y1 - x1 * x2**2 * y0
        ) * inorm_x0x2_cub
        dlog_dx_term2[:, 1, 1] = -xy0_xy2 * basis[:, 1, 1].clamp(eps).reciprocal()
        dlog_dx_term2[:, 1, 2] = (
            -(x0**2) * x1 * y2 + x0**2 * x2 * y1 + x012 * y0 + x2**3 * y1
        ) * inorm_x0x2_cub
        # correct Jacobian for non-(near-)parallel vectors
        dlog_dx_term2 = torch.where(
            not_parallel, theta_cosec * dlog_dx_term2, dlog_dx_term2
        )
        return (dlog_dx_term1 + dlog_dx_term2).view(*ref_vecs.shape[:-1], 2, 3), valid
