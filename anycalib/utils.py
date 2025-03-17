from math import pi

import torch
from torch import Tensor

RAD2DEG = 180 / pi


def fast_small_matmul(a: Tensor, b: Tensor) -> Tensor:
    """Fast batched matrix multiplication for small matrices (size < 8x8).

    Code from Georg Bökman and Horace He:
    https://discuss.pytorch.org/t/multiplying-large-batches-of-small-matrices-fast/201181/2

    Args:
        a: (..., N, M) tensor.
        b: (..., M, P) tensor.

    Returns:
        (..., N, P) matmul result.
    """
    return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(dim=-2)


def solve_2dweighted_lstsq(
    As: Tensor, bs: Tensor, Ws: Tensor | None = None, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Solve linear least-squares by forming the normal equations + LU decomposition.

    This function solves the (possibly weighted or/and masked) linear system of equations
    by forming the normal equations an using LU decomposition. As such, this function is
    fast, at the expense of reduced numerical stability that may be significant for
    ill-conditioned systems.

    Args:
        As: (..., N, 2, D) stacked design matrices.
        bs: (..., N, 2) observations.
        Ws: (..., N, 2, 2) matrix weights for each 2D error.
        mask: (..., N) boolean mask for valid observations.

    Returns:
        (..., D) least-squares solution.
        (...,) integer tensor indicating success. 0 if successful. Otherwise, an
            illegal value was found (<0) or the system is singular (>0).
    """
    WAs = As if Ws is None else fast_small_matmul(Ws, As)
    WAs = WAs if mask is None else WAs * mask[..., None, None]
    AtW = WAs.flatten(-3, -2).transpose(-1, -2)  # (..., D, 2*N)
    AtWA = AtW @ As.flatten(-3, -2)  # (..., D, D)
    AtWb = AtW @ bs.flatten(-2, -1)[..., None]  # (..., D, 1)
    sol, info = torch.linalg.solve_ex(AtWA, AtWb.squeeze(-1))
    sol = sol.nan_to_num(1, 1, 1)
    return sol, info


def solve_2dweighted_lstsq_qr(
    As: Tensor, bs: Tensor, Ws: Tensor | None = None, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Solve linear least-squares by with QR decomposition.

    This method is more numerically accurate than solving the system by forming the
    normal equations, at the expense of being slower.

    Args:
        As: (..., N, 2, D) stacked design matrices.
        bs: (..., N, 2) observations.
        Ws: (..., N, 2, 2) matrix weights for each 2D error.
        mask: (..., N) boolean mask for valid observations.

    Returns:
        (..., D) least-squares solution.
        (...,) integer tensor indicating success. 0 if successful. Currently it just
            checks if the solution is finite. TODO: based this also on the residuals.
    """
    if Ws is not None:
        Ws_chol, info = torch.linalg.cholesky_ex(Ws)
        mask_ = ((info == 0) if mask is None else mask & (info == 0)).unsqueeze(-1)
        WAs = fast_small_matmul(Ws_chol, As) * mask_[..., None]
        Wbs = fast_small_matmul(Ws_chol, bs[..., None]).squeeze(-1) * mask_
    else:
        WAs = As if mask is None else As * mask[..., None, None]
        Wbs = bs if mask is None else bs * mask[..., None]
    results = torch.linalg.lstsq(
        WAs.flatten(-3, -2), Wbs.flatten(-2, -1), driver="gels"
    )
    info = torch.where(results.solution.isfinite().all(dim=-1), 0, 1)
    return results.solution, info


def cxcy_and_pix_ar_from_rays(
    im_coords: Tensor, rays: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Estimate the principal point and pixel aspect ratio (f_y/f_x) from a set of rays.

    This function works for (radial) camera models whose projection function has the form:
        u = f_x * r(R, Z, params)*X + c_x
        v = f_y * r(R, Z, params)*Y + c_y
    where r(R, Z) is the function that maps the input ray (X, Y, Z) to its radial
    distance in the sensor plane and R := sqrt(X^2 + Y^2).

    Args:
        im_coords: (..., N, 2) image coordinates.
        rays: (..., N, 3) rays.
        cxcy: (..., 2) principal point.

    Returns:
        (..., 2) principal point.
        (...,) pixel aspect ratio.
        (...,) integer tensor indicating success. 0 if successful.
    """
    # form linear system
    A = rays.new_empty(rays.shape)  # (..., N, 3)
    A[..., 0] = rays[..., 1] * im_coords[..., 0]
    A[..., 1] = -rays[..., 1]
    A[..., 2] = rays[..., 0]
    AtA = A.transpose(-1, -2) @ A  # (..., 3, 3)
    Atb = A.transpose(-1, -2) @ (im_coords[..., 1:] * rays[..., :1])  # (..., 3)
    sol, info = torch.linalg.solve_ex(AtA, Atb.squeeze(-1))
    pix_ar = sol[..., 0]
    cxcy = torch.stack((sol[..., 1] / pix_ar, sol[..., 2]), dim=-1)
    return cxcy, pix_ar, info


def cxcy_from_rays(im_coords: Tensor, rays: Tensor) -> tuple[Tensor, Tensor]:
    """Estimate the principal point from a set of rays. This function assumes that
    pixels are perfect squares, i.e., that pixel aspect ratio (f_y/f_x) = 1.

    This function works for (radial) camera models whose projection function has the form:
        u = f * r(R, Z, params)*X + c_x
        v = f * r(R, Z, params)*Y + c_y
    where r(R, Z, params) is the function that maps the input ray (X, Y, Z) to its radial
    distance in the sensor plane and R := sqrt(X^2 + Y^2).

    Args:
        im_coords: (..., N, 2) image coordinates.
        rays: (..., N, 3) rays.
        cxcy: (..., 2) principal point.

    Returns:
        (..., 2) principal point.
        (...,) integer tensor indicating success. 0 if successful.
    """
    # form linear system
    A = torch.stack((rays[..., 1], -rays[..., 0]), dim=-1)  # (..., N, 2)
    b = (im_coords.flip(-1) * rays[..., :2]).diff(dim=-1)  # (..., N, 1)
    AtA = A.transpose(-1, -2) @ A  # (..., 3, 3)
    Atb = A.transpose(-1, -2) @ b  # (..., 3)
    cxcy_sol, info = torch.linalg.solve_ex(AtA, Atb.squeeze(-1))
    return cxcy_sol, info


def pixel_aspect_ratio_from_rays(
    im_coords: Tensor, rays: Tensor, cxcy: Tensor
) -> Tensor:
    """Compute the pixel aspect ratio f_x/f_y from a set of rays.

    This method estimates the pixel aspect ratio (f_x/f_y) for radial camera models
    whose projection function has the form:
        u = f * r(R, Z, params)*X + c_x
        v = f * r(R, Z, params)*Y + c_y
    where r(R, Z) is the function that maps the input ray (X, Y, Z) to its radial
    distance in the sensor plane and R := sqrt(X^2 + Y^2).

    Args:
        im_coords: (..., N, 2) image coordinates.
        rays: (..., N, 3) rays.
        cxcy: (..., 2) principal point.

    Returns:
        (...,) pixel aspect ratio.
    """
    # fx/fy = (u-cx)*Y / (v-cy)*X
    num = torch.abs((im_coords[..., 0] - cxcy[..., 0]) * rays[..., 1])  # (..., N)
    den = torch.abs((im_coords[..., 1] - cxcy[..., 1]) * rays[..., 0])
    # mask out rays with X=0 or Y=0 and image coordinates with u-c_x = 0 or v-c_y = 0
    eps = torch.finfo(num.dtype).eps
    mask = (num > eps) & (den > eps)
    num = torch.where(mask, num, 1)
    den = torch.where(mask, den, 1)
    pixel_ar = masked_mean(num / den, mask, dim=-1)
    return pixel_ar


def masked_mean(a: Tensor, mask: Tensor, dim: int | tuple[int, ...]) -> Tensor:
    """Compute the mean of a tensor along a dimension, ignoring masked values.

    Args:
        a: Tensor to compute the mean.
        mask: Boolean mask. Must be broadcastable to a.shape.
        dim: Dimension along which to compute the mean.

    Returns:
        Mean of the tensor.
    """
    return (a * mask).sum(dim) / mask.sum(dim).clamp(min=1)
