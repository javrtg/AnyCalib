from math import exp, log

import torch
from torch import Tensor

import anycalib.utils as ut
from anycalib.cameras import CameraFactory
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3


def normalized_im_radii(pred: dict, data: dict, offset: float = 0.1):
    cam = CameraFactory.create_from_id(data["cam_id"])
    params = data["intrinsics"][: len(pred["intrinsics"])].clone()
    params[..., cam.PARAMS_IDX["cx"]] = params[..., cam.PARAMS_IDX["cy"]] = 0
    radii = torch.linalg.norm(cam.project(params, data["rays"]), dim=-1)  # FIXME
    return radii / radii.max() + offset


def angular_error(pred: dict, data: dict) -> Tensor:
    """L2 norm / geodesic distance of the error in either tangent plane

    NOTE: this norm is agnostic to chosen tangent plane.

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (..., H*W) tensor with the L2 norm of the error in the tangent plane.
    """
    return Unit3.distance(data["rays"], pred["rays"])


def cosine_sim(pred: dict, data: dict) -> Tensor:
    """Cosine similarity between predicted and ground-truth rays

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (..., H*W) tensor with the cosine similarity between predicted and ground-truth rays.
    """
    return 1 - (data["rays"] * pred["rays"]).sum(-1)


def squared_tangent_error(pred: dict, data: dict) -> Tensor:
    """Squared L2 norm / geodesic distance of the error in either tangent plane

    NOTE: this norm is agnostic to chosen tangent plane.

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (..., H*W) tensor with the squared L2 norm of the error in the tangent plane.
    """
    return Unit3.distance(data["rays"], pred["rays"]) ** 2


def squared_tangent_error_at_z1(pred: dict, data: dict) -> Tensor:
    """Squared L2 norm / geodesic distance of the error in the tangent plane at point
    (0, 0, 1) i.e. optical axis

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - or "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.
    """
    # tangent_coords = (
    #     pred["tangent_coords"] if "tangent_coords" in pred else Unit3.logmap_at_z1(pred["rays"])
    # )
    gt_tangent_coords = Unit3.logmap_at_z1(data["rays"])
    return (pred["tangent_coords"] - gt_tangent_coords).square().sum(-1)


def absolute_tangent_error(pred: dict, data: dict, tangent: str = "data") -> Tensor:
    """L1 norm of the error in the tangent plane defined by `tangent`

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (..., H*W) tensor with the L1 norm of the error in the tangent plane.
    """
    assert tangent in ("data", "pred"), f"Unknown tangent plane: {tangent}"
    if tangent == "data":
        return Unit3.logmap(data["rays"], pred["rays"]).abs().sum(-1)
    return Unit3.logmap(pred["rays"], data["rays"]).abs().sum(-1)


def absolute_tangent_error_at_z1(pred: dict, data: dict) -> Tensor:
    """L1 norm of the error in the tangent plane at point (0, 0, 1) i.e. optical axis

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - or "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.
    """
    gt_tangent_coords = Unit3.logmap_at_z1(data["rays"])
    return (pred["tangent_coords"] - gt_tangent_coords).abs().sum(-1)


def absolute_tangent_error_at_z1_radial(pred: dict, data: dict) -> Tensor:
    """L1 norm of the error in the tangent plane at point (0, 0, 1) i.e. optical axis

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - or "rays": (..., H*W, 3) tensor with predicted rays.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.
    """
    gt_tangent_coords = Unit3.logmap_at_z1(data["rays"])
    elwise_err = (pred["tangent_coords"] - gt_tangent_coords).abs().sum(-1)
    return elwise_err * normalized_im_radii(pred, data)


def gaussian_nll(pred: dict, data: dict, tangent: str = "data") -> Tensor:
    """Negative log-likelihood loss for Gaussian distribution

    This loss assumes that the 2D error expressed in the tangent plane chosen in the
    `tangent` argument follows a Gaussian distribution with *diagonal* covariance matrices
    (shape (B, H*W, 2)) that are expressed in the same chosen tangent plane.

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays.
            - "log_covs": (..., H*W, 2) tensor with predicted log-covariances.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (B, H*W) tensor with the negative log-likelihood for each ray.
    """
    assert tangent in ("data", "pred"), f"Unknown tangent plane: {tangent}"
    log_covs: Tensor = pred["log_covs"]  # (B, H*W, 2)

    tangent_errs = (
        Unit3.logmap(data["rays"], pred["rays"])
        if tangent == "data"
        else Unit3.logmap(pred["rays"], data["rays"])
    )
    # nll = C + 0.5*log(det(Σ)) + 0.5*(x-μ)^T*Σ^{-1}*(x-μ)
    # NOTE: log(det(Σ)) = log(σ1*σ2) = log(σ1) + log(σ2)
    nll = 0.5 * (log_covs.sum(dim=-1) + (tangent_errs**2 * torch.exp(-log_covs)).sum(dim=-1))
    return nll


def gaussian_nll_at_z1(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss computed in tangent plane at point (0, 0, 1)

    This loss assumes that the 2D error expressed in the tangent plane at point (0, 0, 1)
    follows a Gaussian distribution with *diagonal* covariance matrices (shape (B, H*W, 2))
    that are expressed in the same tangent plane.

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - "log_covs": (..., H*W, 2) tensor with predicted log-covariances.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (B, H*W) tensor with the negative log-likelihood for each ray.
    """
    log_covs: Tensor = pred["log_covs"]  # (B, H*W, 2)
    err: Tensor = pred["tangent_coords"] - Unit3.logmap_at_z1(data["rays"])
    # nll = C + 0.5*log(det(Σ)) + 0.5*(x-μ)^T*Σ^{-1}*(x-μ)
    # NOTE: log(det(Σ)) = log(σ1*σ2) = log(σ1) + log(σ2)
    nll = 0.5 * (log_covs.sum(dim=-1) + (err**2 * torch.exp(-log_covs)).sum(dim=-1))
    return nll


def gaussian_nll_at_z1_fitted(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss computed in tangent plane at point (0, 0, 1)

    This loss assumes that the 2D error expressed in the tangent plane at point (0, 0, 1)
    follows a Gaussian distribution with covariance matrices resulting from propagating
    the (approximate) covariance matrices of the MLE of the intrinsics.

    Args:
        pred: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with predicted rays,
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - "intrinsics": (..., D) tensor with predicted intrinsics,
            - "intrinsics_covs": (..., D, D) tensor with predicted intrinsics covariances.
            - "cam": corresponding camera instance.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (B, H*W) tensor with the negative log-likelihood for each ray.
    """
    cam: BaseCamera = pred["cam"]
    intrins_covs: Tensor = pred["intrinsics_covs"]  # (..., D, D)

    # propagate intrinsics covariances to tangent plane at point (0, 0, 1)
    dlogz1_dbearing = Unit3.jac_logmap_wrt_vecs_at_z1(pred["rays"])  # (..., H*W, 2, 3)
    dbearing_dintrins = cam.jac_bearings_wrt_params(
        pred["intrinsics"], pred["rays"], pred["im_coords"]
    )  # (...., H*W, 3, D)
    dlogz1_dintrins = ut.fast_small_matmul(dlogz1_dbearing, dbearing_dintrins)  # (..., H*W, 2, D)
    # FIXME: deal with singular matrices
    logz1_icovs, info = torch.linalg.inv_ex(
        dlogz1_dintrins @ intrins_covs.unsqueeze(-3) @ dlogz1_dintrins.transpose(-1, -2)
    )  # (..., H*W, 2, 2)

    # compute Gaussian nll = C + 0.5*log(det(Σ)) + 0.5*(x-μ)^T*Σ^{-1}*(x-μ)
    err: Tensor = pred["tangent_coords"] - Unit3.logmap_at_z1(data["rays"])  # (..., H*W, 2)
    nll = 0.5 * (
        torch.linalg.det(logz1_icovs).clamp(torch.finfo(logz1_icovs.dtype).eps).log()
    ) + ut.fast_small_matmul(  # (..., H*W)
        ut.fast_small_matmul(err.unsqueeze(-2), logz1_icovs), err[..., None]
    ).squeeze((-2, -1))  # fmt: skip
    return nll


def mixture_laplace_at_z1(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss for Laplace mixture distribution

    This loss assumes that each coordinate of the 2D error, expressed in the tangent
    plane at point (0, 0, 1), follow a univriate Laplace mixture distribution of two
    components. For one component, the variance is fixed to 1, while for the other
    component, the variance is learned.

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (B, H*W, 2) tensor with predicted tangent coordinates,
            - "weights": (B, H*W, 2) tensor with predicted weights.
        data: dict with at least the following key-value pair:
            - "rays": (B, H*W, 3) tensor with ground-truth rays.

    Returns:
        (B, H*W) tensor with the sum of negative log-likelihoods of each coordinate.
    """
    weights: Tensor = pred["weights"]  # (B, 1, 2) or (B, H*W, 2)
    log_covs: Tensor = pred["log_covs"]  # (B, H*W, 2)
    err: Tensor = (pred["tangent_coords"] - Unit3.logmap_at_z1(data["rays"])).abs()  # (B, H*W, 2)

    # log(mixture) = -log(exp(w1) + exp(w2)) + log(exp(mix1) + exp(mix2))
    # mix1 = weights[..., :1] - log(2) - err  # assumes b=1 for the 1st component
    mix1 = weights[..., :1] - log(2) - pred["min_b"] - err * exp(-pred["min_b"])
    mix2 = weights[..., 1:2] - log(2) - log_covs - err * torch.exp(-log_covs)
    nll_mixture = (
        torch.logsumexp(weights, dim=-1, keepdim=True)
        - torch.logsumexp(torch.stack((mix1, mix2)), dim=0)
    ).sum(-1)
    return nll_mixture


def laplace_nll_at_z1(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss computed in tangent plane at point (0, 0, 1)

    This loss assumes that each coordinate of the 2D error, expressed in the tangent
    plane at point (0, 0, 1), follow a univriate Laplace distribution.

    Args:
        pred: dict with at least the following key-value pair:
            - "tangent_coords": (..., H*W, 2) tensor with predicted tangent coordinates,
            - "log_covs": (..., H*W, 2) tensor which contains the logarithm of the scale
                parameter of the Laplace distribution. By abuse of notation, and for
                simplifying the code, we use log_b := log_covs.
        data: dict with at least the following key-value pair:
            - "rays": (..., H*W, 3) tensor with ground-truth rays.

    Returns:
        (B, H*W) tensor with the negative log-likelihood for each ray.
    """
    # for simplifying the code, and by abuse of notation, we use b := log_covs
    log_b: Tensor = pred["log_covs"]  # (B, H*W, 2)
    err: Tensor = (pred["tangent_coords"] - Unit3.logmap_at_z1(data["rays"])).abs()
    # nll = log(2) + log(b) + |x-μ|/b
    nll = log_b + err * torch.exp(-log_b)
    return nll.sum(-1)


### EDIT MAPS ###


def laplace_nll_aspect_ratio(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss for Laplace distribution over aspect ratio predictions"""
    assert pred["pix_ar_map"].shape[1] == 2 and pred["pix_ar_map"].dim() == 4
    err = (data["pix_ar"][:, None, None] - pred["pix_ar_map"][:, 0]).abs()  # (B, H, W)
    # nll = log(2) + log(b) + |x-μ|/b
    log_b = pred["pix_ar_map"][:, 1]
    nll = log_b + err * torch.exp(-log_b)
    return nll.view(log_b.shape[0], -1)  # (B, H*W)


def absolute_radii_error(pred: dict, data: dict) -> Tensor:
    h, w = data["image"].shape[-2:]
    b, hp, wp = pred["radii"].shape
    assert h / hp == w / wp and h / hp >= 1
    fac = h // hp  # for subsampling and scaling the predicted radii
    radii_gt = torch.linalg.norm(
        BaseCamera.pixel_grid_coords(h, w, pred["radii"], 0.5) - data["cxcy_gt"][:, None, None],
        dim=-1,
    )
    err = (fac * pred["radii"] - radii_gt[:, fac // 2 :: fac, fac // 2 :: fac]).abs()
    return err.view(b, hp * wp)


#### INTRINSICS ####


def intrinsics_gaussian_nll(pred: dict, data: dict) -> Tensor:
    """Negative log-likelihood loss for Gaussian distribution over intrinsics

    Args:
        pred: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_i) tensors with predicted intrinsics,
            - "intrinsics_icovs": list of (B, D_i, D_i) tensors with predicted inverse covariances.
        data: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_max) tensors with ground-truth intrinsics.
    """
    # nll = C - 0.5*log(det(Σ^{-1})) + 0.5*(x-μ)^T*Σ^{-1}*(x-μ)
    logdet_icovs = -torch.stack(
        [torch.linalg.slogdet(icov)[1] for icov in pred["intrinsics_icovs"]]
    )
    errs = [i - i_gt[: len(i)] for i, i_gt in zip(pred["intrinsics"], data["intrinsics"])]
    mh_dists = torch.stack([err @ icov @ err for err, icov in zip(errs, pred["intrinsics_icovs"])])
    return 0.5 * (logdet_icovs + mh_dists)


def intrinsics_squared_error(pred: dict, data: dict) -> Tensor:
    """Squared L2 norm of the error in intrinsics

    Args:
        pred: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_i) tensors with predicted intrinsics.
        data: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_max) tensors with ground-truth intrinsics.
    """
    return torch.stack(
        [
            (i - i_gt[: len(i)]).square().sum()
            for i, i_gt in zip(pred["intrinsics"], data["intrinsics"])
        ]
    )


def intrinsics_absolute_error(pred: dict, data: dict) -> Tensor:
    """L1 norm of the error in intrinsics

    Args:
        pred: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_i) tensors with predicted intrinsics.
        data: dict with at least the following key-value pair:
            - "intrinsics": list of (B, D_max) tensors with ground-truth intrinsics.

    Returns:
        (B,) tensor with the L1 norm of the error in intrinsics.
    """
    s = max(data["image"].shape[-2:])  # max side
    intrins_loss = []
    for cam_id, intrins, intrins_gt in zip(data["cam_id"], pred["intrinsics"], data["intrinsics"]):
        cam = CameraFactory.create_from_id(cam_id)
        d = cam.params_to_dict(intrins)
        d_gt = cam.params_to_dict(intrins_gt[: len(intrins)])
        # focal and principal point
        f_loss = torch.abs(d["f"] - d_gt["f"]).sum() / s
        c_loss = torch.abs(d["c"] - d_gt["c"]).sum() / s
        # distortion (if any -> will be 0--sum of empty tensor--if no distortion)
        k_loss = torch.abs(d["k"] - d_gt["k"]).sum()
        intrins_loss.append(f_loss + c_loss + k_loss)
    return torch.stack(intrins_loss)
    # return torch.stack(
    #     [
    #         (i - i_gt[: len(i)]).abs().sum()
    #         for i, i_gt in zip(pred["intrinsics"], data["intrinsics"])
    #     ]
    # )
