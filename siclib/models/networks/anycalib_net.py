import logging
import os
from collections import defaultdict
from collections.abc import Callable
from math import pi, sqrt
from typing import NamedTuple

import torch
from omegaconf import OmegaConf
from torch import Tensor

from anycalib.cameras import CameraFactory
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3
from anycalib.optim import GaussNewtonCalib, LevMarCalib
from anycalib.ransac import RANSAC
from siclib.models.base_model import BaseModel
from siclib.models.decoders.dpt_decoder import DPTDecoder
from siclib.models.decoders.dpt_light_decoder import LightDPTDecoder
from siclib.models.decoders.ray_decoder import (
    ConvexRayDecoder,
    ConvexTangentDecoder,
    ConvexTangentEditDecoder,
    RayDecoder,
    TangentDecoder,
)
from siclib.models.encoders.dinov2 import DINOv2
from siclib.models.utils import losses_rays

logger = logging.getLogger(__name__)

Encoder = DINOv2
Decoder = LightDPTDecoder | DPTDecoder
Head = (
    ConvexRayDecoder | RayDecoder | ConvexTangentDecoder | TangentDecoder | ConvexTangentEditDecoder
)
Optimizer = GaussNewtonCalib | LevMarCalib

ENCODERS = {
    "dinov2": DINOv2,
}
DECODERS = {
    "light_dpt_ray_decoder": (LightDPTDecoder, ConvexRayDecoder),
    "light_dpt_tangent_decoder": (LightDPTDecoder, ConvexTangentDecoder),
    "light_dpt_tangent_edit_decoder": (LightDPTDecoder, ConvexTangentEditDecoder),
    "dpt_ray_decoder": (DPTDecoder, RayDecoder),
    "dpt_tangent_decoder": (DPTDecoder, TangentDecoder),
}

RAD2DEG = 180 / pi
DEG2RAD = pi / 180


def get_cam_list(data: dict) -> list[BaseCamera]:
    return [CameraFactory.create_from_id(id_) for id_ in data["cam_id"]]


def subsample(total: int, h: int, w: int, *tensors):
    """Structured subsampling along the (flattened) spatial dimensions.

    NOTE: the first element within `tensors` argument is assumed to be a torch.Tensor.

    Args:
        total: Approximate number of elements to subsample.
        h: Height of the unflattened tensor.
        w: Width of the unflattened tensor.
        tensors: (B, H*W, D) Tensors to subsample.

    Returns:
        (B, ~total, D) Tensors subsampled along the spatial dimensions.
    """
    assert isinstance(tensors[0], Tensor)
    dev = tensors[0].device
    step = sqrt(h * w / min(total, h * w))
    x = torch.arange(0.5 * step, w, step, device=dev).long()
    y = torch.arange(0.5 * step, h, step, device=dev).long()
    x, y = torch.meshgrid(x, y, indexing="xy")
    idx = y.flatten() * w + x.flatten()  # flattened indexes
    subsampled = [t[..., idx, :].contiguous() if isinstance(t, Tensor) else t for t in tensors]
    return subsampled


def remove_borders(h: int, w: int, border: int, *tensors):
    """Remove border pixels from the spatial dimensions.

    NOTE: the first element within `tensors` argument is assumed to be a torch.Tensor.

    Args:
        h: Height of the unflattened tensor.
        w: Width of the unflattened tensor.
        border: Number of pixels to remove from each border.
        tensors: (B, H*W, D) Tensors to subsample or list[None].

    Returns:
        (B, (H-2*border)*(W-2*border), D) Tensors with border pixels removed.
    """
    dev = tensors[0].device
    x = torch.arange(border, w - border, device=dev)
    y = torch.arange(border, h - border, device=dev)
    x, y = torch.meshgrid(x, y, indexing="xy")
    idx = y.flatten() * w + x.flatten()  # flattened indexes
    subsampled = [t[..., idx, :].contiguous() if isinstance(t, Tensor) else t for t in tensors]
    return subsampled


class Calibrator:

    default_conf = {
        # subsampling
        "rm_borders": 0,  # border size to ignore during fitting
        "sample_size": -1,  # negative -> no subsampling
        "cov_guided_sampling": False,
        # detach before non-lin optimization
        "detach_rays": False,
        "detach_lin_fit": True,
        # initialization via linear fit
        "lin_with_covs": False,
        # initialization/fallback via RANSAC
        "init_with_sac": False,
        "fallback_to_sac": False,
        "ransac_conf": RANSAC.DEFAULT_CONF,
        # nonlinear refinement
        "nonlin_opt": {
            "name": "gauss_newton",
            "use_covs": False,
            "conf": GaussNewtonCalib.DEFAULT_CONF,
        },
        # training loss
        "loss": {
            "name": None,
            # "name": "rays-l1",
            # "name": "rays-l2",
            # "name": "rays-nll-gaussian",
            # "name": "rays-nll-gaussian-opt",
            # "name": "intrinsics-l1",
            # "name": "intrinsics-l2",
            "weight": 1.0,
        },
    }

    AVAILABLE_OPTIMIZERS = {
        "gauss_newton": GaussNewtonCalib,
        "lev_mar": LevMarCalib,
    }

    def __init__(self, conf):
        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        self.conf = conf = OmegaConf.merge(default_conf, conf)

        # subsampling
        self.rm_borders = conf.rm_borders
        self.sample_size = conf.sample_size
        assert self.sample_size != 0, "Sample size must be non-zero"
        if self.sample_size > 0:
            raise NotImplementedError("Subsampling not implemented yet")
        self.cov_guided_sampling = conf.cov_guided_sampling

        # initialization via linear fit
        if conf.lin_with_covs:
            raise NotImplementedError("Linear fit with covariances not implemented yet")
        self.lin_with_covs = conf.lin_with_covs  # whether to use covs in linear fit
        # initialization/fallback via RANSAC
        self.init_with_sac = conf.init_with_sac  # use RANSAC instead of linear fit
        self.fallback_to_sac = conf.fallback_to_sac
        self.ransac = RANSAC(OmegaConf.to_container(conf.ransac_conf))  # type: ignore

        # nonlinear refinement
        if conf.nonlin_opt is not None:
            self.optimizer = self.AVAILABLE_OPTIMIZERS[conf.nonlin_opt.name](
                OmegaConf.to_container(conf.nonlin_opt.conf)  # type: ignore
            )
            self.nonlin_opt_w_covs = conf.nonlin_opt.use_covs
        else:
            self.optimizer: Optimizer | None = None
            self.nonlin_opt_w_covs = False

        # training loss
        self.loss_name = conf.loss.name
        self.loss_fn: Callable[..., Tensor] = {
            "rays-l1": losses_rays.absolute_tangent_error,
            "rays-l2": losses_rays.squared_tangent_error,
            "rays-angle": losses_rays.angular_error,
            # NOTE: next, tangent=pred since covariances are defined in the tangent plane of the bearings
            # that are fitted during nonlinear optimization (they are the estimated mean of the distribution)
            # "rays-nll-gaussian": partial(  # reference (supervision) bearings: ground-truth
            #     losses_rays.gaussian_nll, tangent="data"
            # ),
            # "rays-nll-gaussian-opt": partial(  # reference (supervision) bearings: net predictions
            #     losses_rays.gaussian_nll, tangent="pred"
            # ),
            # "rays-nll-gaussian-z1": losses_rays.gaussian_nll_at_z1,
            ##
            # intrinsics
            ##
            "intrinsics-l1": losses_rays.intrinsics_absolute_error,
            "intrinsics-l2": losses_rays.intrinsics_squared_error,
            "intrinsics-nll-gaussian": losses_rays.intrinsics_gaussian_nll,
            None: None,
        }[conf.loss.name]
        if conf.loss.name is not None:
            self.calib_loss = (
                self.intrinsics_loss if "intrinsics" in conf.loss.name else self.rays_loss
            )
        else:
            self.calib_loss: Callable[[dict, dict], Tensor] | None = None
        self.loss_weight = conf.loss.get("weight", 1.0)
        self.loss_str = f"calib-{conf.loss.name}-loss"

        self.detach_lin_fit = conf.detach_lin_fit
        self.detach_rays = conf.detach_rays

    def __call__(self, pred: dict, data: dict) -> dict:
        optimizer = self.optimizer
        cams = get_cam_list(data)
        _, _, h, w = data["image"].shape

        rays: Tensor = pred["rays"].detach() if self.detach_rays else pred["rays"]
        nones = [None] * len(rays)
        icovs = torch.exp(-pred["log_covs"]) if self.nonlin_opt_w_covs else nones
        covs = torch.exp(pred["log_covs"]) if self.lin_with_covs else nones
        # image coords corresponding to rays. Add 0.5 to get coords at pixel *centers*
        im_coords = cams[0].pixel_grid_coords(h, w, rays, 0.5).view(h * w, 2)
        # observations for nonlinear optimization
        if optimizer is None:
            obs = nones
        elif optimizer.res_tangent == "z1":
            obs = pred["tangent_coords"]
        else:
            obs = rays

        # subsample
        if self.rm_borders > 0:
            rays: Tensor
            icovs: Tensor | list[None]
            covs: Tensor | list[None]
            obs: Tensor | list[None]
            im_coords: Tensor
            rays, icovs, covs, obs, im_coords = remove_borders(
                h, w, self.rm_borders, rays, icovs, covs, obs, im_coords
            )

        # control optimization of principal point
        cxcy = data.get("cxcy", None)
        fix_cxcy = cxcy is not None
        cxcy = [None] * len(cams) if cxcy is None else cxcy

        intrinsics, success, intrinsics_icovs = [], [], []
        # iterate over batch since cams may be of different models
        for rays_, icovs_, covs_, cam_, cxcy_, obs_ in zip(rays, icovs, covs, cams, cxcy, obs):
            success_ = rays_.new_ones((), dtype=torch.bool)
            # initialization
            if self.init_with_sac:
                with torch.no_grad():
                    intrinsics_, _ = self.ransac(cam_, im_coords, rays_)
            else:
                # linear fit
                with torch.set_grad_enabled(not self.detach_lin_fit and torch.is_grad_enabled()):
                    intrinsics_, info = cam_.fit(im_coords, rays_, cxcy_, covs_)  # (D,)
                success_ = (info == 0) and not intrinsics_.isnan().any()
                if not success_:
                    logger.warning(f"Linear fit failed, {info=}")
                    if self.fallback_to_sac:
                        with torch.no_grad():
                            intrinsics_, _ = self.ransac(cam_, im_coords, rays_)
                    else:
                        intrinsics.append(torch.ones_like(intrinsics_))
                        success.append(success_)
                        continue

            if optimizer is not None:
                # nonlinear refinement
                intrinsics_opt, cost0, cost, intrins_icovs_ = optimizer(
                    cam_, intrinsics_, im_coords, obs_, icovs_, fix_cxcy
                )
                success_ = success_ and cost < cost0
                intrinsics_icovs.append(intrins_icovs_)
                if cost >= cost0:
                    logger.warning(f"Worse cost after optimization: {cost:.2e} >= {cost0:.2e}")
                else:
                    intrinsics_ = intrinsics_opt

            intrinsics.append(intrinsics_)
            success.append(success_)
        out = {"intrinsics": intrinsics, "success": torch.stack(success)}
        return out if optimizer is None else out | {"intrinsics_icovs": intrinsics_icovs}

    def rays_loss(self, pred: dict, data: dict) -> Tensor:
        assert self.loss_fn is not None, "Loss function not defined"
        b, _, h, w = data["image"].shape
        loss_fn = self.loss_fn
        # data
        cams = get_cam_list(data)
        im_coords = cams[0].pixel_grid_coords(h, w, data["rays"], 0.5).view(h * w, 2)

        n_valid = 0
        loss_batch = []
        net_rays_as_ref = self.loss_name == "rays-nll-gaussian-opt"
        # iterate over batch since cams may be of different models
        for i in range(b):
            if not pred["success"][i]:
                loss_batch.append(data["rays"][i].new_zeros(()))
                continue

            calib_pred = {}
            calib_pred["rays"], valid = cams[i].unproject(data["intrinsics"][i], im_coords)
            mask = data["rays_mask"][i] if valid is None else data["rays_mask"][i] & valid

            if "z1" in self.loss_name:
                calib_pred["tangent_coords"] = Unit3.logmap_at_z1(calib_pred["rays"])

            if "nll" in self.loss_name:
                # intrinsics covariance
                calib_pred["intrinsics_covs"], info = torch.linalg.inv_ex(
                    data["intrinsics_icovs"][i]
                )
                if info != 0:
                    loss_batch.append(data["rays"][i].new_zeros(()))
                    continue
                calib_pred |= {
                    "cam": cams[i],
                    "intrinsics": data["intrinsics"][i],
                    "im_coords": im_coords,
                }

            n_valid = n_valid + mask.sum()

            ref_ = {"rays": pred["rays"][i]} if net_rays_as_ref else {"rays": data["rays"][i]}
            loss_batch.append((loss_fn(calib_pred, ref_) * mask).sum())

        # multiply by b so that when the train script computes the mean across the batch
        # it is correctly calculated as 1/N_valid Σ_i loss_i
        return b / max(n_valid, 1) * torch.stack(loss_batch)  # type: ignore

    # def rays_loss(self, pred: dict, data: dict) -> Tensor:
    #     assert self.loss_fn is not None, "Loss function not defined"
    #     b, _, h, w = data["image"].shape
    #     loss_fn = self.loss_fn
    #     # data
    #     cams = get_cam_list(data)
    #     gt_rays: Tensor = data["rays"]  # (B, H*W, 3)
    #     gt_rays_mask: Tensor = data["rays_mask"]  # (B, H*W)
    #     # preds
    #     net_rays: Tensor = pred["rays"]
    #     success: Tensor = pred["success"]
    #     intrinsics: list[Tensor] = pred["intrinsics"]
    #     # image coords corresponding to rays. Add 0.5 to get coords at pixel *centers*
    #     im_coords = cams[0].pixel_grid_coords(h, w, gt_rays, 0.5).view(h * w, 2)

    #     n_valid = 0
    #     loss_batch = []
    #     pred_ = {"log_covs": pred["log_covs"]} if "log_covs" in pred else {}
    #     net_rays_as_ref = self.loss_name == "rays-nll-gaussian-opt"
    #     for cam, intrins, success_, gt_rays_, gt_mask_, net_rays_ in zip(
    #         cams, intrinsics, success, gt_rays, gt_rays_mask, net_rays
    #     ):
    #         if not success_:
    #             loss_batch.append(gt_rays_.new_zeros(()))
    #             continue

    #         pred_rays_, valid = cam.unproject(intrins, im_coords)
    #         if "z1" in self.loss_name:
    #             pred_["tangent_coords"] = Unit3.logmap_at_z1(pred_rays_)
    #             # if "fit" in self.loss_name:
    #             #     # we need the covariances of the intrinsics
    #             #     intrins_covs, valid = torch.linalg.inv_ex(intrins_icovs_)

    #         mask = gt_mask_ if valid is None else gt_mask_ & valid
    #         n_valid = n_valid + mask.sum()

    #         pred_["rays"] = pred_rays_
    #         ref_ = {"rays": net_rays_} if net_rays_as_ref else {"rays": gt_rays_}
    #         loss_batch.append((loss_fn(pred_, ref_) * mask).sum())

    #     # multiply by b so that when the train script computes the mean across the batch
    #     # it is correctly calculated as 1/N_valid Σ_i loss_i
    #     return b / max(n_valid, 1) * torch.stack(loss_batch)  # type: ignore

    def intrinsics_loss(self, pred: dict, data: dict) -> Tensor:
        """loss between ground-truth and predicted intrinsics.

        This method returns a (B,) Tensor with the sum of losses computed with `loss_fn`
        at each datapoint in the batch. Each element is multiplied by B/N, where N is the
        *total* (across batch) number of valid observations. This is done to comply with
        the main train script that computes the loss as the mean of this output. This way
        the mean is correctly calculated as:
            loss = 1/B Σ_i loss_datapoint_i = 1/B Σ_i B/N loss_i
                    = 1/N Σ_i loss_sum_i

        Args:
            pred: Dict with predictions.
            data: Dict with ground-truth data.
            fn: Loss function.

        Returns:
            (B,) Tensor with the sum of losses at each datapoint in the batch.
        """
        assert self.loss_fn is not None, "Loss function not defined"
        success = pred["success"]
        loss = (len(success) / max(success.sum(), 1)) * self.loss_fn(pred, data) * success  # (B,)
        if not torch.isfinite(loss).all():
            logger.warning("Non-finite intrinsics loss encountered.")
            loss = torch.zeros_like(loss)
        return loss


Loss = NamedTuple("Loss", [("fn", Callable), ("weight", float), ("name", str)])


class AnyCalib(BaseModel):

    default_conf = {
        "backbone": {
            "name": "dinov2",
            "conf": {
                "model_name": "dinov2_vitl14",
                "num_trainable_blocks": -1,  # -1 -> all blocks trainable
                "intermediate_layers": None,  # None -> default DPT's intermediate layers
            },
        },
        "decoder": {
            "name": "light_dpt_tangent_decoder",
            "conf": {"dim_dhat": 256, "post_process_channels": None},
            "conf_head": {"predict_covs": False, "use_tanh": False, "logvar_lims": (-20, 10)},
        },
        "calibrator": {
            # subsampling
            "rm_borders": 0,  # border size to ignore during fitting
            "lin_with_covs": False,
            "sample_size": -1,  # negative -> no subsampling
            # detach before non-lin optimization
            "detach_rays": False,
            "detach_lin_fit": True,
            # initialization via linear fit
            "cov_guided_sampling": False,
            # initialization/fallback via RANSAC
            "init_with_sac": False,
            "fallback_to_sac": False,
            "ransac_conf": RANSAC.DEFAULT_CONF,
            # nonlinear refinement
            "nonlin_opt": {
                "name": "gauss_newton",
                "use_covs": False,
                "conf": GaussNewtonCalib.DEFAULT_CONF,
            },
            # training loss
            "loss": {
                "name": None,
                "weight": 1.0,
            },
        },
        "loss": {"names": ["l1-z1"], "weights": [1.0]},
        "recall_thresholds": [1, 5, 10],
    }
    required_data_keys = ["image"]

    def _init(self, conf):
        logger.debug(f"Initializing AnyCalib with {conf}")

        self.backbone: Encoder = ENCODERS[conf.backbone.name](**conf.backbone.conf)
        decoder_cls, head_cls = DECODERS[conf.decoder.name]
        self.decoder: Decoder = decoder_cls(embed_dim=self.backbone.embed_dim, **conf.decoder.conf)
        self.head: Head = head_cls(in_channels=self.decoder.out_channels, **conf.decoder.conf_head)
        self.calibrator = Calibrator(OmegaConf.to_container(conf.calibrator))

        # check if covariances are needed but not predicted
        loss_with_covs = any(["nll" in loss for loss in conf.loss.names])
        if not self.head.predict_covs and any(
            (self.calibrator.lin_with_covs, self.calibrator.nonlin_opt_w_covs, loss_with_covs)
        ):
            nonlin_opt_w_covs = self.calibrator.nonlin_opt_w_covs
            lin_with_covs = self.calibrator.lin_with_covs
            raise ValueError(
                "Head does not predict covariances but are needed for either: \n"
                f"\t- linear fit: {lin_with_covs}\n\t- nonlin opt.: {nonlin_opt_w_covs}\n"
                f"\t- loss: {loss_with_covs}"
            )

        # losses
        assert len(conf.loss.names) == len(conf.loss.weights), "len(loss_names) != len(weights)"
        self.losses = []
        for loss_name, loss_weight in zip(conf.loss.names, conf.loss.weights):
            self.losses.append(
                Loss(
                    fn={
                        "nll-gaussian": losses_rays.gaussian_nll,
                        "nll-gaussian-z1": losses_rays.gaussian_nll_at_z1,
                        "nll-laplace-z1": losses_rays.laplace_nll_at_z1,
                        "mix-laplace-z1": losses_rays.mixture_laplace_at_z1,
                        "l1": losses_rays.absolute_tangent_error,
                        "l1-z1": losses_rays.absolute_tangent_error_at_z1,
                        "l1-z1-r": losses_rays.absolute_tangent_error_at_z1_radial,
                        "l2": losses_rays.squared_tangent_error,
                        "l2-z1": losses_rays.squared_tangent_error_at_z1,
                        "angle": losses_rays.angular_error,
                        "cos": losses_rays.cosine_sim,
                        # edit maps
                        "laplace-ar": losses_rays.laplace_nll_aspect_ratio,
                        "l1-r": losses_rays.absolute_radii_error,
                    }[loss_name],
                    weight=loss_weight,
                    name=f"rays-{loss_name}-loss",
                )
            )

        # check if calibration rays loss is needed
        self.with_calib_loss = self.calibrator.calib_loss is not None

    def _forward(self, data):
        out = self.backbone(data["image"])
        # decode to rays and log_covs
        out: dict[str, Tensor] = self.head(self.decoder(out))

        # reshape rays and covs to (B, H*W, {3, 2})
        b, _, h, w = data["image"].shape
        out["rays"] = out["rays"].permute(0, 2, 3, 1).view(b, h * w, 3)
        if "log_covs" in out:
            out["log_covs"] = out["log_covs"].permute(0, 2, 3, 1).view(b, h * w, 2)
        if "tangent_coords" in out:
            out["tangent_coords"] = out["tangent_coords"].permute(0, 2, 3, 1).view(b, h * w, 2)
        if "weights" in out:
            out["weights"] = out["weights"].permute(0, 2, 3, 1).view(b, -1, 2)

        if self.training and not self.with_calib_loss:  # skip calibration
            return out

        with torch.set_grad_enabled(self.with_calib_loss and torch.is_grad_enabled()):
            # NOTE: if gradients are enabled they will only be computed if out requires grad
            out |= self.calibrator(out, data)
        return out

    def loss(self, pred, data):
        assert len(self.losses) > 0 or self.calibrator.calib_loss is not None, "Loss not defined"

        losses = {}
        total = 0
        for loss in self.losses:
            ray_loss = self.ray_loss(pred, data, loss)
            losses[loss.name] = ray_loss
            total = total + loss.weight * ray_loss

        if self.calibrator.calib_loss is not None:
            calib_loss = self.calibrator.calib_loss(pred, data)
            # only apply calbrator loss when angular error is acceptable
            rays_gt: Tensor = data["rays"]  # (B, H*W, 3)
            rays_gt_mask: Tensor = data["rays_mask"]  # (B, H*W)
            rays: Tensor = pred["rays"]
            ok = (Unit3.distance(rays_gt, rays) * rays_gt_mask).mean(dim=1) < 5 * DEG2RAD
            calib_loss = torch.stack(
                [v if ok_ else v.new_zeros(()) for v, ok_ in zip(calib_loss, ok)]
            )
            if not ok.all():
                logger.warning(f"Calibration loss ignored for {(~ok).sum()} samples")

            losses[self.calibrator.loss_str] = calib_loss
            total = total + self.calibrator.loss_weight * calib_loss

        losses["total"] = total
        return losses, self.metrics(pred, data)

    @torch.no_grad()
    def metrics(self, pred, data):
        out = {}
        # gt
        rays_gt: Tensor = data["rays"]  # (B, H*W, 3)
        rays_gt_mask: Tensor = data["rays_mask"]  # (B, H*W)
        # pred
        rays: Tensor = pred["rays"]

        # mask out invalid observations to not contaminate the mean with outliers
        errors = Unit3.logmap(rays_gt, rays) * rays_gt_mask.unsqueeze(-1)  # (B, H*W, 2)

        # mahalanobis distance
        if "log_covs" in pred:
            out["maha_dist_error"] = (errors * torch.exp(-pred["log_covs"]) * errors).sum(2).mean(1)
        # angular errors in degrees
        ang_errors = torch.linalg.norm(errors, dim=2) * RAD2DEG
        out["angular_error"] = ang_errors.mean(dim=1)
        for th in self.conf.recall_thresholds:
            rec = (ang_errors < th).float().mean(dim=1)
            out[f"angular_error_recall@{th}"] = rec

        if "intrinsics" not in pred:
            return out

        # intrinsics errors
        h = data["image"].shape[2]
        cams = get_cam_list(data)
        vfov_errs = []
        # param_errs = {f"{cam.NAME}_{p}_error": [] for cam in cams for p in cam.PARAMS_IDX.keys()}
        param_errs = defaultdict(list)
        for p_gt, p, success_, cam in zip(
            data["intrinsics"], pred["intrinsics"], pred["success"], cams
        ):
            p_gt = p_gt[: len(p)]
            err = torch.abs(p - p_gt) * success_
            for param, err in zip(cam.PARAMS_IDX.keys(), err):
                param_errs[f"{cam.NAME}_{param}_error"].append(err)

            vfov_gt, _ = cam.get_vfov(p_gt, h)
            vfov, _ = cam.get_vfov(p, h)
            vfov_errs.append(RAD2DEG * torch.abs(vfov - vfov_gt))

        out |= {k: torch.stack(v) for k, v in param_errs.items()}
        out["vfov_error"] = torch.stack(vfov_errs)
        return out

    def ray_loss(self, pred: dict, data: dict, loss: Loss) -> Tensor:
        """Elementwise loss between ground-truth and predicted rays.

        This method returns a (B,) Tensor with the sum of losses computed with `loss_fn`
        at each datapoint in the batch. Each element is multiplied by B/N, where N is the
        *total* (across batch) number of valid observations. This is done to comply with
        the main train script that computes the loss as the mean of this output. This way
        the mean is correctly calculated as:
            loss = 1/B Σ_i loss_datapoint_i = 1/B Σ_i B/N loss_sum_i
                    = 1/N Σ_i loss_sum_i

        Args:
            pred: Dict with predictions.
            data: Dict with ground-truth data.
            fn: Loss function.

        Returns:
            (B,) Tensor with the sum of losses at each datapoint in the batch.
        """
        gt_rays_mask: Tensor = data["rays_mask"]
        if loss.name in ("rays-laplace-ar-loss", "rays-l1-r-loss"):  # no mask for these losses
            return loss.fn(pred, data).mean(dim=1)
        errors = loss.fn(pred, data) * gt_rays_mask  # (B, H*W)
        loss_ = (gt_rays_mask.shape[0] / gt_rays_mask.sum()) * errors.sum(dim=1)
        return loss_

    def load_weights_from_ckpt(self, ckpt_path: str) -> "AnyCalib":
        """Load model from training checkpoint."""
        assert ckpt_path.endswith(".tar")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"]

        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self.named_parameters()))
        diff = model_params - dict_params
        if len(diff) > 0:
            subs = os.path.commonprefix(list(diff)).rstrip(".")
            logger.warning(f"Missing {len(diff)} parameters in {subs}: {diff}")
        self.load_state_dict(state_dict, strict=False)
        return self


if __name__ == "__main__":
    import time

    from anycalib.visualization.viz_batch import make_batch_figures
    from siclib.datasets.simple_dataset_rays import SimpleDataset
    from siclib.utils.tensor import batch_to_device
    from siclib.utils.tools import fork_rng

    dconf = SimpleDataset.default_conf
    dconf["name"] = "simple_dataset_geom"
    dconf["dataset_dir"] = "data/openpano_v2/openpano_v2_radial"
    dconf["train_batch_size"] = 2  # 48, 24 for loss in calibrator
    dconf["preprocessing"]["edge_divisible_by"] = 14
    dconf["num_workers"] = 0
    dconf["prefetch_factor"] = None

    # torch.set_grad_enabled(False)
    dataset = SimpleDataset(dconf)
    loader = dataset.get_data_loader("train")

    model = AnyCalib(
        {
            "backbone": {
                "name": "dinov2",
                "conf": {"num_trainable_blocks": -1},
            },
            "decoder": {
                "name": "light_dpt_tangent_edit_decoder",
                "conf": {},
                "conf_head": {"predict_covs": False},
            },
            "calibrator": {
                "detach_rays": False,
                "loss": {
                    "name": "intrinsics-l1",
                    "weight": 1.0,
                },
            },
            "loss": {
                "names": ["l1-z1"],
                "weights": [1.0],
            },
            # "loss": {"names": ["l1-z1"], "weights": [1.0]},
            # "loss": [{"name": "nll-gaussian", "weight": 1.0}],
        },
    ).to("cuda")

    with fork_rng(seed=42):
        for data in loader:
            data = batch_to_device(data, "cuda")

            torch.cuda.synchronize()
            start_time = time.time()
            pred = model(data)
            # loss, metrics = model.loss(pred, data)
            torch.cuda.synchronize()
            end_time = time.time()
            print(f"Execution Time: {end_time - start_time:.4f} seconds")

            loss, metrics = model.loss(pred, data)
            torch.mean(loss["total"]).backward()
            # print(pred["log_covs"])
            break
        fig = make_batch_figures(pred, data, n_pairs=4)  # type: ignore

    fig["radial"].savefig("radial.png")
    fig["errors"].savefig("errors.png")
    if "editmaps" in fig:
        fig["editmaps"].savefig("editmaps.png")

    # random data
    # torch.manual_seed(42)
    # h, w = 336, 336
    # data = {
    #     "image": torch.rand(1, 3, h, w, device="cuda"),
    #     "rays": torch.rand(1, h * w, 3, device="cuda"),
    #     "rays_mask": torch.ones(1, h * w, device="cuda"),
    #     "intrinsics": torch.rand(1, 4, device="cuda"),
    #     "cam_id": ["pinhole"],
    # }
    # pred = model(data)
    # loss, metrics = model.loss(pred, data)

    # fig = make_batch_figures(pred, data, n_pairs=4)  # type: ignore

    # fig["radial"].savefig("radial.png")
    # fig["errors"].savefig("errors.png")
