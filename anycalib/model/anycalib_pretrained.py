import os
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor

from anycalib.cameras import CameraFactory
from anycalib.cameras.base import BaseCamera
from anycalib.model.dinov2 import DINOv2
from anycalib.model.dpt_light_decoder import LightDPTDecoder
from anycalib.model.ray_decoder import ConvexTangentDecoder
from anycalib.optim import GaussNewtonCalib, LevMarCalib
from anycalib.ransac import RANSAC


def get_cam_list(data: dict) -> list[BaseCamera]:
    return [CameraFactory.create_from_id(id_) for id_ in data["cam_id"]]


def subsample(total: int, h: int, w: int, *tensors):
    """Subsampling along the (flattened) spatial dimensions.

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
    subsampled = [
        t[..., idx, :].contiguous() if isinstance(t, Tensor) else t for t in tensors
    ]
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
    subsampled = [
        t[..., idx, :].contiguous() if isinstance(t, Tensor) else t for t in tensors
    ]
    return subsampled


class Calibrator:

    def __init__(
        self,
        nonlin_opt_method: str = "gauss_newton",
        nonlin_opt_conf: dict | None = None,
        init_with_sac: bool = False,  # use RANSAC instead of nonminimal fit for init
        fallback_to_sac: bool = False,
        ransac_conf: dict | None = None,
        rm_borders: int = 0,  # border size to ignore during fitting
        sample_size: int = -1,  # negative -> no subsampling)
    ):
        # subsampling
        self.rm_borders = rm_borders
        self.sample_size = sample_size
        assert self.sample_size != 0, "Sample size must be non-zero"
        if self.sample_size > 0:
            raise NotImplementedError("Subsampling not implemented yet")
        # initialization/fallback via RANSAC
        self.init_with_sac = init_with_sac
        self.fallback_to_sac = fallback_to_sac
        self.ransac = RANSAC(ransac_conf)
        # nonlinear refinement
        if nonlin_opt_method == "gauss_newton":
            self.optimizer = GaussNewtonCalib(nonlin_opt_conf)
        elif nonlin_opt_method == "lev_mar":
            self.optimizer = LevMarCalib(nonlin_opt_conf)
        else:
            raise ValueError(
                "The nonlinear optimizer must be either 'gauss_newton' or 'lev_mar'. "
                f"However, got: {nonlin_opt_method}"
            )

    def __call__(self, pred: dict, data: dict) -> dict:
        optimizer = self.optimizer
        cams = get_cam_list(data)
        _, _, h, w = data["image"].shape
        rays: Tensor = pred["rays"]
        # image coords corresponding to rays. Add 0.5 to get coords at pixel *centers*
        im_coords = cams[0].pixel_grid_coords(h, w, rays, 0.5).view(h * w, 2)
        # observations for nonlinear optimization
        obs = pred["tangent_coords"] if optimizer.res_tangent == "z1" else rays

        # remove borders and subsample
        if self.rm_borders > 0:
            rays: Tensor
            obs: Tensor | list[None]
            im_coords: Tensor
            rays, obs, im_coords = remove_borders(
                h, w, self.rm_borders, rays, obs, im_coords
            )
        if self.sample_size > 0:
            rays: Tensor
            obs: Tensor | list[None]
            im_coords: Tensor
            rays, obs, im_coords = subsample(
                self.sample_size, h, w, rays, obs, im_coords
            )

        # control optimization of principal point
        cxcy = data.get("cxcy", None)
        fix_cxcy = cxcy is not None
        cxcy = [None] * len(cams) if cxcy is None else cxcy

        intrinsics, success, intrinsics_icovs = [], [], []
        # iterate over batch since cams may be of different models
        for rays_, cam_, cxcy_, obs_ in zip(rays, cams, cxcy, obs):
            success_ = rays_.new_ones((), dtype=torch.bool)
            # initialization
            if self.init_with_sac:
                intrinsics_, _ = self.ransac(cam_, im_coords, rays_)
            else:
                intrinsics_, info = cam_.fit(im_coords, rays_, cxcy_)  # (D,)
                success_ = (info == 0) and intrinsics_.isfinite().all()
                if not success_:
                    print(f"WARNING: Linear fit failed, {info=}")
                    if self.fallback_to_sac:
                        intrinsics_, _ = self.ransac(cam_, im_coords, rays_)
                    else:
                        intrinsics.append(torch.ones_like(intrinsics_))
                        success.append(success_)
                        continue

            # nonlinear refinement
            intrinsics_opt, cost0, cost, intrins_icovs_ = optimizer(
                cam_, intrinsics_, im_coords, obs_, None, fix_cxcy
            )
            success_ = success_ and cost < cost0
            intrinsics_icovs.append(intrins_icovs_)
            if cost > cost0:
                print(
                    f"WARNING: Worse cost after optimization: {cost:.2e} > {cost0:.2e}"
                )
            else:
                intrinsics_ = intrinsics_opt

            intrinsics.append(intrinsics_)
            success.append(success_)

        out = {"intrinsics": intrinsics, "success": torch.stack(success)}
        return (
            out if optimizer is None else out | {"intrinsics_icovs": intrinsics_icovs}
        )


class AnyCalib(torch.nn.Module):
    """AnyCalib class.

    Args for instantiation:
        model_id: one of {'anycalib_pinhole', 'anycalib_gen', 'anycalib_dist', 'anycalib_edit'}.
            Each model differes in the type of images they seen during training:
                * 'anycalib_pinhole': Perspective (pinhole) images,
                * 'anycalib_gen': General images, including perspective, distorted and
                    strongly distorted images, and
                * 'anycalib_dist': Distorted images using the Brown-Conrady camera model
                    and strongly distorted images, using the EUCM camera model,
                * 'anycalib_edit': Trained on edited (stretched and cropped) perspective
                    images.
            Default: 'anycalib_pinhole'.
        nonlin_opt_method: nonlinear optimization method: 'gauss_newton' or 'lev_mar'.
            Default: 'gauss_newton'
        nonlin_opt_conf: nonlinear optimization configuration.
            This config can be used to control the number of iterations and the space
            where the residuals are minimized. See the classes `GaussNewtonCalib` or
            `LevMarCalib` under anycalib/optim for details. Default: None.
        init_with_sac: use RANSAC instead of nonminimal fit for initializating the
            intrinsics. Default: False.
        fallback_to_sac: use RANSAC if nonminimal fit fails. Default: True.
        ransac_conf: RANSAC configuration. This config can be used to control e.g. the
            inlier threshold or the number of minimal samples to try. See the class
            `RANSAC` in anycalib/ransac.py for details. Default: None.
        rm_borders: border size of the dense FoV fields to ignore during fitting.
            Default: 0.
        sample_size: approximate number of 2D-3D correspondences to use for fitting the
            intrinsics. Negative value -> no subsampling. Default: -1.
    """

    EDGE_DIVISIBLE_BY = 14
    AR_RANGE = (0.5, 2)  # H/W range seen during training
    RESOLUTION = 102_400  # resolution seen during training

    AVAILABLE_MODELS = {
        "anycalib_pinhole",
        "anycalib_dist",
        "anycalib_gen",
        "anycalib_edit",
    }

    def __init__(
        self,
        model_id: str = "anycalib_pinhole",
        nonlin_opt_method: str = "gauss_newton",
        nonlin_opt_conf: dict | None = None,
        init_with_sac: bool = False,
        fallback_to_sac: bool = True,
        ransac_conf: dict | None = None,
        rm_borders: int = 0,
        sample_size: int = -1,
    ):
        super().__init__()
        if model_id not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model id: {model_id=}. Available models:\n"
                "\n".join(self.AVAILABLE_MODELS)
            )

        self.backbone = DINOv2(model_name="dinov2_vitl14")
        self.decoder = LightDPTDecoder(embed_dim=self.backbone.embed_dim)
        self.head = ConvexTangentDecoder(in_channels=self.decoder.out_channels)
        self.calibrator = Calibrator(
            nonlin_opt_method=nonlin_opt_method,
            nonlin_opt_conf=nonlin_opt_conf,
            init_with_sac=init_with_sac,
            fallback_to_sac=fallback_to_sac,
            ransac_conf=ransac_conf,
            rm_borders=rm_borders,
            sample_size=sample_size,
        )

        # load pretrained weights
        url = (
            f"https://github.com/javrtg/AnyCalib/releases/download/v1.0.0/{model_id}.pt"
        )
        model_dir = f"{torch.hub.get_dir()}/anycalib"
        state_dict = torch.hub.load_state_dict_from_url(
            url, model_dir, map_location="cpu", file_name=f"{model_id}.pt"
        )
        self.load_state_dict(state_dict, strict=True)

    def forward(self, data):
        # get ray and FoV fields
        out = self.backbone(data["image"])
        out: dict[str, Tensor] = self.head(self.decoder(out))
        # reshape to (B, H*W, {3, 2})
        b, _, h, w = data["image"].shape
        out["rays"] = out["rays"].permute(0, 2, 3, 1).view(b, h * w, 3)
        out["tangent_coords"] = out["fov_field"] = (
            out["tangent_coords"].permute(0, 2, 3, 1).view(b, h * w, 2)
        )
        out |= self.calibrator(out, data)
        return out

    @torch.inference_mode()
    def predict(self, im: Tensor, cam_id: str | list[str]) -> dict:
        """Single-view camera calibration

        Args:
            im: (B, 3, H, W) or (3, H, W) input image with RGB values in [0, 1].
            cam_id: string containing the camera id or list of string cam ids. If a
                string, the same camera id is used for all images in the batch.
        """
        non_batched = im.dim() == 3
        if non_batched:
            im = im.unsqueeze(0)
        if isinstance(cam_id, str):
            cam_id = [cam_id] * im.shape[0]
        assert len(cam_id) == im.shape[0], f"{len(cam_id)=} != {im.shape[0]=}"

        ho, wo = im.shape[-2:]
        target_ar = max(self.AR_RANGE[0], min(ho / wo, self.AR_RANGE[1]))
        target_size = self.compute_target_size(self.RESOLUTION, target_ar)

        im, scale_xy, shift_xy = self.set_im_size(im, target_size)
        pred = self.forward({"image": im, "cam_id": cam_id})

        # based on the initial resize, correct focal length and principal point
        for i, (intrins, cam_id) in enumerate(zip(pred["intrinsics"], cam_id)):
            cam = CameraFactory.create_from_id(cam_id)
            pred["intrinsics"][i] = cam.reverse_scale_and_shift(
                intrins, scale_xy, shift_xy
            )
        if non_batched:
            pred = {k: v[0] for k, v in pred.items()}
        pred |= {"pred_size": target_size}
        return pred

    def compute_target_size(
        self, target_res: float, target_ar: float
    ) -> tuple[int, int]:
        """Compute the target image size given the target resolution and aspect ratio."""
        w = sqrt(target_res / target_ar)
        h = target_ar * w
        # closest image size satisfying `edge_divisible_by` constraint
        div = self.EDGE_DIVISIBLE_BY
        target_size = (round(h / div) * div, round(w / div) * div)
        return target_size

    @staticmethod
    def set_im_size(
        im: Tensor, target_size: tuple[int, int]
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transform an image to the target size by center cropping and downscaling.

        This function also returns the scales and offsets needed to update the intrinsics
        corresponding to the "digitizing process" (focals fx, fy and pral. points cx, cy),
        to account for the cropping and scaling(s). Since this function does the following:
            1) (optional) upsampling with scale s1,
            2) center cropping of [shift_x, shift_y] pixels from the right or top of image,
            to achieve the target aspect ratio, and
            3) downsampling with scales [s2_x, s2_y] to the target resolution.
        Then to update:
            a) the focals (fx, fy): we need to multiply by s1 * [s2_x, s2_y],
            b) the principal point (cx, cy): scale also by s1 * [s2_x, s2_y], followed by
            shift of -[s2_x, s2_y]*[shift_x, shift_y] pixels.

        Args:
            im: (B, 3, H, W) input image with RGB values in [0, 1].
            target_size: Integer 2-tuple with target resolution (height, width).

        Returns:
            im_transformed: (B, 3, *target_size) Transformed image(s).
            scale_xy: (2,) Scales for updating the intrinsics (focals and principal point).
            shift_xy: (2,) Shifts for updating the principal point.
        """
        assert im.dim() == 4, f"Expected 4D tensor, got {im.dim()} with {im.shape=}"
        if im.shape[-2:] == target_size:
            # no need to resize
            return im, torch.ones(2, device=im.device), torch.zeros(2, device=im.device)

        h, w = im.shape[-2:]
        ht, wt = target_size

        # upsample preserving the aspect ratio so that no side is shorter than the targets
        if h < ht or w < wt:
            scale_1 = max(ht / h, wt / w)
            im = F.interpolate(
                im,
                scale_factor=scale_1,
                mode="bicubic",
                align_corners=False,
            ).clamp(0, 1)
            # update
            h_, w_ = im.shape[-2:]
            scale_1_xy = torch.tensor((w_ / w, h_ / h), device=im.device)
            h, w = h_, w_
        else:
            scale_1_xy = 1.0  # no upsampling

        # center crop from one side (either width or height) to achieve the target aspect ratio
        shift_xy = torch.zeros(2, device=im.device)
        ar_t = wt / ht
        if w / h > ar_t:
            # crop (negative pad) width, otherwise we would need to pad the height
            crop_w = round(w - h * ar_t)
            im = im[..., crop_w // 2 : w - crop_w + crop_w // 2]
            shift_xy[0] = -(
                crop_w // 2
            )  # NOTE: careful: -(crop_w // 2) != -crop_w // 2
        else:
            # crop height
            crop_h = round(h - w / ar_t)
            im = im[..., crop_h // 2 : h - crop_h + crop_h // 2, :]
            shift_xy[1] = -(crop_h // 2)
        h, w = im.shape[-2:]

        # downsample to the target resolution
        im = F.interpolate(
            im, target_size, mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        scale_2_xy = torch.tensor((wt / w, ht / h), device=im.device)
        # for updating the intrinsics
        scale_xy = scale_1_xy * scale_2_xy
        shift_xy = shift_xy * scale_2_xy
        return im, scale_xy, shift_xy

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
            print(f"WARNING: Missing {len(diff)} parameters in {subs}: {diff}")
        self.load_state_dict(state_dict, strict=False)
        return self
