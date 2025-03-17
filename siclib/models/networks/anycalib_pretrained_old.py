from math import sqrt

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from anycalib.cameras import CameraFactory
from siclib.models.networks.anycalib_net import AnyCalib


class AnyCalibPretrained(torch.nn.Module):

    EDGE_DIVISIBLE_BY = 14

    def __init__(
        self,
        ckpt_path: str,
        conf: dict | OmegaConf | None = None,
        ar_range: tuple[float, float] = (0.5, 2),  # H/W range
        resolution: int = 102_400,
    ):
        super().__init__()
        self.model = AnyCalib({} if conf is None else conf).load_weights_from_ckpt(ckpt_path).eval()
        self.ar_range = ar_range
        self.resolution = resolution

    @torch.inference_mode()
    def predict(self, im: Tensor, cam_id: str | list[str], cxcy: Tensor | None = None) -> dict:
        non_batched = im.dim() == 3
        if non_batched:
            im = im.unsqueeze(0)
        cam_id = [cam_id] if isinstance(cam_id, str) else cam_id
        assert len(cam_id) == im.shape[0], f"{len(cam_id)=} != {im.shape[0]=}"

        ho, wo = im.shape[-2:]
        target_ar = max(self.ar_range[0], min(ho / wo, self.ar_range[1]))
        target_size = self.compute_target_size(self.resolution, target_ar)

        im, scale_xy, shift_xy = self.set_im_size(im, target_size)
        # TODO: update cxcy (if given) with scale_xy, shift_xy
        pred = self.model({"image": im, "cam_id": cam_id})

        # based on the initial resize, correct focal length and principal point
        for i, (intrins, cam_id) in enumerate(zip(pred["intrinsics"], cam_id)):
            cam = CameraFactory.create_from_id(cam_id)
            pred["intrinsics"][i] = cam.reverse_scale_and_shift(intrins, scale_xy, shift_xy)
        if non_batched:
            pred = {k: v[0] for k, v in pred.items()}
        pred |= {"pred_size": target_size}
        return pred

    def forward(self, data: dict) -> dict:
        im = data["image"]
        cam_id = data["cam_id"]
        cxcy = data.get("cxcy", None)
        return self.predict(im, cam_id, cxcy)

    def compute_target_size(self, target_res: float, target_ar: float) -> tuple[int, int]:
        """Compute the target image size given the target resolution and aspect ratio."""
        w = sqrt(target_res / target_ar)
        h = target_ar * w
        # closest image size satisfying `edge_divisible_by` constraint
        div = self.EDGE_DIVISIBLE_BY
        target_size = (round(h / div) * div, round(w / div) * div)
        return target_size

    @staticmethod
    def set_im_size(im: Tensor, target_size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
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
            shift_xy[0] = -(crop_w // 2)  # NOTE: careful: -(crop_w // 2) != -crop_w // 2
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
