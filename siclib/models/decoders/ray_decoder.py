import logging
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from anycalib.manifolds import Unit3

logger = logging.getLogger(__name__)


def cvx_upsample(x: Tensor, mask: Tensor, up_factor: int = 7) -> Tensor:
    """Upsample [H/k, W/k, C] -> [H, W, C] using convex combination of 3x3 patches.

    Code adapted from RAFT (Teed and Deng, 2020):
    https://github.com/princeton-vl/RAFT/blob/3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02/core/raft.py#L72

    Args:
        x: (N, C, H, W) input tensor
        mask: (N, 9, 1, 1, H, W) already softmaxed mask tensor
        up_factor: upsample factor
    """
    N, C, H, W = x.shape
    up_x = F.unfold(x, (3, 3), padding=1)
    up_x = up_x.view(N, C, 9, 1, 1, H, W)
    up_x = torch.sum(mask * up_x, dim=2)
    up_x = up_x.permute(0, 1, 4, 2, 5, 3)
    return up_x.reshape(N, C, up_factor * H, up_factor * W)


class ConvexRayDecoder(nn.Module):
    """Convex Ray Decoder

    This decoder direcly predicts the 3D ray coordinates for each pixel in the image.
    It uses convex upsampling (from RAFT -Teed and Deng, 2020) to upsample the rays from
    a lower resolution to the input resolution.

    Args:
        in_channels: number of input channels
        up_factor: upsampling factor
        predict_covs: whether to predict covariances
        logvar_lims: limits for clamping the log variances
    """

    def __init__(
        self,
        in_channels: int = 256,
        up_factor: int = 7,
        predict_covs: bool = False,
        logvar_lims: tuple[float, float] = (-20, 10),
        **ignored_kwargs,
    ):
        super().__init__()
        assert len(logvar_lims) == 2 and logvar_lims[0] < logvar_lims[1], logvar_lims
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored arguments: {ignored_kwargs}")

        self.logvar_max = logvar_lims[1]
        self.logvar_min = logvar_lims[0]
        self.in_channels = in_channels
        self.up_factor = up_factor
        self.predict_covs = predict_covs

        # rays head
        self.rays_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # unnormalized rays (3) + covs (2)
            nn.Conv2d(in_channels // 2, 5 if predict_covs else 3, 1),
        )
        # weights head for convex upsampling to input resolution
        self.upsampling_weights_head = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, up_factor**2 * 9, 1, padding=0),
            nn.Unflatten(1, (1, 9, up_factor, up_factor)),
            nn.Softmax(dim=2),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # head
        rays_pred = self.rays_head(x)  # (B, 5, H/7, W/7)
        weights = self.upsampling_weights_head(x)
        # postprocess + upsample
        rays = F.normalize(rays_pred[:, :3], dim=1)
        rays = F.normalize(cvx_upsample(rays, weights, self.up_factor), dim=1)
        out = {"rays": rays}
        if self.predict_covs:
            log_covs = cvx_upsample(rays_pred[:, 3:], weights, self.up_factor)
            log_covs = log_covs.clamp(self.logvar_min, self.logvar_max)
            out["log_covs"] = log_covs
        return out


class ConvexTangentDecoder(nn.Module):
    """Convex Tangent Coordinates Decoder.

    This decoder predicts ray coordinates projected onto the tangent plane of the optical
    axis. Thereby this decoder predicts a minimal (2D) representation of the 3D rays.

    Args:
        in_channels: number of input channels
        up_factor: upsampling factor
        predict_covs: whether to predict covariances
        logvar_lims: limits for clamping the log variances
    """

    def __init__(
        self,
        in_channels: int = 256,
        up_factor: int = 7,
        predict_covs: bool = False,
        predict_mixture: bool = False,
        use_tanh: bool = False,
        logvar_lims: tuple[float, float] = (-20, 10),
        **ignored_kwargs,
    ):
        super().__init__()
        if len(logvar_lims) != 2 or logvar_lims[0] >= logvar_lims[1]:
            raise ValueError(f"Invalid logvar_lims: {logvar_lims}")
        if predict_covs and predict_mixture:
            raise ValueError("Cannot predict both covariances and mixture parameters.")

        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored arguments: {ignored_kwargs}")

        self.logvar_max = logvar_lims[1]
        self.logvar_min = logvar_lims[0]
        self.in_channels = in_channels
        self.up_factor = up_factor
        self.predict_covs = predict_covs
        self.predict_mixture = predict_mixture
        self.use_tanh = use_tanh

        if predict_mixture:
            out_dim = 6
        elif predict_covs:
            out_dim = 4
        else:
            out_dim = 2

        # tangent head
        self.tangent_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # tangent coords (2) + covs (2)
            nn.Conv2d(in_channels // 2, out_dim, 1),
        )
        # weights head for convex upsampling to input resolution
        self.upsampling_weights_head = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, up_factor**2 * 9, 1, padding=0),
            nn.Unflatten(1, (1, 9, up_factor, up_factor)),
            nn.Softmax(dim=2),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | float]:
        # head
        tangent_pred = self.tangent_head(x)  # (B, 5, H/7, W/7)
        weights = self.upsampling_weights_head(x)
        # upsample
        tangent_pred = cvx_upsample(tangent_pred, weights, self.up_factor)
        # postprocess
        tangent_coords = tangent_pred[:, :2]
        if self.use_tanh:
            tangent_coords = pi * torch.tanh(tangent_coords)
        rays = Unit3.expmap_at_z1(tangent_coords.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out: dict[str, Tensor | float] = {"rays": rays, "tangent_coords": tangent_coords}

        if self.predict_covs:
            out["log_covs"] = tangent_pred[:, 2:].clamp(self.logvar_min, self.logvar_max)
        if self.predict_mixture:
            # NOTE: due to current api, we call Laplace's b parameters as "log_covs"
            # despite not being log variances
            out["weights"] = tangent_pred[:, 2:4].mean(dim=(2, 3), keepdim=True)
            out["log_covs"] = tangent_pred[:, 4:].clamp(self.logvar_min, self.logvar_max)
            out["min_b"] = self.logvar_min
        return out


class ConvexTangentEditDecoder(nn.Module):
    """Decoder for edit maps (pixel aspect ratio and distortion center) + tangent coordinates,
    followed by convex upsampling to input resolution.

    This decoder predicts ray coordinates projected onto the tangent plane of the optical
    axis. Thereby this decoder predicts a minimal (2D) representation of the 3D rays.

    Args:
        in_channels: number of input channels
        up_factor: upsampling factor
        predict_covs: whether to predict covariances
        logvar_lims: limits for clamping the log variances
    """

    def __init__(
        self,
        in_channels: int = 256,
        up_factor: int = 7,
        predict_covs: bool = False,
        logvar_lims: tuple[float, float] = (-20, 10),
        **ignored_kwargs,
    ):
        super().__init__()
        if len(logvar_lims) != 2 or logvar_lims[0] >= logvar_lims[1]:
            raise ValueError(f"Invalid logvar_lims: {logvar_lims}")

        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored arguments: {ignored_kwargs}")

        self.logvar_max = logvar_lims[1]
        self.logvar_min = logvar_lims[0]
        self.in_channels = in_channels
        self.up_factor = up_factor
        self.predict_covs = predict_covs

        self.edit_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # fy/fx (1) + scale (uncertainty) (1) + distance to principal point (1)
            nn.Conv2d(in_channels // 2, 3, 1),
        )
        self.post_edit = nn.Sequential(
            nn.Conv2d(3, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.ReLU(True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            # nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(True),
        )

        # remaining heads
        self.tangent_head = nn.Sequential(
            # nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            nn.Conv2d(in_channels // 2 + in_channels // 4, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # tangent coords (2) + covs (2)
            nn.Conv2d(in_channels // 2, 4 if predict_covs else 2, 1),
        )
        # weights head for convex upsampling to input resolution
        self.upsampling_weights_head = nn.Sequential(
            # convex combination of 3x3 patches
            nn.Conv2d(in_channels // 2 + in_channels // 4, in_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, up_factor**2 * 9, 1, padding=0),
            nn.Unflatten(1, (1, 9, up_factor, up_factor)),
            nn.Softmax(dim=2),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor | float]:
        # edit maps: fy/fx (pixel aspect ratio) | scale (uncertainty) and pix radii
        edit_pred = self.edit_head(x)  # (B, 4, H/7, W/7)
        out: dict[str, Tensor | float] = {"pix_ar_map": edit_pred[:, :2], "radii": edit_pred[:, 2]}
        # weighted residual connection
        # x = self.post_edit(edit_pred) + self.project(x)
        x = torch.cat((self.post_edit(edit_pred), self.project(x)), dim=1)
        # tangent coordinates + upsampling
        tangent_pred = self.tangent_head(x)  # (B, 4, H/7, W/7)
        weights = self.upsampling_weights_head(x)
        tangent_pred = cvx_upsample(tangent_pred, weights, self.up_factor)
        # postprocess
        tangent_coords = tangent_pred[:, :2]
        rays = Unit3.expmap_at_z1(tangent_coords.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out |= {"rays": rays, "tangent_coords": tangent_coords}
        if self.predict_covs:
            out["log_covs"] = tangent_pred[:, 2:].clamp(self.logvar_min, self.logvar_max)
        return out


### DECODERS WITH ORIGINAL DPT ###


class RayDecoder(nn.Module):
    """Ray Decoder"""

    def __init__(
        self,
        in_channels: int = 256,
        predict_covs: bool = False,
        logvar_lims: tuple[float, float] = (-20, 10),
        **ignored_kwargs,
    ):
        super().__init__()
        assert len(logvar_lims) == 2 and logvar_lims[0] < logvar_lims[1], logvar_lims
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored arguments: {ignored_kwargs}")

        self.logvar_max = logvar_lims[1]
        self.logvar_min = logvar_lims[0]
        self.in_channels = in_channels
        self.predict_covs = predict_covs

        self.rays_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # unnormalized rays (3) + covs (2)
            nn.Conv2d(in_channels // 2, 5 if predict_covs else 3, 1),
        )

    def forward(self, x: tuple[Tensor, tuple[int, int]]) -> dict[str, Tensor]:
        x_, (h, w) = x
        # head
        rays_pred = self.rays_head(x_)
        # post-process and resize
        rays_pred = F.interpolate(rays_pred, size=(h, w), mode="bilinear", align_corners=False)
        out = {"rays": F.normalize(rays_pred[:, :3], dim=1)}
        if self.predict_covs:
            out["log_covs"] = rays_pred[:, 3:].clamp(self.logvar_min, self.logvar_max)
        return out


class TangentDecoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 256,
        predict_covs: bool = False,
        logvar_lims: tuple[float, float] = (-20, 10),
        use_tanh: bool = False,
        **ignored_kwargs,
    ):
        super().__init__()
        assert len(logvar_lims) == 2 and logvar_lims[0] < logvar_lims[1], logvar_lims
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored arguments: {ignored_kwargs}")

        self.logvar_max = logvar_lims[1]
        self.logvar_min = logvar_lims[0]
        self.in_channels = in_channels
        self.predict_covs = predict_covs
        self.use_tanh = use_tanh

        self.tangent_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.ReLU(True),
            # tangent coords (2) + covs (2)
            nn.Conv2d(in_channels // 2, 4 if predict_covs else 2, 1),
        )

    def forward(self, x: tuple[Tensor, tuple[int, int]]) -> dict[str, Tensor]:
        x_, (h, w) = x
        # predict tangent coordinates and optionally covariances at (h, w) resolution
        tangent_pred = self.tangent_head(
            F.interpolate(x_, size=(h, w), mode="bilinear", align_corners=False)
        )
        tangent_coords = tangent_pred[:, :2].contiguous()
        if self.use_tanh:
            tangent_coords = pi * torch.tanh(tangent_coords)
        rays = Unit3.expmap_at_z1(tangent_coords.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = {"rays": rays, "tangent_coords": tangent_coords}
        if self.predict_covs:
            out["log_covs"] = tangent_pred[:, 2:].clamp(self.logvar_min, self.logvar_max)
        return out


if __name__ == "__main__":
    import time

    def predict_and_measure(model, x, repeats=10, discard=2):
        times = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            start_time = time.time()
            pred = model(x)
            torch.cuda.synchronize()
            end_time = time.time()
            times.append(end_time - start_time)
        # discard first (warmup) runs
        times = times[discard:]
        timing = sum(times) / len(times)
        return pred, timing

    h, w = 46, 46
    x = torch.randn(2, 256, h, w, device="cuda")
    # cvx_decoder = ConvexRayDecoder().cuda()
    # pred, t = predict_and_measure(cvx_decoder, x, repeats=10)
    # print(f"ConvexRayDecoder: {t*1000:.2f} ms")
    # print(f"{pred['rays'].shape=}")
    # print(f"{pred['log_covs'].shape=}")

    cvx_decoder_tangent = ConvexTangentDecoder(predict_covs=True).cuda()
    pred, t = predict_and_measure(cvx_decoder_tangent, x, repeats=10)
    print(f"ConvexTangentDecoder: {t*1000:.2f} ms")
    print(f"{pred['rays'].shape=}")
    print(f"{pred['tangent_coords'].shape=}")
    print(f"{pred['log_covs'].shape=}")

    # h, w = 184, 184
    # x = torch.randn(2, 256, h, w, device="cuda")
    # ray_decoder = RayDecoder().cuda()
    # pred, t = predict_and_measure(ray_decoder, (x, (h, w)), repeats=10)
    # print(f"RayDecoder: {t*1000:.2f} ms")
    # print(f"{pred['rays'].shape=}")
    # print(f"{pred['log_covs'].shape=}")

    # tangent_decoder = TangentDecoder().cuda()
    # pred, t = predict_and_measure(tangent_decoder, (x, (h, w)), repeats=10)
    # print(f"TangentDecoder: {t*1000:.2f} ms")
    # print(f"{pred['rays'].shape=}")
    # print(f"{pred['tangent_coords'].shape=}")
    # print(f"{pred['log_covs'].shape=}")
