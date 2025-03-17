import logging

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class ResidualConvUnit(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x) + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block (right side of DPT's paper Figure 1)"""

    def __init__(self, in_channels: int, does_fusion: bool = True):
        super().__init__()
        self.does_fusion = does_fusion
        if does_fusion:
            self.res_conv_unit1 = ResidualConvUnit(in_channels)
        self.res_conv_unit2 = ResidualConvUnit(in_channels)
        self.project = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: Tensor, x_from_top: Tensor | None = None) -> Tensor:
        if x_from_top is not None:
            assert self.does_fusion
            # the downsampling at ReassembleBlocks may result in spatial shapes that are
            # not multiples of 2, causing them not to match after the final ×2 upsampling.
            if x.shape != x_from_top.shape:
                x = F.interpolate(
                    x, size=x_from_top.shape[-2:], mode="bilinear", align_corners=False
                )
            x = self.res_conv_unit1(x) + x_from_top
        x = self.res_conv_unit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.project(x)
        return x


class ReassembleBlocks(nn.Module):
    """Reassemble block with 'ignore" readout and DPT's resampling strategy."""

    def __init__(self, embed_dim: int, post_process_channels: list[int]):
        super().__init__()
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, out_channel, 1) for out_channel in post_process_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=post_process_channels[0],
                    out_channels=post_process_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=post_process_channels[1],
                    out_channels=post_process_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=post_process_channels[3],
                    out_channels=post_process_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return [
            resize(project(x))
            for x, project, resize in zip(inputs, self.projects, self.resize_layers)
        ]


class DPTDecoder(nn.Module):
    """DPT decoder https://arxiv.org/pdf/2103.13413

    Based on DINOv2 and DPT implementations:
    * https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/eval/depth/models/decode_heads/dpt_head.py#L227
    * https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L221
    * https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/blocks.py


    Args:
        embed_dim: dimension of the encoded features (e.g. 768 for vitb).
        readout_type: readout type, only "ignore" is currently supported.
        post_process_channels: list of output channels for each reassemble block.
        dim_dhat: intermediate feature dimension (D^hat in DPT's paper).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        readout_type: str = "ignore",
        post_process_channels: list[int] | None = None,
        dim_dhat: int = 256,
    ):
        super().__init__()

        if readout_type != "ignore":
            raise NotImplementedError

        if post_process_channels is None:
            # e.g. [96, 192, 384, 768] for vitb (embed_dim=768)
            post_process_channels = [embed_dim // 2 ** (3 - i) for i in range(4)]
        post_process_channels = (
            list(post_process_channels) * 4
            if len(post_process_channels) == 1
            else post_process_channels
        )

        # "resample"'s 1x1 convs + up/downsampling -> projection to D^ dims
        self.reassemble_blocks = ReassembleBlocks(embed_dim, post_process_channels)
        # projection to D^ dims
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(out_channel, dim_dhat, 3, padding=1)
                for out_channel in post_process_channels
            ]
        )
        # fusion between intermediate layers without upsampling
        self.fusion_blocks = nn.ModuleList(
            [FeatureFusionBlock(dim_dhat, does_fusion=(i != 0)) for i in range(4)]
        )
        self.project = nn.Sequential(nn.Conv2d(dim_dhat, dim_dhat, 3, padding=1), nn.ReLU(True))
        # self.project = nn.Conv2d(dim_dhat, dim_dhat, 3, padding=1)

        # info
        self.out_channels = dim_dhat
        self.post_process_channels = post_process_channels
        self.num_post_process_channels = len(post_process_channels)
        self.num_reassemble_blocks = len(self.reassemble_blocks.projects)
        self.num_fusion_blocks = len(self.fusion_blocks)
        assert self.num_post_process_channels == self.num_reassemble_blocks
        assert self.num_post_process_channels == self.num_fusion_blocks

    def forward(self, inputs: dict[str, list[Tensor]]) -> tuple[Tensor, tuple[int, int]]:
        """Forward pass

        Args:
            inputs: Dict with the following key-value pairs:
                - outputs: list of (B, embdedding_dim, H // 14, W // 14) dinov2 embeddings.
                - class_tokens: list of (B, embdedding_dim) class tokens.

        Returns:
            dict with the following key-value pairs:
                - "rays": (B, 3, H, W) normalized rays.
                - "log_covs": (B, 2, H, W) logarithm of the diagonal elements of the
                    covariance matrices.
        """
        x = inputs["outputs"]  # ignore class tokens
        assert len(x) == self.num_reassemble_blocks
        w = 14 * x[0].shape[-1]  # to match the input size at the end
        h = 14 * x[0].shape[-2]

        # DPT decoder
        x = self.reassemble_blocks(x)
        x = [conv(x_) for conv, x_ in zip(self.convs, x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, self.num_fusion_blocks):
            out = self.fusion_blocks[i](x[-i - 1], out)
        out = self.project(out)  # out.shape = (B, dim_dhat, 8*H/14, 8*W/14)
        return out, (h, w)


if __name__ == "__main__":
    import time

    import torch

    from siclib.models.encoders.dinov2 import DINOv2

    def closest_multiple(a, div=14):
        return div * round(a / div)

    backbone = DINOv2(model_name="dinov2_vits14", norm_layer=False)
    backbone.cuda()
    decoder = DPTDecoder(embed_dim=backbone.embed_dim)
    decoder.cuda()

    x = torch.randn(2, 3, closest_multiple(320), closest_multiple(320)).cuda()
    out = backbone(x)
    print(f"#Layers: {len(out['outputs'])}, shapes: {[y.shape for y in out['outputs']]}")

    torch.cuda.synchronize()
    start_time = time.time()
    dpt_pred = decoder(out)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"{dpt_pred['rays'].shape=}")
    print(f"{dpt_pred['log_covs'].shape=}")
    print(f"Execution Time: {1e3 * (end_time - start_time):.2f} ms")
