import logging

import torch
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
    def __init__(self, in_channels: int, does_fusion: bool = True):
        super().__init__()
        self.does_fusion = does_fusion
        if does_fusion:
            self.res_conv_unit1 = ResidualConvUnit(in_channels)
        self.res_conv_unit2 = ResidualConvUnit(in_channels)

    def forward(self, x: Tensor, x_from_top: Tensor | None = None) -> Tensor:
        if x_from_top is not None:
            assert self.does_fusion and x.shape == x_from_top.shape
            x = self.res_conv_unit1(x) + x_from_top
        x = self.res_conv_unit2(x)
        return x


class ReassembleBlocks(nn.Module):
    """Reassemble block with 'ignore" readout and 2x bilinear up-sampling"""

    def __init__(self, embed_dim: int, post_process_channels: list[int]):
        super().__init__()
        self.projects = nn.ModuleList(
            [nn.Conv2d(embed_dim, out_channel, 1) for out_channel in post_process_channels]
        )

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        return [
            F.interpolate(project(x), scale_factor=2, mode="bilinear", align_corners=False)
            for x, project in zip(inputs, self.projects)
        ]


class LightDPTDecoder(nn.Module):
    """DPT decoder https://arxiv.org/pdf/2103.13413

    Based on DINOv2 and DPT implementations:
    * https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/eval/depth/models/decode_heads/dpt_head.py#L227
    * https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L221
    * https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/blocks.py

    This implementation differs from the one of DINOv2 in that we return a prediction
    at (H/7, W/7) spatial resolution (instead of (4/7 H, 4/7 W) as DINOv2's DPT---prior
    to depth estimation).

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
            post_process_channels * 4 if len(post_process_channels) == 1 else post_process_channels
        )

        # just "resample"'s 1x1 convs -> 2x upsampling -> projection to D^ dims
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

    def forward(self, inputs: dict[str, list[Tensor]]) -> Tensor:
        """Forward pass

        Args:
            inputs: Dict with the following key-value pairs:
                - outputs: list of (B, embdedding_dim, H // 14, W // 14) dinov2 embeddings.
                - class_tokens: list of (B, embdedding_dim) class tokens.

        Returns:
            (B, dim_dhat, H/7, W/7) tensor
        """
        x = inputs["outputs"]  # ignore class tokens
        assert len(x) == self.num_reassemble_blocks

        x = self.reassemble_blocks(x)
        x = [conv(x_) for conv, x_ in zip(self.convs, x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, self.num_fusion_blocks):
            out = self.fusion_blocks[i](x[-i - 1], out)
        out = self.project(out)
        return out


if __name__ == "__main__":
    import time

    import torch

    from siclib.models.encoders.dinov2 import DINOv2

    def closest_multiple(a, div=14):
        return div * round(a / div)

    backbone = DINOv2(model_name="dinov2_vits14", norm_layer=False)
    backbone.cuda()
    decoder = LightDPTDecoder(embed_dim=backbone.embed_dim)
    decoder.cuda()

    x = torch.randn(2, 3, closest_multiple(320), closest_multiple(320)).cuda()
    out = backbone(x)
    print(f"#Layers: {len(out['outputs'])}, shapes: {[y.shape for y in out['outputs']]}")

    torch.cuda.synchronize()
    start_time = time.time()
    dpt_pred = decoder(out)
    torch.cuda.synchronize()
    end_time = time.time()

    print(f"{dpt_pred.shape=}")
    print(f"Execution Time: {1e3 * (end_time - start_time):.2f} ms")
