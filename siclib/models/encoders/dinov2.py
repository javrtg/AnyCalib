import torch
import torch.nn as nn
from torch import Tensor

from siclib.models.encoders.vision_transformer import (
    DinoVisionTransformer,
    vit_base,
    vit_giant2,
    vit_large,
    vit_small,
)

DINOV2_CFG = {
    "dinov2_vits14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "url_r": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
        "embed_dim": 384,
        "constructor": vit_small,
        "out_index": [2, 5, 8, 11],
    },
    "dinov2_vitb14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "url_r": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
        "embed_dim": 768,
        "constructor": vit_base,
        "out_index": [2, 5, 8, 11],
    },
    "dinov2_vitl14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
        "url_r": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
        "embed_dim": 1024,
        "constructor": vit_large,
        "out_index": [4, 11, 17, 23],
    },
    "dinov2_vitg14": {
        "url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
        "url_r": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
        "embed_dim": 1536,
        "constructor": vit_giant2,
        "out_index": [9, 19, 29, 39],
    },
}


class DINOv2(nn.Module):
    """DINOv2 wrapper based on:

    * hubconf of DINOv2:
        https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/hub/backbones.py
    """

    image_mean: Tensor
    image_std: Tensor

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        norm_layer: bool = False,
        num_trainable_blocks: int = -1,
        with_registers: bool = False,
        intermediate_layers: int | list[int] | None = None,
    ):
        super().__init__()

        if with_registers:
            raise NotImplementedError

        dinov2_cfg = DINOV2_CFG[model_name]
        # load and, if required, download dinov2 weights
        url = dinov2_cfg["url_r" if with_registers else "url"]
        model_dir = f"{torch.hub.get_dir()}/checkpoints"  # default
        weights = torch.hub.load_state_dict_from_url(url, model_dir, map_location="cpu")
        # args based on https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/hub/backbones.py
        self.model: DinoVisionTransformer = dinov2_cfg["constructor"](
            img_size=518,
            patch_size=14,
            init_values=1.0,
            ffn_layer="swiglufused" if model_name.endswith("vitg14") else "mlp",
            block_chunks=0,
            num_register_tokens=4 if with_registers else 0,
            interpolate_antialias=with_registers,
            interpolate_offset=0.0 if with_registers else 0.1,
        )
        self.model.load_state_dict(weights)

        if not -1 <= num_trainable_blocks <= len(self.model.blocks):
            raise ValueError(
                f"Number of trainable blocks: {num_trainable_blocks}, exceeds the total"
                f" number of blocks: {len(self.model.blocks)}, or is less than -1."
            )
        num_trainable_blocks = (
            len(self.model.blocks) if num_trainable_blocks == -1 else num_trainable_blocks
        )

        # freeze mask token, only used in DINOv2's training
        self.model.mask_token.requires_grad_(False)
        # # freeze input embedding parameters and class token
        # self.model.cls_token.requires_grad_(False)
        # self.model.pos_embed.requires_grad_(False)
        # self.model.patch_embed.requires_grad_(False)
        # freeze non-trainable blocks
        freeze_up_to = len(self.model.blocks) - num_trainable_blocks
        for blk in self.model.blocks[:freeze_up_to]:  # type: ignore
            for param in blk.parameters():
                param.requires_grad_(False)
        # freeze norm layer if not used
        if not norm_layer:
            self.model.norm.requires_grad_(False)

        # intermediate layers to extract features from
        if intermediate_layers is None:
            self.out_index = dinov2_cfg["out_index"]
        elif isinstance(intermediate_layers, int):
            # extract the output of the last n blocks
            total_blocks = len(self.model.blocks)
            self.out_index = list(range(total_blocks - intermediate_layers, total_blocks))
        else:
            self.out_index = intermediate_layers
        assert len(self.out_index) == 4, "Only 4 intermediate layers are supported."

        self.with_registers = with_registers
        self.embed_dim = dinov2_cfg["embed_dim"]
        self.num_trainable_blocks = num_trainable_blocks
        self.freeze_up_to = freeze_up_to
        self.norm_layer = norm_layer
        self.patch_size = 14

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def forward(self, x: Tensor) -> dict[str, list[Tensor]]:
        """Extract features from last/intermediate layer/s

        Args:
            x: (B, 3, H, W) input tensor.

        Returns:
            Dict with the following key-value pairs:
                - outputs: list of (B, embdedding_dim, H // 14, W // 14) dinov2 embeddings.
                - class_tokens: list of (B, embdedding_dim) class tokens.
        """
        b, _, h, w = x.shape
        x = (x - self.image_mean) / self.image_std
        x = self.model.prepare_tokens_with_masks(x)

        outputs: list[Tensor] = []
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_index:
                outputs.append(x)
        if self.norm_layer:
            outputs = [self.model.norm(out) for out in outputs]

        class_tokens = [out[:, 0].contiguous() for out in outputs]

        p = self.patch_size
        outputs = [
            out[:, 1:].reshape(b, h // p, w // p, self.embed_dim).permute(0, 3, 1, 2).contiguous()
            for out in outputs
        ]
        return {"outputs": outputs, "class_tokens": class_tokens}


# # To test the forward pass and check if gradients are flowing:
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model_str = "dinov2_vits14"
#     trainable_blocks = 2
#     model = DINOv2_(model_name=model_str, norm_layer=False, num_trainable_blocks=trainable_blocks)
#     model = model.to(device)
#     model.train()
#     for param in model.parameters():
#         assert param.requires_grad

#     x = torch.randn((10, 3, 224, 224), device=device)  # Example input tensor
#     output = model(x)

#     loss = output.sum()  # Simple loss for testing
#     loss.backward()

#     model2 = DINOv2(model_name=model_str, num_trainable_blocks=trainable_blocks)
#     model2 = model2.to(device)
#     model2.train()

#     outputs_2, cls_ = model2(x)
#     output2 = outputs_2[-1]

#     loss2 = output2.sum()  # Simple loss for testing
#     loss2.backward()

#     assert torch.allclose(output, output2)
#     # check for equal gradients
#     counter = 0
#     for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
#         if p1.grad is None:
#             if p2.grad is not None:
#                 print(n1, n2)
#             # assert p2.grad is None, (n1, n2)
#             continue
#         assert torch.allclose(p1.grad, p2.grad, atol=1e-6)
#         counter += 1

#     # counter = 0
#     # for p1, p2 in zip(model.parameters(), model2.parameters()):
#     #     if p1.grad is None:
#     #         if p2.grad is not None:
#     #             print(p1.names, p2.names, p2.grad.shape)
#     #         # assert p2.grad is None, (n1, n2)
#     #         continue
#     #     assert torch.allclose(p1.grad, p2.grad, atol=1e-6)
#     #     counter += 1
#     print(counter)
