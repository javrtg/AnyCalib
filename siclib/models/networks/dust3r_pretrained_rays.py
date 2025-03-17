"""Wrapper for DUSt3R model to estimate focal length.

DUSt3R: Geometric 3D Vision Made Easy, https://arxiv.org/abs/2312.14132
"""

import numpy as np
import torch
import torch.nn.functional as F
from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import load_model
from dust3r.utils.image import load_images
from PIL import Image

from siclib.models import BaseModel


class Dust3R(BaseModel):
    """DUSt3R model for focal length estimation."""

    default_conf = {
        "model_path": "weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "device": "cuda",
        "batch_size": 1,
        "schedule": "cosine",
        "lr": 0.01,
        "niter": 300,
        "show_scene": False,
    }

    required_data_keys = ["path"]

    def _init(self, conf):
        self.model = load_model(conf["model_path"], conf["device"])

    def _forward(self, data):
        assert len(data["path"]) == 1, f"Only batch size of 1 is supported (bs={len(data['path'])}"

        path = data["path"][0]
        images = [path] * 2
        ho, wo = data["image"].shape[-2:] if "image" in data else Image.open(path).size[::-1]

        # with torch.enable_grad():
        images = load_images(images, size=512)
        pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
        output = inference(
            pairs, self.model, self.conf["device"], batch_size=self.conf["batch_size"]  # type: ignore
        )

        # raw ray predictions
        rays = F.normalize(output["pred1"]["pts3d"][:1], dim=-1).permute(0, 3, 1, 2)  # type: ignore
        # resize the grid of rays to the image size for visualization purposes
        rays = F.normalize(
            F.interpolate(rays, (ho, wo), mode="bilinear", align_corners=False), dim=1
        )  # (1, 3, H, W)
        rays = rays.view(1, 3, -1).permute(0, 2, 1).contiguous()  # (1, H*W, 3)

        # fit to raw scene
        scene = global_aligner(
            output, device=self.conf["device"], mode=GlobalAlignerMode.PairViewer  # type: ignore
        )
        # get scale for intrinsics
        ht, wt = images[0]["true_shape"][:, 0], images[0]["true_shape"][:, 1]
        scale = float(np.mean([ho / ht, wo / wt]))
        # intrinsics
        intrinsics = scene.get_intrinsics().mean(dim=0)
        f = intrinsics[0, 0]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        intrinsics = scale * (
            torch.stack([f, cx, cy])
            if "simple" in data["cam_id"][0]
            else torch.stack([f, f, cx, cy])
        )
        return {"intrinsics": intrinsics[None], "rays": rays}

    def loss(self, pred, data):
        raise NotImplementedError


if __name__ == "__main__":
    from pathlib import Path

    dir_root = Path(__file__).parents[3]
    path = dir_root / "data/lamar2k/images/655367721.jpg"
    dust3r = Dust3R({})
    output = dust3r({"path": [str(path)], "cam_id": ["pinhole"]})
    print(output)
    print(output["intrinsics"].shape)
