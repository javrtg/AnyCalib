import torch
import torch.nn.functional as F
from moge.model import MoGeModel

from siclib.models import BaseModel


class MoGe(BaseModel):
    """MoGe model for focal length estimation."""

    default_conf = {"force_projection": False}

    # required_data_keys = ["path"]
    required_data_keys = ["image"]

    def _init(self, conf):
        """Initialize the MoGe model."""
        self.device = torch.device("cuda")
        self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
        self.force_projection = conf.force_projection

    def _forward(self, data):
        """MoGe forward pass"""
        assert len(data["image"]) == 1, "Only batch size = 1 is supported."
        if "cam_id" in data:
            assert (
                data["cam_id"][0] == "pinhole"
            ), f"Only pinhole camera supported. Got: {data['cam_id']}"
        image = data["image"].squeeze(0)
        try:
            # if force_projection is False, we get the raw points
            pred = self.model.infer(image, force_projection=self.force_projection)
        except ValueError:
            # some outlier cases lead to crash. Return bogus intrinsics
            return {"intrinsics": torch.tensor([[1.0, 1.0, 1.0, 1.0]], device=self.device)}
        # normalized intrinsics
        K = pred["intrinsics"]
        fx, cx = K[0, 0], K[0, 2]
        fy, cy = K[1, 1], K[1, 2]
        # unnormalize
        h, w = image.shape[-2:]
        fx, cx = fx * w, cx * w
        fy, cy = fy * h, cy * h
        pred["intrinsics"] = torch.tensor([[fx, fy, cx, cy]], device=self.device)
        pred["rays"] = F.normalize(pred["points"], dim=-1).view(1, -1, 3)
        return pred

    def loss(self, pred, data):
        """Loss function for DUSt3R model."""
        return {}, {}


if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from PIL import Image

    dir_root = Path(__file__).parents[3]

    # load image
    path = dir_root / "data/lamar2k/images/655367721.jpg"
    im = torch.from_numpy(np.array(Image.open(path).convert("RGB"))) / 255.0
    im_ = im.permute(2, 0, 1).unsqueeze(0).cuda()

    model = MoGe({})
    output = model({"image": im_})

    print(output["intrinsics"])
