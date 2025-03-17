import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF

from siclib.models import BaseModel


class WildCam(BaseModel):

    default_conf = {"with_assumption": False}

    def _init(self, conf):
        """Initialize WildCam model."""
        model = NEWCRFIF(version="large07", pretrained=None)
        # pretrained
        pretrained_resource = "https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/wild_camera_all.pth"
        state_dict = torch.hub.load_state_dict_from_url(pretrained_resource, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        self.model = model

    def _forward(self, data):
        """WildCam forward pass"""
        assert len(data["image"]) == 1, "Only batch size = 1 supported."
        if "cam_id" in data:
            assert (
                "pinhole" in data["cam_id"][0]
            ), f"Only pinhole camera supported. Got: {data['cam_id']}"

        # convert tensor to PIL image
        image = Image.fromarray(
            (data["image"].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )
        w, h = image.size

        K, rays = self.model.inference(image, wtassumption=self.conf.with_assumption)

        # the predicted grid of rays by WildCam always have a shape of (480, 640), so we
        # resize the grid of rays to the image size for visualization purposes
        rays = F.normalize(
            F.interpolate(rays, (h, w), mode="bilinear", align_corners=False), dim=1
        )  # (1, 3, H, W)
        rays = rays.view(1, 3, -1).permute(0, 2, 1).contiguous()  # (1, H*W, 3)

        if "cam_id" in data and data["cam_id"][0] == "simple_pinhole":
            intrinsics = torch.tensor([K[0, 0], K[0, 2], K[1, 2]]).unsqueeze(0)
        else:
            intrinsics = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]).unsqueeze(0)
        return {"intrinsics": intrinsics, "rays": rays}

    def loss(self, pred, data):
        raise NotImplementedError


if __name__ == "__main__":
    from pathlib import Path

    dir_root = Path(__file__).parents[3]

    # load image
    path = dir_root / "data/lamar2k/images/655367721.jpg"
    im = torch.from_numpy(np.array(Image.open(path).convert("RGB"))) / 255.0
    im_ = im.permute(2, 0, 1).unsqueeze(0).cuda()

    model = WildCam({})
    output = model({"image": im_})

    print(output["intrinsics"])
    print(output["rays"].shape)
