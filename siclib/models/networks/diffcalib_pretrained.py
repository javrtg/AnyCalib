from pathlib import Path

import torch
from diffcalib.diffcalib.diffcalib_pipeline_rgb_12inchannels import DiffcalibPipeline
from diffcalib.diffcalib.util.seed_all import seed_all
from diffcalib.tools.calibrator import MonocularCalibrator
from diffusers import UNet2DConditionModel  # type: ignore
from diffusers.schedulers import DDIMScheduler  # type: ignore
from PIL import Image
from torchvision import transforms

from siclib.models import BaseModel


class DiffCalib(BaseModel):

    default_conf = {
        "denoise_steps": 10,
        "checkpoint": "third_party/diffcalib/checkpoint/stable-diffusion-2-1-marigold-8inchannels",
        "unet_ckpt_path": "third_party/diffcalib/checkpoint/diffcalib-best-0.07517",
        "preprocessing_res": 768,
        "device": "cuda",
    }

    def _init(self, conf):
        """Initialize DiffCalib model."""
        repo_root = Path(__file__).parents[3]
        checkpoint_path = str(repo_root / conf.checkpoint)
        unet_ckpt_path = str(repo_root / conf.unet_ckpt_path)
        diffcalib_params_ckpt = {
            "torch_dtype": torch.float32,
            "unet": UNet2DConditionModel.from_pretrained(
                unet_ckpt_path, subfolder="unet", revision=None
            ),
            "scheduler": DDIMScheduler.from_pretrained(checkpoint_path, subfolder="scheduler"),
        }

        pipe = DiffcalibPipeline.from_pretrained(checkpoint_path, **diffcalib_params_ckpt)
        pipe.enable_xformers_memory_efficient_attention()

        self.dev = torch.device(conf.device)
        self.pipe = pipe.to(self.dev)
        self.totensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=0.5, std=0.5)
        self.monocalibrator = MonocularCalibrator(l1_th=0.02)

    @torch.no_grad()
    def _forward(self, data):
        assert len(data["path"]) == 1, f"Only batch size of 1 is supported (bs={len(data['path'])}"
        cam_id = data["cam_id"][0]
        assert cam_id == "pinhole"

        rgb = Image.open(data["path"][0])
        wo, ho = rgb.size

        # resize
        rgb = rgb.resize((self.conf.preprocessing_res, self.conf.preprocessing_res))
        rgb = self.normalize(self.totensor(rgb))
        pipe_out = self.pipe(
            # validation_prompt,
            rgb,
            denoising_steps=10,
            mode="incident",
        )
        incidence = pipe_out["incident_np"]

        # if args.mode == 'incident':
        K = self.monocalibrator.calibrate_camera_4DoF(
            torch.tensor(incidence).unsqueeze(0).to(self.dev), self.dev, RANSAC_trial=2048
        )
        scale_x = wo / self.conf.preprocessing_res
        scale_y = ho / self.conf.preprocessing_res
        intrinsics = torch.stack(
            (
                K[0, 0] * scale_x,
                K[1, 1] * scale_y,
                K[0, 2] * scale_x,
                K[1, 2] * scale_y,
            )
        ).unsqueeze(0)
        return {"intrinsics": intrinsics}

    def loss(self, pred, data):
        raise NotImplementedError


if __name__ == "__main__":
    from pathlib import Path

    dir_root = Path(__file__).parents[3]
    path = dir_root / "data/lamar2k/images/655367721.jpg"

    model = DiffCalib({})
    output = model({"path": [str(path)], "cam_id": ["pinhole"]})
    print(output)
