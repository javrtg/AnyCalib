"""Interface for GeoCalib inference package with inference compatible with
models that predict rays"""

import torch
from geocalib import GeoCalib

from siclib.models.base_model import BaseModel


# mypy: ignore-errors
class GeoCalibPretrained(BaseModel):
    """GeoCalib pretrained model."""

    default_conf = {
        "camera_model": "pinhole",
        "model_weights": "pinhole",
    }

    def _init(self, conf):
        """Initialize pretrained GeoCalib model."""
        self.model = GeoCalib(weights=conf.model_weights)

    def _forward(self, data):
        """Forward pass."""
        priors = {}
        if "prior_gravity" in data:
            priors["gravity"] = data["prior_gravity"]

        if "prior_focal" in data:
            priors["focal"] = data["prior_focal"]

        results = self.model.calibrate(
            data["image"], camera_model=self.conf.camera_model, priors=priors
        )

        # assert all(id_ == "pinhole" for id_ in data["cam_id"]), "Only pinhole is supported for now."
        cam = results["camera"]
        fxfy = cam.f  # type: ignore
        cxcy = cam.c  # type: ignore
        if self.conf.camera_model == "pinhole":
            assert all(id_ == "pinhole" for id_ in data["cam_id"]), data["cam_id"]
            results["intrinsics"] = torch.cat([fxfy, cxcy], dim=-1)

        elif self.conf.camera_model == "simple_radial":
            assert all(id_ == "radial:1" for id_ in data["cam_id"]), data["cam_id"]
            k1 = cam.k1.unsqueeze(-1)  # type: ignore
            params = torch.cat([fxfy, cxcy, k1], dim=-1)
            assert params.shape[-1] == 5
            results["intrinsics"] = params

        elif self.conf.camera_model == "simple_divisional":
            assert all(id_ == "division:1" for id_ in data["cam_id"]), data["cam_id"]
            k1 = cam.k1.unsqueeze(-1)  # type: ignore
            params = torch.cat([fxfy, cxcy, k1], dim=-1)
            assert params.shape[-1] == 5
            results["intrinsics"] = torch.cat([fxfy, cxcy, k1], dim=-1)

        else:
            raise NotImplementedError
        return results

    def metrics(self, pred, data):
        """Compute metrics."""
        raise NotImplementedError("GeoCalibPretrained does not support metrics computation.")

    def loss(self, pred, data):
        """Compute loss."""
        raise NotImplementedError("GeoCalibPretrained does not support loss computation.")
