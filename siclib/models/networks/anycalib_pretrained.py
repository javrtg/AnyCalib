import torch.nn.functional as F

from anycalib.manifolds import Unit3
from anycalib.model import AnyCalib
from siclib.models.base_model import BaseModel


class AnyCalibPretrained(BaseModel):
    """AnyCalib pretrained model for evaluation."""

    default_conf = {
        "model_id": "anycalib_pinhole",
        "nonlin_opt_method": "gauss_newton",
        "nonlin_opt_conf": None,
        "init_with_sac": False,
        "fallback_to_sac": True,
        "ransac_conf": None,
        "rm_borders": 0,
        "sample_size": -1,
    }

    def _init(self, conf):
        """Initialize pretrained AnyCalib model."""
        self.model = AnyCalib(
            model_id=conf.model_id,
            nonlin_opt_method=conf.nonlin_opt_method,
            nonlin_opt_conf=conf.nonlin_opt_conf,
            init_with_sac=conf.init_with_sac,
            fallback_to_sac=conf.fallback_to_sac,
            ransac_conf=conf.ransac_conf,
            rm_borders=conf.rm_borders,
            sample_size=conf.sample_size,
        )

    def _forward(self, data: dict):
        assert len(data["image"]) == 1, "Batch size must be 1"
        pred = self.model.predict(data["image"], data["cam_id"])
        # upsample tangent_coords (FoV field) to input resolution for visualization
        h, w = pred["pred_size"]
        ho, wo = data["image"].shape[-2:]
        pred["tangent_coords"] = (
            F.interpolate(
                pred["tangent_coords"].view(1, h, w, 2).permute(0, 3, 1, 2),
                size=(ho, wo),
                mode="bilinear",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .view(1, ho * wo, 2)
        )
        # map to rays
        pred["rays"] = Unit3.expmap_at_z1(pred["tangent_coords"])
        return pred

    def metrics(self, pred, data):
        raise NotImplementedError

    def loss(self, pred, data):
        raise NotImplementedError
