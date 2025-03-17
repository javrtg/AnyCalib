import importlib

from anycalib.cameras import EUCM
from anycalib.cameras.base import BaseCamera


class SimpleEUCM(EUCM):
    """Implementation of the Enhanced Unified Camera Model (EUCM) [1, Sec. II].

    The (ordered) intrinsic parameters are f, cx, cy, xi:
        - f [pixels] are the focal lengths,
        - (cx, cy) [pixels] is the principal points.
        - alpha
        - beta


    [1] An Enhanced Unified Camera Model. B. Khomutenko et al., RA-L 2015.
    """

    NAME = "simple_eucm"
    # number of focal lengths
    NUM_F = 1
    PARAMS_IDX = {
        "f": 0,
        "cx": 1,
        "cy": 2,
        "k1": 3,  # alpha
        "k2": 4,  # beta
    }
    num_k = 2

    def __init__(
        self,
        proxy_cam_id: str = "simple_kb:3",
        proxy_cam_id_sac: str = "simple_kb:2",
        safe_optim: bool = True,
        beta_optim_min: float = 1e-6,
        beta_optim_max: float = 1e2,
    ):
        assert "simple" in proxy_cam_id and "simple" in proxy_cam_id_sac
        # FIXME: ugly import to avoid circular imports
        CameraFactory = importlib.import_module("anycalib.cameras.factory").CameraFactory  # fmt: skip
        # Intermediate camera model used during linear fitting
        self.proxy_cam: BaseCamera = CameraFactory.create_from_id(proxy_cam_id)
        self.proxy_cam_sac: BaseCamera = CameraFactory.create_from_id(proxy_cam_id_sac)
        self.safe_optim = safe_optim
        # bounds for β during optimization (ignored if safe_optim=False)
        assert beta_optim_max >= beta_optim_min > 0, "β_max >= β_min > 0 not satisfied."
        self.beta_min = beta_optim_min
        self.beta_max = beta_optim_max
        self.beta_ptp = beta_optim_max - beta_optim_min
