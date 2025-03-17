"""Script to create a dataset from panorama images."""

import hashlib
import logging
from math import acos, cos, hypot, pi, sin, sqrt, tan
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from anycalib.cameras import CameraFactory
from anycalib.cameras.base import BaseCamera

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

RAD2DEG = 180 / pi
DEG2RAD = pi / 180

# we disable the image-size safeguard that PIL has, since otherwise it will raise an error
# when saving images.
Image.MAX_IMAGE_PIXELS = None


def rad2rotmat(roll: float, pitch: float, yaw: float, device: torch.device) -> torch.Tensor:
    """Convert (batched) roll, pitch, yaw angles (in radians) to rotation matrix.

    Adapted from siclib.utils.conversions.rad2rotmat.

    Args:
        roll: Roll angle in radians.
        pitch: Pitch angle in radians.
        yaw: Yaw angle in radians.
        ref_tensor: Reference tensor to determine the device.

    Returns:
        torch.Tensor: (3, 3) Rotation matrix.
    """
    Rx = torch.zeros(3, 3, device=device)
    Rx[0, 0] = 1
    Rx[1, 1] = cos(pitch)
    Rx[1, 2] = sin(pitch)
    Rx[2, 1] = -sin(pitch)
    Rx[2, 2] = cos(pitch)

    Ry = torch.zeros(3, 3, device=device)
    Ry[0, 0] = cos(yaw)
    Ry[0, 2] = -sin(yaw)
    Ry[1, 1] = 1
    Ry[2, 0] = sin(yaw)
    Ry[2, 2] = cos(yaw)

    Rz = torch.zeros(3, 3, device=device)
    Rz[0, 0] = cos(roll)
    Rz[0, 1] = sin(roll)
    Rz[1, 0] = -sin(roll)
    Rz[1, 1] = cos(roll)
    Rz[2, 2] = 1

    return Rz @ Rx @ Ry


class DatasetGenerator:
    """Dataset generator class to create datasets from panoramas."""

    default_conf = {
        "name": "???",
        # paths
        "base_dir": "???",
        "pano_dir": "${.base_dir}/panoramas",
        "pano_train": "${.pano_dir}/train",
        "pano_val": "${.pano_dir}/val",
        "pano_test": "${.pano_dir}/test",
        "im_dir": "${.base_dir}/${.name}",
        "im_train": "${.im_dir}/train",
        "im_val": "${.im_dir}/val",
        "im_test": "${.im_dir}/test",
        "train_h5": "${.im_dir}/train.h5",
        "val_h5": "${.im_dir}/val.h5",
        "test_h5": "${.im_dir}/test.h5",
        # general
        "images_per_pano": 16,
        "device": "cpu",
        "overwrite": False,
        "im_size": (640, 640),
        "resize_factor": {
            "type": "uniform",
            "options": {"loc": 1.2, "scale": 0.5},
        },
        # intrinsics
        "intrinsics": [
            {
                "cam_id": "simple_pinhole",
                "weight": 1.0,
                "vfov": {
                    "type": "uniform",
                    "options": {"loc": 20 * DEG2RAD, "scale": 85 * DEG2RAD},
                },
                "dist": None,
            }
        ],
    }

    def __init__(self, conf):
        """Init the class by merging and storing the config."""
        self.conf = OmegaConf.merge(
            OmegaConf.create(self.default_conf),
            OmegaConf.create(conf),
        )
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.conf)}")
        self.device = self.conf.device

        # for selecting the camera model during data generation
        self.cam_ids = [spec.cam_id for spec in self.conf.intrinsics]
        self.cam_weights = [spec.weight for spec in self.conf.intrinsics]
        self.cams = {cam_id: CameraFactory.create_from_id(cam_id) for cam_id in self.cam_ids}
        assert all(
            "simple_" not in id_ for id_ in self.cam_ids
        ), "Camera models must have 2 focal lengths."
        assert len(self.cam_ids) > 0, "No camera models specified."
        assert sum(self.cam_weights) == 1.0, "Camera weights do not sum to 1."
        self.cam_specs = {cam_spec.cam_id: cam_spec for cam_spec in self.conf.intrinsics}
        self.cam_selector = np.random.default_rng(0)

    def sample_value(self, param_conf: DictConfig, seed: int | str | None = None) -> float:
        """Sample a value from the specified distribution."""
        if param_conf.type == "fix":
            return float(param_conf.value)
        # fix seed for reproducibility
        generator = None
        if seed:
            if not isinstance(seed, (int, float)):
                seed = int(hashlib.sha256(seed.encode()).hexdigest(), 16) % (2**32)
            generator = np.random.default_rng(seed)
        sampler = getattr(scipy.stats, param_conf.type)
        return float(sampler.rvs(random_state=generator, **param_conf.options))

    def plot_distributions(self):
        """Plot parameter distributions."""
        # gather parameters across splits as a Dataframe
        rows = {"train": [], "val": [], "test": []}
        base_row = {
            "vfov": None,
            "roll": None,
            "pitch": None,
            "f": None,
            "k1": None,  # radial:1
            "alpha": None,  # eucm
            "beta": None,  # eucm
        }
        for split in ["train", "val", "test"]:
            rows_ = rows[split]
            with h5py.File(self.conf[f"{split}_h5"], "r") as h5_file:  # type:ignore
                for group in h5_file.values():
                    row_ = base_row.copy()
                    attrs = group.attrs
                    row_["vfov"] = RAD2DEG * attrs["vfov"]
                    row_["roll"] = RAD2DEG * attrs["roll"]
                    row_["pitch"] = RAD2DEG * attrs["pitch"]
                    row_["f"] = attrs["params"][0]
                    if attrs["cam_id"] == "radial:1":
                        row_["k1"] = attrs["params"][-1]
                    elif attrs["cam_id"] == "eucm":
                        row_["alpha"] = attrs["params"][-2]
                        row_["beta"] = attrs["params"][-1]
                    rows_.append(row_)
        dfs = {}
        for split in ["train", "val", "test"]:
            df = pd.DataFrame(rows[split])
            dfs[split] = df
            # dfs[split] = df.dropna(axis=1, how="all")  # drop columns with all NaNs

        # plot distributions for each parameter across splits
        nplots = max(len(df.columns) for df in dfs.values())
        fig, axs = plt.subplots(3, nplots, figsize=(5 * nplots, 15))
        for i, split in enumerate(["train", "val", "test"]):
            df = dfs[split]
            for j, param in enumerate(df.columns):
                if df[param].isnull().all():
                    continue
                axs[i, j].hist(df[param], bins=100)
                axs[i, j].set_xlabel(param)
                axs[i, j].set_ylabel(f"Count {split}")
        fig.tight_layout()
        fig.savefig(Path(self.conf.im_dir) / "distributions.png", bbox_inches="tight")
        # fig.savefig(Path(self.conf.im_dir) / "distributions.pdf", bbox_inches="tight")

        # plot pairwise scatter plots to check correlations
        comparisons = [
            ("roll", "pitch"),
            ("roll", "vfov"),
            ("pitch", "vfov"),
            ("vfov", "f"),
            ("vfov", "k1"),
            ("vfov", "alpha"),
            ("vfov", "beta"),
            ("alpha", "beta"),
        ]
        ncolumns = 3
        nrows = len(comparisons) // ncolumns + 1
        fig, axs = plt.subplots(nrows, ncolumns, figsize=(5 * ncolumns, 5 * nrows))
        for i, (param1, param2) in enumerate(comparisons):
            ax = axs[i // ncolumns, i % ncolumns]
            for split in ["train", "val", "test"]:
                df = dfs[split]
                ax.scatter(df[param1], df[param2], s=1, label=split)
            ax.set_xlabel(param1)
            ax.set_ylabel(param2)
            ax.legend()
        fig.tight_layout()
        fig.savefig(Path(self.conf.im_dir) / "distributions_scatter.png", bbox_inches="tight")
        # fig.savefig(Path(self.conf.im_dir) / "distributions_scatter.pdf", bbox_inches="tight")

    def get_safe_params(
        self, cam: BaseCamera, cam_spec: DictConfig, seed_id: tuple[str, int]
    ) -> tuple[Tensor, float]:
        """Sample intrinsics ensuring that they lead to unique image projections.

        To ensure unique projections, we follow [1] for the radial camera model and [2]
        for other radial distortion models.
        [1] On the Maximum Radius of Polynomial Lens Distortion. Leotta et al., WACV 2022.
        [2] The Double Sphere Camera Model. Usenko et al., 3DV 2018.
        """
        h, w = self.conf.im_size
        stem, i = seed_id
        cam_id = cam_spec.cam_id
        dist_specs = cam_spec.dist
        vfov = self.sample_value(cam_spec.vfov, f"{stem}vfov{i}")

        if cam_id == "pinhole":
            assert dist_specs is None, "pinhole model does not have distortion parameters."
            f = 0.5 * h / tan(0.5 * vfov)
            params = torch.tensor((f, f, 0.5 * w, 0.5 * h), device=self.device)
            return params, vfov

        elif cam_id == "radial:1":
            assert len(dist_specs) == 1, "radial:1 requires 1 distortion parameter spec."
            # follow GeoCalib's sampling procedure
            k1_hat_spec = dist_specs[0]
            k1_hat = self.sample_value(k1_hat_spec, f"{stem}k1_hat{i}")
            # NOTE: f is not given by this formula for radial cams--it's an approximation.
            f = 0.5 * h / tan(0.5 * vfov)
            k1 = k1_hat * f / h
            # guard for focal
            max_r_needed_im = 0.5 * hypot(h, w)
            max_r_allowed = float("inf") if k1 >= 0 else 1 / sqrt(-3 * k1)
            min_f_allowed = max_r_needed_im / (max_r_allowed * (1 + k1 * max_r_allowed**2))
            if f < min_f_allowed:
                logger.debug(f"[radial:1] {stem}_{i} clamps: focal {f:.2f} -> {min_f_allowed:.2f}")
                f = min_f_allowed
            # final params and obtain *real* fov
            params = torch.tensor((f, f, 0.5 * w, 0.5 * h, k1), device=self.device)
            vfov, valid = cam.get_vfov(params, h)
            assert valid, "Invalid vfov estimation"
            return params, vfov.item()

        elif cam_id == "eucm":
            assert len(dist_specs) == 2, "eucm requires 2 distortion parameter specs."
            alpha_spec, beta_spec = dist_specs
            assert alpha_spec.name == "alpha" and beta_spec.name == "beta"
            alpha = self.sample_value(alpha_spec, f"{stem}alpha{i}")
            beta = self.sample_value(beta_spec, f"{stem}beta{i}")
            # compute f from vFoV: f = (h/2) / eucm_dist(vFoV/2, alpha, beta)
            R, Z = sin(0.5 * vfov), cos(0.5 * vfov)
            f = f0 = 0.5 * h * (alpha * sqrt(beta * R**2 + Z**2) + (1 - alpha) * Z) / R
            # guard for focal
            max_r2_needed_im = 0.25 * (h**2 + w**2)
            min_f2_allowed = 0 if alpha <= 0.5 else max_r2_needed_im * beta * (2 * alpha - 1)
            min_f_allowed = sqrt(min_f2_allowed)
            if f < min_f_allowed:
                f = min_f_allowed
                a, b = alpha, beta
                # recalculate vFoV using closed-form unprojection
                my2 = (0.5 * h / f) ** 2
                mz = (1 - b * a**2 * my2) / (a * sqrt(1 - (2 * a - 1) * b * my2) + (1 - a))
                vfov_new = 2 * acos(mz / sqrt(my2 + mz**2))
                logger.debug(
                    f"[eucm] {stem}_{i} clamps: focal {f0:.2f} -> {f:.2f}, "
                    f"vfov: {vfov*180/pi:.2f} -> {vfov_new*180/pi:.2f}"
                )
                vfov = vfov_new
            # final params
            params = torch.tensor((f, f, 0.5 * w, 0.5 * h, alpha, beta), device=self.device)
            return params, vfov

        else:
            raise NotImplementedError

    def generate_images_from_pano(self, h5file: h5py.File, pano_path: Path, out_dir: Path):
        """Generate perspective images from a single panorama."""
        stem = pano_path.stem
        h, w = self.conf.im_size
        dev = self.device

        pano = torch.tensor(np.array(Image.open(pano_path)), device=dev) / 255
        pano_img = pano.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        h_pano, w_pano = pano_img.shape[-2:]

        yaws = torch.linspace(0, 2 * pi, self.conf.images_per_pano + 1, device=dev)[:-1]
        cam_ids = self.cam_selector.choice(
            self.cam_ids, size=self.conf.images_per_pano, p=self.cam_weights
        )
        for i, cam_id in enumerate(cam_ids):
            yaw = yaws[i]
            cam: BaseCamera = self.cams[cam_id]
            cam_spec = self.cam_specs[cam_id]

            # sample {in,ex}trinsics and resize factor
            res_fac = self.sample_value(self.conf.resize_factor, f"{stem}resize_factor{i}")
            roll = self.sample_value(self.conf.roll, f"{stem}roll{i}")
            pitch = self.sample_value(self.conf.pitch, f"{stem}pitch{i}")
            params, vfov = self.get_safe_params(cam, cam_spec, (stem, i))

            ## extract image from pano
            # following Geocalib, quote: "resize the panorama such that its fov has the
            # same height as the image"
            scale = pi / vfov * h / h_pano * res_fac
            h_pano_new, w_pano_new = (int(h_pano * scale), int(w_pano * scale))
            resized_pano = F.interpolate(
                pano_img,
                size=(h_pano_new, w_pano_new),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0, 1)

            # unit bearings in the camera's reference system
            bearings, valid = cam.ray_grid(h, w, params, offset=0.5)
            assert valid is None or valid.all(), "Invalid bearings"
            bearings = bearings.reshape(h * w, 3)
            # rotate according to extrinsics
            rotated_bearings = bearings @ rad2rotmat(roll, pitch, yaw.item(), dev)
            # spherical coordinates
            lon = torch.atan2(rotated_bearings[:, 0], rotated_bearings[:, 2])
            lat = torch.atan2(
                rotated_bearings[:, 1],
                torch.linalg.norm(rotated_bearings[:, [0, 2]], dim=-1),
            )

            # project and sample from panorama
            min_lon, max_lon = -pi, pi
            min_lat, max_lat = -pi / 2, pi / 2
            min_x, max_x = 0, w_pano_new - 1
            min_y, max_y = 0, h_pano_new - 1
            # map spherical coords. to panoramic coords.
            nx = (lon - min_lon) / (max_lon - min_lon) * (max_x - min_x) + min_x
            ny = (lat - min_lat) / (max_lat - min_lat) * (max_y - min_y) + min_y
            mapx = nx.reshape((1, h, w))
            mapy = ny.reshape((1, h, w))
            grid = torch.stack((mapx, mapy), dim=-1)  # (1, H, W, 2)
            # normalize to [-1, 1]
            grid = 2 * grid / torch.tensor([w_pano_new - 1, h_pano_new - 1], device=dev) - 1
            im = F.grid_sample(
                resized_pano,
                grid,
                align_corners=False,
                padding_mode="border",
            )[0].clamp(0, 1)  # (3, H, W) # fmt:skip

            # discard images with >=1% black pixels
            valid = torch.mean((im.sum(0) == 0).float()) < 0.01
            fname = f"{stem}_{i}.jpg"
            if not valid:
                logger.debug(f"[{cam_id}] {fname} has too many black pixels.")
                continue
            # save params
            grp = h5file.create_group(fname)
            grp.attrs["h"] = h
            grp.attrs["w"] = w
            grp.attrs["roll"] = roll  # only true for panos aligned with gravity
            grp.attrs["pitch"] = pitch  # only true for panos aligned with gravity
            grp.attrs["vfov"] = vfov
            grp.attrs["resize_factor"] = res_fac
            grp.attrs["cam_id"] = str(cam_id)
            grp.attrs["params"] = params.cpu().numpy()
            # save image
            Image.fromarray((255 * im.permute(1, 2, 0)).byte().cpu().numpy()).save(out_dir / fname)

    def generate_split(self, split: str):
        """Generate a single split of a dataset."""
        h5_path = Path(self.conf[f"{split}_h5"])  # type: ignore
        if h5_path.exists() and not self.conf.overwrite:  # type: ignore
            logger.info(f"Dataset for {split}: {str(h5_path)} already exists.")
            return

        out_dir = Path(self.conf[f"im_{split}"])  # type:ignore
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing images to {str(out_dir)}")

        panorama_paths = sorted(
            [
                path
                for path in Path(self.conf[f"pano_{split}"]).glob("*")  # type: ignore
                if not path.name.startswith(".")
            ]
        )
        with h5py.File(h5_path, "w") as h5_file:
            for pano_path in tqdm(panorama_paths):
                self.generate_images_from_pano(h5_file, pano_path, out_dir)

    def generate_dataset(self):
        """Generate all splits of a dataset."""
        out_dir = Path(self.conf.im_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
        OmegaConf.save(self.conf, out_dir / "config.yaml")

        for split in ["train", "val", "test"]:
            self.generate_split(split=split)

        for split in ["train", "val", "test"]:
            with h5py.File(self.conf[f"{split}_h5"], "r") as h5_file:  # type:ignore
                total = sum(1 for _ in h5_file)
            logger.info(f"Generated {total} {split} images.")

        self.plot_distributions()


@hydra.main(version_base=None, config_path="configs", config_name="openpano_v2_radial")
def main(cfg: DictConfig) -> None:
    """Run dataset generation."""
    generator = DatasetGenerator(conf=cfg)
    generator.generate_dataset()


if __name__ == "__main__":
    main()
