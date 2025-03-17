import logging
from functools import partial
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, NamedTuple

import h5py
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data.distributed
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader

from anycalib.cameras.base import BaseCamera
from anycalib.cameras.factory import CameraFactory
from siclib.datasets.augmentations import IdentityAugmentation, augmentations
from siclib.datasets.base_dataset import BaseDataset, collate, worker_init_fn
from siclib.utils.conversions import fov2focal
from siclib.utils.image_rays import ImagePreprocessor, load_image
from siclib.utils.tools import fork_rng

logger = logging.getLogger(__name__)

# mypy: ignore-errors


MAX_NPARAMS = BaseCamera.MAX_NPARAMS

DataPoint = NamedTuple(
    "DataPoint",
    [
        ("name", str),
        ("file_name", str),
        ("h", int),
        ("w", int),
        ("cam_id", str),
        ("params", np.ndarray),
        # NOTE: if "params" is Tensor, this will likely lead to "RuntimeError: Too many open files." in distributed mode
        # (https://pytorch.dev.org.tw/docs/2.3/multiprocessing.html#file-descriptor-file-descriptor)
        # ("params", Tensor),
    ],
)


def load_csv_openpano_format(
    csv_file: Path, img_root: Path, simple: bool = False
) -> list[DataPoint]:
    datapoints = []
    df = pd.read_csv(csv_file)
    is_radial = "radial" in csv_file.parent.name
    cam_id = "radial:1" if is_radial else "pinhole"
    if simple:
        cam_id = "simple_" + cam_id
    for _, row in df.iterrows():
        h = row["height"]
        w = row["width"]

        f = row.get("focal", fov2focal(torch.tensor(row["vfov"]), h))
        cx = row.get("px", 0.5 * w)
        cy = row.get("py", 0.5 * h)
        k1 = row.get("k1", None)  # GeoCalib's radial datasets only have k1 (k2=0)
        assert not is_radial or k1 is not None, "k1 must be provided for radial cameras"

        params = np.concatenate(
            (
                (f,) if simple else (f, f),
                (cx, cy),
                (k1,) if is_radial else (),  # type: ignore # k1 may be 0 for pinhole
            ),
            dtype=np.float32,
        )

        datapoints.append(
            DataPoint(
                name=row["fname"],
                file_name=str(img_root / row["fname"]),
                h=h,
                w=w,
                cam_id=cam_id,
                params=params,
            )
        )
    return datapoints


def load_h5_anycalib_format(h5_file: Path, img_root: Path) -> list[DataPoint]:
    with h5py.File(h5_file, "r", libver="latest") as f:
        datapoints = [
            DataPoint(
                name=name,
                file_name=str(img_root / name),
                h=int(group.attrs["h"]),
                w=int(group.attrs["w"]),
                cam_id=group.attrs["cam_id"],
                params=group.attrs["params"].astype(np.float32),
            )
            for name, group in f.items()
        ]
    return datapoints


class EditableConfig:
    """Simple context manager that ensures a config is editable inside the block."""

    def __init__(self, conf):
        self.conf = conf
        self.readonly = OmegaConf.is_readonly(conf)

    def __enter__(self):
        OmegaConf.set_readonly(self.conf, False)

    def __exit__(self, exc_type, exc_value, traceback):
        OmegaConf.set_readonly(self.conf, self.readonly)


def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple


def element_from_rand_idx(arr, rand):
    """Return an element from an array given a random index in [0, 1).

    Args:
        arr (np.ndarray): array of elements.
        rand (float): random number in [0, 1).

    Returns:
        Any: element from the array.
    """
    idx = int(rand * len(arr))
    assert 0 <= idx <= len(arr), f"{idx} not in [0, {len(arr)})"
    # clip the index to len(arr) - 1 due to floating-point rounding errors
    return arr[min(idx, len(arr) - 1)]


def map_rand_to_range(interval, rand):
    """Map a random number in [0, 1) to a value in the interval.

    Args:
        interval (tuple): (lim_inf, lim_sup) of the interval.
        rand (float): random number in [0, 1).

    Returns:
        float: value in the interval.
    """
    return interval[0] + rand * (interval[1] - interval[0])


class BatchedRandomSampler:
    """Random sampling of indices along a finite number (>=0) of random numbers that are the
    same in each batch.

    The index returned is a tuple (sample_idx, *[random_float, ...]) where each random_float is
    sampled from a uniform distribution in [0, 1) and is the same for each batch.

    This sampler class is adapted from DUSt3R [1]:
    https://github.com/naver/dust3r/blob/main/dust3r/datasets/base/batched_sampler.py

    [1] DUSt3R: Geometric 3d vision made easy, S. Wang et al., CVPR 2024.
    """

    def __init__(self, dataset, batch_size, npools, drop_last=False):
        self.batch_size = batch_size
        self.npools = npools

        # distributed sampler
        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0
        self.epoch = None

        self.len_dataset = N = len(dataset)
        self.total_size = round_by(N, batch_size * self.world_size) if drop_last else N
        assert self.world_size == 1 or drop_last, "must drop the last batch in distributed mode"

        logger.info(f"BatchedRandomSampler instantiate with {self.world_size} GPUs.")

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        # prepare RNG
        if self.epoch is None:
            assert (
                self.world_size == 1 and self.rank == 0
            ), "use set_epoch() if distributed mode is used"
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        rng = np.random.default_rng(seed=seed)

        # random indices (will restart from 0 if not drop_last)
        sample_idxs = np.arange(self.total_size)
        rng.shuffle(sample_idxs)

        # random numbers in [0, 1) shared across batches
        n_batches = (self.total_size + self.batch_size - 1) // self.batch_size
        rns = np.repeat(rng.random((n_batches, self.npools)), self.batch_size, axis=0)
        rns = rns[: self.total_size]

        # Distributed sampler: we select a subset of batches
        # make sure the slice for each node is aligned with batch_size
        size_per_proc = self.batch_size * (
            (self.total_size + self.world_size * self.batch_size - 1)
            // (self.world_size * self.batch_size)
        )
        sample_idxs = sample_idxs[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]
        rns = rns[self.rank * size_per_proc : (self.rank + 1) * size_per_proc]

        if self.npools > 0:
            yield from ((idx, *rn) for idx, rn in zip(sample_idxs, rns))
        else:
            yield from (idx for idx in sample_idxs)


class SimpleDataset(BaseDataset):
    """Dataset for images created with 'create_dataset_from_pano.py'.

    NOTE on key "im_geom_transform": It is an *optional* dict that determines the geometric
    transformations applied to the images during training and eval. During test, they represent the
    configuration used during training and are used to resize the test images to the closest training transform.
    Key-value pairs:
        - "aspect_ratio":  (defined as H / W)
            * None to maintain the aspect-ratio of the input image.
            * tuple[float, float] to sample, during training, an aspect ratio in the range
                (lim_inf, lim_sup).
            * str: to sample aspect ratios from either the class attribute with the same name,
                or from a file containing the aspect ratios. The file is expected to be a .npy file.
        - "change_pixel_ar": (bool) if True, the aspect ratio refer to the pixel aspect ratio
            (it leads to changes in the ratio of focal lengths fy/fx).
        - "resolution":  (defined as H * W)
            * None to maintain the resolution of the input image.
            * float to set a fixed resolution.
            * tuple[float, float] to sample, during training, a resolution in the range
                (lim_inf, lim_sup).
        - "crop":  (to modify the location of the principal point)
            * None to do not crop the input image.
            * float in [0, 1] corresponding to normalized image dimensions. Two crop factors,
                say, cf_x and cf_y, will be sampled in [-crop, crop].
                If cf_i < 0, the image will be cropped from the left (width) or top (height).
                If cf_i > 0, the image will be cropped from the right (width) or bottom (height).

    If None, then, the size (and resolution) of the images is defined by the key "resize"
    of the dictionary under the "preprocessing" key. The principal point will not be
    modified.
    """

    default_conf = {
        # paths
        "dataset_dir": "???",
        "train_img_dir": "${.dataset_dir}/train",
        "val_img_dir": "${.dataset_dir}/val",
        "test_img_dir": "${.dataset_dir}/test",
        "train_csv": "${.dataset_dir}/train.csv",
        "train_h5": "${.dataset_dir}/train.h5",
        "val_csv": "${.dataset_dir}/val.csv",
        "val_h5": "${.dataset_dir}/val.h5",
        "test_csv": "${.dataset_dir}/test.csv",
        "test_h5": "${.dataset_dir}/test.h5",
        # data options:
        "use_prior_cxcy": False,
        "simple_if_possible": False,
        "to_closest_train_size": False,
        "cam_id": None,  # None -> same as GT's cam_id
        # image options
        "grayscale": False,
        "im_geom_transform": {
            "aspect_ratio": (0.5, 2.0),  # None | 2-tuple | 2-list | attribute | file name
            "change_pixel_ar": False,  # bool
            "resolution": 320**2,  # None | float | 2-tuple
            "crop": None,  # None or float
            "edit_prob": 1.0,  # probability to apply change_pixel_ar and crop (if they are not None)
        },
        "preprocessing": ImagePreprocessor.default_conf,
        "augmentations": {"name": "geocalib", "verbose": False},
        "p_rotate": 0.0,  # probability to rotate image by +/- 90°
        "reseed": False,
        "seed": 0,
        # data loader options
        "num_workers": 8,
        "prefetch_factor": 2,
        "train_batch_size": 32,
        "val_batch_size": 32,
        "test_batch_size": 32,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split: str) -> "_SimpleDataset":
        """Return a dataset for a given split."""
        return _SimpleDataset(self.conf, split)  # type: ignore

    def get_data_loader(self, split, shuffle=None, pinned=False, distributed=False):
        """Return a data loader for a given split."""
        assert split in ("train", "val", "test")
        dataset = self.get_dataset(split)
        try:
            batch_size = self.conf[f"{split}_batch_size"]  # type: ignore
        except omegaconf.MissingMandatoryValue:
            batch_size = self.conf.batch_size
        num_workers = self.conf.get("num_workers", batch_size)  # type: ignore

        # get sampler
        drop_last = split != "test" and distributed
        if split != "test" and dataset.npools > 0:
            sampler = BatchedRandomSampler(dataset, batch_size, dataset.npools, drop_last=True)
        elif distributed:
            shuffle = False
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=drop_last)
        else:
            sampler = None
            if shuffle is None:
                shuffle = split == "train" and self.conf.shuffle_training

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=pinned,
            collate_fn=collate,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            prefetch_factor=self.conf.prefetch_factor,
        )


class _SimpleDataset(torch.utils.data.Dataset):
    """Dataset for dataset for images created with 'create_dataset_from_pano.py'."""

    OPENPANO_FORMAT = {
        "openpano",
        "openpano_v2",
        "openpano_v2_clean",
        "openpano_radial",
        "tartanair",
        "megadepth2k",
        "megadepth2k-radial",
        "stanford2d3d",
        "lamar2k",
    }

    ANYCALIB_FORMAT = {
        "tartanair_edit",
        "megadepth2k_edit",
        "megadepth2k-radial_edit",
        "stanford2d3d_edit",
        "lamar2k_edit",
        "openpano_v2_radial",
        "openpano_v2_dist",
        "openpano_v2_gen",
        "scannetpp2k",
        "monovo2k",
    }

    # based on DSINE's aspect ratios
    ASPECT_RATIOS: tuple[float, ...] = (  # H / W
        (320 / 960),
        (384 / 800),
        (448 / 672),
        (480 / 640),
        (512 / 608),
        (576 / 544),
        (640 / 480),
        (704 / 448),
        (768 / 416),
        (832 / 384),
        (896 / 352),
        (960 / 320),
    )

    def __init__(self, conf: DictConfig, split: str):
        """Initialize the dataset."""
        self.conf = conf
        self.split = split
        self.is_test = is_test = split == "test"
        self.img_dir = Path(conf.get(f"{split}_img_dir"))
        self.dset_name = Path(conf.dataset_dir).name

        # preprocessing: random image center only in training:
        if is_test and self.conf.preprocessing.random_center:
            logger.warning("random_center is only allowed in training or val. Setting it to False.")
            with EditableConfig(self.conf):
                self.conf.preprocessing.random_center = False
        self.preprocessor = ImagePreprocessor(conf.preprocessing)
        self.edge_divisible_by = self.preprocessor.conf.edge_divisible_by

        # load image information
        if self.dset_name in self.OPENPANO_FORMAT:
            assert f"{split}_csv" in conf, f"Missing {split}_csv in conf"
            infos_path = Path(self.conf.get(f"{split}_csv"))
            self.datapoints = load_csv_openpano_format(
                infos_path, self.img_dir, self.conf.simple_if_possible
            )
        elif self.dset_name in self.ANYCALIB_FORMAT:
            assert f"{split}_h5" in conf, f"Missing {split}_h5 in conf"
            self.datapoints = load_h5_anycalib_format(
                Path(self.conf.get(f"{split}_h5")), self.img_dir
            )
        else:
            raise ValueError(f"Unknown dataset format: {self.dset_name}")

        # define augmentations
        aug_name = conf.augmentations.name
        assert (
            aug_name in augmentations.keys()
        ), f'{aug_name} not in {" ".join(augmentations.keys())}'
        if self.split == "train":
            self.augmentation = augmentations[aug_name](conf.augmentations)
        else:
            self.augmentation = IdentityAugmentation()

        # get rngs or fixed values for: aspect_ratio | resolution | ppoint displacement
        geom_tfs: DictConfig | None = conf.im_geom_transform
        self.change_pixel_ar = False if is_test or geom_tfs is None else geom_tfs.change_pixel_ar
        self.rng_tfs_order = rng_tfs_order = []
        if geom_tfs is not None:
            # resolution strategy
            rs = geom_tfs.resolution
            if rs is None:  # maintain resolution
                self.res_pool = self.res_sampler_fn = None
            elif isinstance(rs, (int, float)):  # fixed resolution
                self.res_pool = rs
                self.res_sampler_fn = None
            elif isinstance(rs, (ListConfig, tuple, list)):  # sample resolution
                assert rs[0] <= rs[1], f"({rs[0]=}) > ({rs[1]=})"
                assert len(rs) == 2, f"Resolution range must have 2 elements. Got {len(rs)=}"
                self.res_pool = tuple(rs)
                self.res_sampler_fn = partial(map_rand_to_range, rs)
                rng_tfs_order.append("res")
            else:
                raise ValueError(f"Unknown resolution strategy: {rs}")

            # aspect ratio strategy
            ar = geom_tfs.aspect_ratio
            if ar is None:  # maintain aspect ratio
                self.ar_pool = self.ar_sampler_fn = None
            elif isinstance(ar, str):  # sample aspect ratio from file or class attribute
                if ar.endswith(".npy"):
                    self.ar_pool = ar_pool = np.load(Path(__file__).parents[2] / ar)
                elif hasattr(self, ar):
                    self.ar_pool = ar_pool = np.array(getattr(self, ar))
                else:
                    raise ValueError(
                        f"Unknown aspect ratio strategy: {ar}. Only .npy files or class attributes are supported."
                    )
                self.ar_sampler_fn = partial(element_from_rand_idx, ar_pool)
                rng_tfs_order.append("ar")
            elif isinstance(ar, (ListConfig, tuple, list)):  # sample aspect ratio
                assert ar[0] <= ar[1], f"({ar[0]=}) > ({ar[1]=})"
                assert len(ar) == 2, f"Aspect ratio range must have 2 elements. Got {len(ar)=}"
                self.ar_pool = ar = tuple(ar)
                self.ar_sampler_fn = partial(map_rand_to_range, ar)
                rng_tfs_order.append("ar")
            else:
                raise ValueError(f"Unknown aspect ratio strategy: {ar}")

            # principal point displacement
            max_crop = geom_tfs.crop
            if max_crop is None:  # maintain principal point
                self.crop_pool = self.crop_sampler_fn = None
            elif isinstance(max_crop, (int, float)):  # sample principal point displacement
                assert 0 <= max_crop <= 1, f"Invalid max_crop: {max_crop}. Must be in [0, 1]."
                self.crop_pool = (-max_crop, max_crop)
                self.crop_sampler_fn = partial(map_rand_to_range, self.crop_pool)
                rng_tfs_order.append("crop_x")
                rng_tfs_order.append("crop_y")
            else:
                raise ValueError(f"Unknown principal point displacement strategy: {max_crop}")

            # probability to apply change_pixel_ar (when True) and crop
            assert 0 <= geom_tfs.edit_prob <= 1, f"Invalid {self.edit_prob=}: Not in [0, 1]."
            if not self.change_pixel_ar and self.crop_pool is None or is_test:
                self.edit_prob_rng = self.edit_prob = None
            else:
                self.edit_prob = geom_tfs.edit_prob
                self.edit_prob_rng = np.random.default_rng(seed=conf.seed)

        else:
            self.ar_sampler_fn = self.res_sampler_fn = self.crop_sampler_fn = None
            self.res_pool = self.ar_pool = self.crop_pool = None
            self.edit_prob_rng = self.edit_prob = None

        # number of pools from which to sample during training/val
        self.npools = len(rng_tfs_order)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        if not self.conf.reseed:
            return self.getitem(idx)
        with fork_rng(self.conf.seed + idx, False):
            return self.getitem(idx)

    def _read_image(
        self,
        datapoint: DataPoint,
        target_size: tuple[int, int] | None = None,
        crop: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        """Read and preprocess image and intrinsics.

        NOTE: This class, and the methods it calls, assume that the origin (0, 0) of the image
        is located at the top-left *corner* of the top-left pixel. This mainly affects the
        convention for defining the principal point.
        """

        # load image as uint8 and HWC for augmentation
        path = Path(datapoint.file_name)
        image = load_image(path, self.conf.grayscale, return_tensor=False)
        # augment image
        image = self.augmentation(image, return_tensor=True)  # (3, H, W)
        h, w = image.shape[-2:]
        assert (w, h) == (datapoint.w, datapoint.h), f"{(h, w)}, {datapoint.h, datapoint.w}"

        # transform to closest training image size based on aspect ratio and resolution
        if self.is_test and self.conf.to_closest_train_size:
            # target aspect ratio
            arp = self.ar_pool
            ar = h / w
            if isinstance(arp, np.ndarray):
                ar = float(arp[np.abs(arp - h / w).argmin()])
            elif isinstance(arp, tuple):
                # if aspect ratio not into in trained range, set to closest in the pool
                if ar < arp[0]:
                    ar = arp[0]
                elif ar > arp[1]:
                    ar = arp[1]
            elif arp is not None:
                raise ValueError(f"Unknown aspect ratio pool: {arp}")
            # target resolution
            resp = self.res_pool
            res = h * w
            if isinstance(resp, tuple):
                res = h * w if resp[0] <= h * w <= resp[1] else mean(resp)
            elif isinstance(resp, (int, float)):
                res = resp
            elif resp is not None:
                raise ValueError(f"Unknown resolution pool: {resp}")
            target_size = self.compute_target_size(h, w, res, ar)
        elif self.is_test:
            assert target_size is None, f"{target_size=}"
        assert not self.is_test or (
            crop is None and not self.change_pixel_ar
        ), f"{crop=}, {self.change_pixel_ar=}"

        # skip image crop/pixel-ar modifications during training/val?
        change_pix_ar = self.change_pixel_ar
        if self.edit_prob_rng is not None:
            assert self.edit_prob is not None
            allow_edit = self.edit_prob_rng.random() < self.edit_prob
            change_pix_ar = self.change_pixel_ar and allow_edit
            crop = crop if allow_edit else None

        # preprocess (resize + crop)
        data = self.preprocessor(image, target_size, crop, change_pix_ar)

        conf_id = self.conf.cam_id
        assert conf_id is None or (("simple" in conf_id) == ("simple" in datapoint.cam_id))  # fmt: skip
        cam = CameraFactory.create_from_id(datapoint.cam_id)
        params = torch.tensor(datapoint.params, dtype=torch.float32, device=image.device)
        # adapt intrinsics to the resized/cropped image
        params = cam.scale_and_shift(params, data["scale_xy"], data["shift_xy"], copy=False)

        # *pixel* aspect ratio
        if "f" in cam.PARAMS_IDX:
            pix_ar = np.array(1.0, dtype=np.float32)
        elif "fx" in cam.PARAMS_IDX:
            pix_ar = params[cam.PARAMS_IDX["fy"]] / params[cam.PARAMS_IDX["fx"]]
        else:
            raise ValueError(f"Can't infer pixel aspect for camera {cam.id}.")
        cxcy_gt = cam.params_to_dict(params)["c"]

        # get ground-truth rays (set offset to 0.5 to get rays at the *center* of the pixels)
        h, w = data["image"].shape[-2:]
        rays, valid = cam.ray_grid(h, w, params, 0.5)
        rays = rays.view(h * w, 3)
        valid = rays.new_ones(h * w, dtype=torch.bool) if valid is None else valid.view(-1)

        # pad intrinsics to be stackable
        params = torch.cat((params, params.new_zeros(MAX_NPARAMS - len(params))))
        params_gt = torch.cat(
            (torch.tensor(datapoint.params), params.new_zeros(MAX_NPARAMS - len(datapoint.params)))
        )
        # principal point prior
        cxcy = cxcy_gt if self.conf.use_prior_cxcy else None

        return {
            "name": datapoint.name,
            "path": str(path),
            "rays": rays,
            "rays_mask": valid,
            "intrinsics": params,
            "intrinsics_gt": params_gt,
            "cam_id": cam.id if self.conf.cam_id is None else self.conf.cam_id,
            "cam_id_gt": datapoint.cam_id,
            "cxcy": cxcy,
            "pix_ar": pix_ar,
            "cxcy_gt": cxcy_gt,
            **data,
        }

    def getitem(self, idx: int | tuple) -> dict[str, Any]:
        """Get and preprocess datapoint."""
        if isinstance(idx, int):  # no specific target image size or crop
            dpoint = self.datapoints[idx]
            target_size = None
            crop = None

        elif isinstance(idx, tuple):
            idx_, *rands = idx
            dpoint = self.datapoints[idx_]
            rand_cfg = {k: v for k, v in zip(self.rng_tfs_order, rands)}

            if "ar" in rand_cfg:
                assert callable(self.ar_sampler_fn)
                target_ar = self.ar_sampler_fn(rand_cfg["ar"])
            else:
                target_ar = None

            if "res" in rand_cfg:
                assert callable(self.res_sampler_fn)
                target_res = self.res_sampler_fn(rand_cfg["res"])
            else:
                assert isinstance(self.res_pool, (int, float)) or self.res_pool is None
                target_res = self.res_pool

            target_size = self.compute_target_size(dpoint.h, dpoint.w, target_res, target_ar)

            if "crop_x" in rand_cfg and "crop_y" in rand_cfg:
                assert callable(self.crop_sampler_fn)
                crop = (  # sample crops for both height and width
                    int(self.crop_sampler_fn(rand_cfg["crop_y"]) * dpoint.h),
                    int(self.crop_sampler_fn(rand_cfg["crop_x"]) * dpoint.w),
                )
            else:
                crop = None

        else:
            raise ValueError(f"idx can be either int or tuple. Got {idx}")

        data = self._read_image(dpoint, target_size, crop)
        return data

    def compute_target_size(
        self, ho: int, wo: int, target_res: float | None, target_ar: float | None
    ) -> tuple[int, int]:
        """Compute the target image size given the target resolution and aspect ratio."""
        if target_res is None:
            target_res = ho * wo
        if target_ar is None:
            target_ar = ho / wo

        w = sqrt(target_res / target_ar)
        h = target_ar * w
        # closest image size satisfying `edge_divisible_by` constraint
        div = self.edge_divisible_by
        target_size = (
            (int(h), int(w)) if div is None else (round(h / div) * div, round(w / div) * div)
        )
        return target_size


if __name__ == "__main__":
    import argparse
    from copy import deepcopy

    import torch.nn.functional as F

    from anycalib.visualization.viz_batch import make_batch_figures

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="simple_dataset_rays")
    parser.add_argument("--data_dir", type=str, default="data/openpano/openpano")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--n_rows", type=int, default=4)
    args = parser.parse_intermixed_args()

    torch.set_grad_enabled(False)

    dconf = SimpleDataset.default_conf
    dconf["name"] = args.name
    dconf["num_workers"] = 0
    dconf["prefetch_factor"] = None

    dconf["dataset_dir"] = args.data_dir
    dconf[f"{args.split}_batch_size"] = args.n_rows

    # dconf["im_geom_transform"] = None
    dconf["im_geom_transform"] = {
        ## aspect ratio
        "aspect_ratio": (0.5, 2.0),
        # "aspect_ratio": "ASPECT_RATIOS",
        # "aspect_ratio": "data/megascenes/aspect_ratios.npy",
        ## resolution
        "resolution": 320**2,
        # "resolution": (100_000, 200_000),
        ## principal point displacement
        "crop": 0.5,
        # "crop": None,
        "change_pixel_ar": True,
        "edit_prob": 0.9,
    }
    dconf["to_closest_train_size"] = True
    dconf["preprocessing"]["edge_divisible_by"] = 14

    dataset = SimpleDataset(dconf)
    loader = dataset.get_data_loader(args.split, args.shuffle)

    with fork_rng(seed=42):
        for i, data in enumerate(loader):
            # for k, v in data.items():
            #     if isinstance(v, torch.Tensor):
            #         print(f"{k}: {v.dtype}, {v.shape}")
            #     else:
            #         print(f"{k}: {v}")

            b, _, h, w = data["image"].shape
            print(f"resolution: {h}x{w} ({h * w})")
            print(f"aspect ratio: {h / w:.3f}")
            # print(f"intrinsics: {data['intrinsics']}")
            # print(f"scale: {data['scale_xy']}")
            print(f"shift: {data['shift_xy']}")

            pred = deepcopy(data)
            pred["rays"] = F.normalize(pred["rays"] + 1e-3 * torch.randn_like(pred["rays"]), dim=-1)
            pred["log_covs"] = 0.01 * torch.randn_like(pred["rays"][..., :2])
            pred["pix_ar_map"] = pred["image"].new_ones(b, 2, h, w)
            pred["radii"] = torch.linalg.norm(
                BaseCamera.pixel_grid_coords(h, w, pred["intrinsics"], 0.5)
                - data["cxcy_gt"][:, None, None],
                dim=-1,
            )

            print(data["path"])
            fig = make_batch_figures(pred, data, n_pairs=args.n_rows)
            fig["radial"].savefig(f"radial_{i}.png")
            fig["errors"].savefig(f"errors_{i}.png")
            fig["editmaps"].savefig(f"editmaps_{i}.png")
            if i == 5:
                break
