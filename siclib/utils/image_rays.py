"""Image preprocessing utilities."""

import random
from pathlib import Path

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from omegaconf import OmegaConf
from torch import Tensor
from torchvision.transforms import InterpolationMode

from siclib.utils.tensor import fit_features_to_multiple

# mypy: ignore-errors

TORCHVISION_INTERPOLATIONS = {
    "nearest": InterpolationMode.NEAREST,
    "nearest_exact": (
        InterpolationMode.NEAREST_EXACT
        if hasattr(InterpolationMode, "NEAREST_EXACT")
        else InterpolationMode.NEAREST
    ),
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
}


class ImagePreprocessor:
    """Preprocess images for calibration."""

    default_conf = {
        "edge_divisible_by": None,
        "resize": None,  # target edge length, None for no resizing
        "side": "short",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "resize_backend": "kornia",  # torchvision, kornia
        "square_crop": False,
        "random_center": False,
        "random_center_scale": 0.015,
    }

    def __init__(self, conf):
        """Initialize the image preprocessor."""

        default_conf = OmegaConf.create(self.default_conf)
        OmegaConf.set_struct(default_conf, True)
        self.conf = OmegaConf.merge(default_conf, conf)

        if self.conf.resize_backend not in ("kornia", "torchvision"):
            raise ValueError(
                "resize_backend must be one of 'kornia' or 'torchvision'. Got "
                f"'{self.conf.resize_backend}'."
            )
        if self.conf.side not in ("short", "long", "vert", "horz"):
            raise ValueError(
                f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{self.conf.side}'"
            )

        if self.conf.resize_backend == "torchvision":
            conf.interpolation = TORCHVISION_INTERPOLATIONS[conf.interpolation]

    def __call__(
        self,
        im: Tensor,
        target_size: tuple[int, int] | None = None,
        crop: tuple[int, int] | None = None,
        change_pix_ar: bool = False,
    ) -> dict:
        """Rescaling and cropping image preprocessing."""
        assert im.dim() == 3, f"Expected 3D tensor, got {im.dim()} with {im.shape=}"
        if self.conf.edge_divisible_by is not None and target_size is not None:
            div = self.conf.edge_divisible_by
            assert (
                target_size[0] % div == 0 == target_size[1] % div
            ), f"Target resolution {target_size} must be divisible by {div}."

        ho, wo = h, w = im.shape[-2:]
        # init transformation of intrinsics
        scale_xy = torch.ones(2, device=im.device)  # for focals fxfy and cxcy
        shift_xy = torch.zeros(2, device=im.device)  # for principal point cxcy

        if crop is not None:
            cropy, cropx = crop
            if cropx < 0:  # crop from left
                im = im[..., -cropx:]
                shift_xy[0] += cropx
            elif cropx > 0:  # right
                im = im[..., :-cropx]
            if cropy < 0:  # top
                im = im[..., -cropy:, :]
                shift_xy[1] += cropy
            elif cropy > 0:  # bottom
                im = im[..., :-cropy, :]
            # update
            h, w = im.shape[-2:]

        if self.conf.random_center:
            im, shift_xy_ = self.randomize_image_center(im, self.conf.random_center_scale)
            # update
            shift_xy += shift_xy_
            h, w = im.shape[-2:]

        if target_size is not None:
            assert len(target_size) == 2, f"Expected 2-tuple, got {target_size}."
            if change_pix_ar:
                im = F.interpolate(
                    im[None],
                    target_size,
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )[0].clamp(0, 1)
                # update
                scale_xy[0] *= target_size[1] / w
                scale_xy[1] *= target_size[0] / h
                shift_xy[0] *= target_size[1] / w
                shift_xy[1] *= target_size[0] / h
            else:
                im, scale_xy_, shift_xy_ = self.set_im_size(im, target_size)
                # update
                scale_xy *= scale_xy_
                shift_xy = shift_xy * scale_xy_ + shift_xy_
            h, w = im.shape[-2:]
            return {
                "image": im,
                "scale_xy": scale_xy,
                "shift_xy": shift_xy,
                "image_size": np.array((w, h)),
                "original_image_size": np.array((wo, ho)),
            }

        if self.conf.square_crop:
            crop_size = min(h, w)
            offset = (h - crop_size) // 2, (w - crop_size) // 2
            im = im[:, offset[0] : offset[0] + crop_size, offset[1] : offset[1] + crop_size]
            # update
            h, w = im.shape[-2:]
            shift_xy[0] -= offset[1]
            shift_xy[1] -= offset[0]

        if self.conf.resize is not None:
            size = (
                self.get_new_image_size(h, w)
                if isinstance(self.conf.resize, int)
                else tuple(self.conf.resize)
            )
            im = self.resize(im, size)
            # update
            scale_xy[0] = size[1] / w
            scale_xy[1] = size[0] / h
            shift_xy *= scale_xy

        if self.conf.edge_divisible_by is not None:
            # crop to make the edge divisible by a number
            im, lrtb_crop = fit_features_to_multiple(im, self.conf.edge_divisible_by, crop=True)
            shift_xy[0] += lrtb_crop[0]  # lrtb_crop[0] = (w_cropped - w)//2
            shift_xy[1] += lrtb_crop[2]  # lrtb_crop[2] = (h_cropped - h)//2

        return {
            "image": im,
            "scale_xy": scale_xy,
            "shift_xy": shift_xy,
            "image_size": np.array((im.shape[-1], im.shape[-2])),
            "original_image_size": np.array((wo, ho)),
        }

    def resize(self, img: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize an image using the specified backend."""
        if self.conf.resize_backend == "kornia":
            return kornia.geometry.transform.resize(
                img,
                size,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
                interpolation=self.conf.interpolation,
            )
        return tvF.resize(
            img,
            size,  # type: ignore
            interpolation=self.conf.interpolation,
            antialias=self.conf.antialias,
        )

    def load_image(self, image_path: Path) -> dict:
        """Load an image from a path and preprocess it."""
        return self(load_image(image_path))

    def get_new_image_size(self, h: int, w: int) -> tuple[int, int]:
        """Get the new image size after resizing."""
        side = self.conf.side
        side_size = self.conf.resize
        aspect_ratio = w / h
        return (
            (side_size, int(side_size * aspect_ratio))
            if side == "vert" or (side != "horz" and (side == "short") ^ (aspect_ratio < 1.0))
            else (int(side_size / aspect_ratio), side_size)
        )

    @staticmethod
    def set_im_size(im: Tensor, target_size: tuple[int, int]) -> tuple[Tensor, Tensor, Tensor]:
        """Transform an image to the target size by center cropping and downscaling.

        This function tries to preserve center of the image such that the displacement of
        the principal point is minimal.
        This function also returns the scales and offsets needed to update the intrinsics
        corresponding to the "digitizing process" (focals fx, fy and pral. points cx, cy),
        to account for the cropping and scaling(s). Since this function does the following:
            1) (optional) upsampling with scale s1,
            2) center cropping of [shift_x, shift_y] pixels from the right or top of image,
            to achieve the target aspect ratio, and
            3) downsampling with scales [s2_x, s2_y] to the target resolution.
        Then to update:
            a) the focals (fx, fy): we need to multiply by s1 * [s2_x, s2_y],
            b) the principal point (cx, cy): scale also by s1 * [s2_x, s2_y], followed by
            shift of -[s2_x, s2_y]*[shift_x, shift_y] pixels.

        Args:
            im: (3, H, W) input image with RGB values in [0, 1].
            target_size: Integer 2-tuple with target resolution (height, width).

        Returns:
            im_transformed: (3, *target_size) Transformed image(s).
            scale_xy: (2,) Scales for updating the intrinsics (focals and principal point).
            shift_xy: (2,) Shifts for updating the principal point.
        """
        assert im.dim() == 3, f"Expected 3D tensor, got {im.dim()} with {im.shape=}"
        if im.shape[-2:] == target_size:
            # no need to resize
            return im, torch.ones(2, device=im.device), torch.zeros(2, device=im.device)

        h, w = im.shape[-2:]
        ht, wt = target_size

        # upsample preserving the aspect ratio so that no side is shorter than the targets
        if h < ht or w < wt:
            scale_1 = max(ht / h, wt / w)
            im = F.interpolate(
                im[None],
                scale_factor=scale_1,
                mode="bicubic",
                align_corners=False,
            )[0].clamp(0, 1)
            # update
            h_, w_ = im.shape[-2:]
            scale_1_xy = torch.tensor((w_ / w, h_ / h), device=im.device)
            h, w = h_, w_
        else:
            scale_1_xy = 1.0  # no upsampling

        # center crop from one side (either width or height) to achieve the target aspect ratio
        shift_xy = torch.zeros(2, device=im.device)
        ar_t = wt / ht
        if w / h > ar_t:
            # crop (negative pad) width, otherwise we would need to pad the height
            crop_w = round(w - h * ar_t)
            im = im[..., crop_w // 2 : w - crop_w + crop_w // 2]
            shift_xy[0] = -(crop_w // 2)  # NOTE: careful: -(crop_w // 2) != -crop_w // 2
        else:
            # crop height
            crop_h = round(h - w / ar_t)
            im = im[..., crop_h // 2 : h - crop_h + crop_h // 2, :]
            shift_xy[1] = -(crop_h // 2)
        h, w = im.shape[-2:]

        # downsample to the target resolution
        im = F.interpolate(
            im[None], target_size, mode="bicubic", align_corners=False, antialias=True
        )[0].clamp(0, 1)
        scale_2_xy = torch.tensor((wt / w, ht / h), device=im.device)
        # for updating the intrinsics
        scale_xy = scale_1_xy * scale_2_xy
        shift_xy = shift_xy * scale_2_xy
        return im, scale_xy, shift_xy

    @staticmethod
    def randomize_image_center(im: Tensor, scale: float = 0.015) -> tuple[Tensor, Tensor]:
        """Randomize the center of an image by random cropping of its borders.

        This results in also randomizing the rleative location of the principal point w.r.t.
        the image.

        Args:
            im: (3, H, W) input image.
            scale: Maximum displacement of the center as a fraction of the image size.

        Returns:
            im_c: (3, Hc, Wc) Cropped image.
            shift_xy: (2,) Shift to add for updating the principal point of the image.
        """
        h, w = im.shape[-2:]
        # displacement of the center of the image
        delta_x = random.randint(-round(w * scale), round(w * scale))
        delta_y = random.randint(-round(h * scale), round(h * scale))
        # image window to crop
        l = max(0, delta_x)
        r = min(w, w + delta_x)
        t = max(0, delta_y)
        b = min(h, h + delta_y)
        im_c = im[..., t:b, l:r]
        # offset to add for updating the principal point
        shift_xy = torch.tensor((-l, -t), device=im.device)
        return im_c, shift_xy


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def torch_image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Normalize and reorder the dimensions of an image tensor."""
    if image.ndim == 3:
        image = image.permute((1, 2, 0))  # CxHxW to HxWxC
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return (image.cpu().detach().numpy() * 255).astype(np.uint8)


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def write_image(img: torch.Tensor, path: Path):
    """Write an image tensor to a file."""
    img = torch_image_to_numpy(img) if isinstance(img, torch.Tensor) else img  # type: ignore
    cv2.imwrite(str(path), img[..., ::-1])  # type: ignore


def load_image(path: Path, grayscale: bool = False, return_tensor: bool = True) -> torch.Tensor:
    """Load an image from a path and return as a tensor."""
    image = read_image(path, grayscale=grayscale)
    if return_tensor:
        return numpy_image_to_torch(image)

    assert image.ndim in [2, 3], f"Not an image: {image.shape}"
    image = image[None] if image.ndim == 2 else image
    return torch.tensor(image.copy(), dtype=torch.uint8)


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt
    from PIL import Image

    from anycalib.cameras import Pinhole

    parser = argparse.ArgumentParser()
    parser.add_argument("--im_path", type=str)
    args = parser.parse_args()

    path = Path(args.im_path)
    im_ = torch.from_numpy(np.array(Image.open(path).convert("RGB"))) / 255
    im = im_.permute(2, 0, 1)

    h, w = im.shape[-2:]
    coords = Pinhole.pixel_grid_coords(h, w, im, 0)
    coords = coords[::50, ::50].reshape(-1, 2)
    colors = plt.get_cmap("hsv")(np.linspace(0, 1.0, len(coords)))
    np.random.shuffle(colors)

    cfg = {
        "resize": 336,
        "square_crop": True,
        "random_center": False,
        "random_center_scale": 0.5,
    }
    prep_r = ImagePreprocessor(cfg)
    # data_r = prep_r(im, target_size=None)
    # data_r = prep_r(im, target_size=(100, 200))
    data_r = prep_r(im, target_size=(h // 2, w - 100), crop=(0, 100), change_pix_ar=True)

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(im_)
    axs[0].scatter(coords[:, 0] - 0.5, coords[:, 1] - 0.5, c=colors, s=10, marker="x")

    coords_ = coords * data_r["scale_xy"] + data_r["shift_xy"]
    axs[1].imshow(data_r["image"].permute(1, 2, 0))
    axs[1].scatter(coords_[:, 0] - 0.5, coords_[:, 1] - 0.5, c=colors, s=10, marker="x")
    fig.tight_layout()
    fig.savefig("preprocessing.jpg", bbox_inches="tight")
