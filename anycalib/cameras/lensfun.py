"""Class and function utilities for working with Lensfun database."""

import xml.etree.ElementTree as ET
from math import pi, sqrt
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
from scipy.optimize import newton
from tqdm import tqdm

from anycalib.cameras.division import radii_via_companion
from anycalib.cameras.factory import CameraFactory
from anycalib.optim import GaussNewtonCalib, LevMarCalib


RAD2DEG = 180 / pi
DEG2RAD = pi / 180
URL_LENSFUN_DB = "https://api.github.com/repos/lensfun/lensfun/contents/data"


def download_github_folder(data_dir: Path, gh_api_url: str):
    """Download the Lensfun database using the GitHub API."""
    data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(gh_api_url)
    response.raise_for_status()
    contents = response.json()

    for item in tqdm(contents):
        if item["type"] == "file":
            response = requests.get(item["download_url"])
            response.raise_for_status()
            with open(data_dir / item["name"], "wb") as f:
                f.write(response.content)

        elif item["type"] == "dir":
            download_github_folder(data_dir / item["name"], item["url"])


class LensFunCam:

    DEFAULT_ASPECT_RATIO = 1.5  # 3:2
    DEFAULT_TYPE = "rectilinear"
    PROJECTIONS = {
        "rectilinear",
        "equisolid",
        "orthographic",
        "stereographic",
        "fisheye",
    }
    CAM_MAPS = {"ucm", "eucm", "division:1", "division:2"}

    def __init__(self):
        self.data_dir = data_dir = Path(__file__).parents[2] / "data" / "lensfun"
        self.db_dir = db_dir = data_dir / "db"

        # lensfun database
        if not db_dir.is_dir() or len(list(db_dir.glob("*.xml"))) == 0:
            print(f"Lensfun database not found. Downloading it in {data_dir}...")
            download_github_folder(data_dir, URL_LENSFUN_DB)

        self.gn_optimizer = GaussNewtonCalib({"max_iters": 10, "res_tangent": "fitted"})
        self.lm_optimizer = LevMarCalib({"max_iters": 10, "res_tangent": "fitted"})

        # create handy csv with data for projection and unprojection of each lens
        self.calib_data: pd.DataFrame = self.get_calib_data()

    @property
    def ref_sensor_diag_mm(self) -> float:
        """The refference diagonal is from a full-frame 35mm sensor"""
        return sqrt(36**2 + 24**2)

    def get_calib_data(self) -> pd.DataFrame:
        if (self.data_dir / "calib_data.csv").is_file():
            return pd.read_csv(self.data_dir / "calib_data.csv")
        print(f"Creating calib_data.csv at {self.data_dir}...")

        def ar_to_float(ar_str: str) -> float:
            """Convert aspect ratio string to float, e.g. '3:2' -> 1.5."""
            if ":" in ar_str:
                num, den = map(int, ar_str.split(":"))
                return num / den
            else:
                return float(ar_str)

        data_row = {
            "maker": None,
            "model": None,
            "mount": None,
            "crop_factor": None,
            "aspect_ratio": self.DEFAULT_ASPECT_RATIO,
            "type": self.DEFAULT_TYPE,
            "dist_model": None,
            "focal": None,
            "a": None,
            "b": None,
            "c": None,
            "k1": None,
            "k2": None,
            "k3": None,
        }
        data = []

        for file in tqdm(sorted(self.db_dir.glob("*.xml")), desc="Parsing XML files"):
            tree = ET.parse(file)
            root = tree.getroot()

            for lens in root.findall("lens"):
                data_row_ = data_row.copy()
                # required fields
                data_row_["maker"] = lens.find("maker").text  # type: ignore
                data_row_["model"] = lens.find("model").text  # type: ignore
                data_row_["mount"] = lens.find("mount").text  # type: ignore
                data_row_["crop_factor"] = lens.find("cropfactor").text  # type: ignore

                # optional fields
                aspect_ratio = lens.find("aspect-ratio")
                if aspect_ratio is not None:
                    data_row_["aspect_ratio"] = ar_to_float(aspect_ratio.text)  # type: ignore
                type_ = lens.find("type")
                if type_ is not None:
                    data_row_["type"] = type_.text

                # calibration data
                calibration = lens.find("calibration")
                if calibration is None:
                    continue
                dists = calibration.findall("distortion")
                if len(dists) == 0:
                    continue

                for dist in dists:
                    row = data_row_.copy()
                    dist_model = dist.get("model")
                    row["dist_model"] = dist_model
                    row["focal"] = dist.get("real-focal", dist.get("focal", None))
                    row["a"] = dist.get("a", None)
                    row["b"] = dist.get("b", None)
                    row["c"] = dist.get("c", None)
                    row["k1"] = dist.get("k1", None)
                    row["k2"] = dist.get("k2", None)
                    row["k3"] = dist.get("k3", None)

                    # ignore cameras with missing calibration data
                    if dist_model == "ptlens" and None in (row["a"], row["b"], row["c"]):  # fmt: skip
                        continue
                    elif dist_model == "poly3" and row["k1"] is None:
                        continue
                    elif dist_model == "poly5" and None in (row["k1"], row["k2"]):
                        continue

                    data.append(row)

        df = pd.DataFrame(data)
        # set numeric columns as such
        for col in (
            "crop_factor",
            "aspect_ratio",
            "focal",
            "a",
            "b",
            "c",
            "k1",
            "k2",
            "k3",
        ):
            df[col] = df[col].astype(float)
        df.to_csv(self.data_dir / "calib_data.csv", index=False)
        return df

    def rescale_coefficients(self, data_row: pd.Series) -> np.ndarray:
        """Rescale Hugin coefficients so that they accept/operate with radii expressed in mm.

        Hugin convention (see https://hugin.sourceforge.io/docs/manual/Lens_correction_model.html):
        "the largest circle that completely fits into an image is said to have radius=1.0.
        (In other words, radius=1.0 is half the smaller side of the image.)".
        This function is based on:
        https://github.com/lensfun/lensfun/blob/5faa1e5e1d2090f4a977382a0e0efe6f2d985795/libs/lensfun/mod-coord.cpp#L37-L71
        In this slightly modified version, we rescale the coefficients so that we can
        use them when working in *mm*.

        Args:
            data_row: calibration data for the lens.

        Returns:
            rescaled coefficients for the distortion model.
        """
        coeffs = np.zeros(3)
        crop_factor = data_row.crop_factor
        sensor_ar = data_row.aspect_ratio

        # constant factor for converting mm into Hugin's convention
        # NOTE: hugin convention defines r=1 as half of the smaller side of the image
        sensor_half_diag_mm = 0.5 * self.ref_sensor_diag_mm / crop_factor
        sensor_half_diag_hugin = sqrt(1 + sensor_ar**2)
        # sensor_half_diag_hugin = sqrt(1 + sensor_ar)
        mm_to_hugin = sensor_half_diag_hugin / sensor_half_diag_mm

        dist_model = data_row.dist_model.lower()
        if dist_model == "poly3":
            k1 = data_row.k1
            d = 1 - k1
            coeffs[0] = k1 * mm_to_hugin**2 / d**3

        elif dist_model == "poly5":
            k1, k2 = data_row.k1, data_row.k2
            coeffs[0] = k1 * mm_to_hugin**2
            coeffs[1] = k2 * mm_to_hugin**4

        elif dist_model == "ptlens":
            a, b, c = data_row.a, data_row.b, data_row.c
            d = 1 - a - b - c
            coeffs[0] = a * mm_to_hugin**3 / d**4
            coeffs[1] = b * mm_to_hugin**2 / d**3
            coeffs[2] = c * mm_to_hugin / d**2

        else:
            raise ValueError(f"Unknown distortion model: {dist_model}")

        return coeffs

    def undistort_radii(
        self, rd: np.ndarray, data_row: pd.Series
    ) -> tuple[np.ndarray, np.ndarray]:
        """Undistort radii that is expressed in mm.

        Args:
            rd: (n,) distorted radii in mm.
            data_row: calibration data for the lens.

        Returns:
            ru: (n,) undistorted radii in mm.
            converged: (n,) boolean array indicating if the undistortion converged.
        """

        coeffs = self.rescale_coefficients(data_row)  # (3,)
        # use Newton method for undistorting the radii
        func, fprime = {
            "poly3": (self._poly3, self._dpoly3_dru),
            "poly5": (self._poly5, self._dpoly5_dru),
            "ptlens": (self._ptlens, self._dptlens_dru),
        }[data_row.dist_model.lower()]
        ru, converged = newton(  # type: ignore
            func,
            rd,
            fprime=fprime,
            args=(coeffs, rd),
            tol=1e-8,
            maxiter=50,
            disp=False,
            full_output=True,
        )[:2]
        return ru, converged  # type: ignore

    def map_distortion_coefficients(
        self,
        target_cam_id: str,
        data_row: pd.Series,
        grid_side: int = 50,
        linfit_only_coeffs: bool = False,
        nonlin_opt: bool = False,
        max_polar_angle: float = 90,  # degrees
        fix_params: tuple[str] | None = None,
        safe_optim=True,
        optimizer="lm",  # "gn" or "lm"
    ) -> tuple[np.ndarray, bool]:
        """Map distortion coefficients from a Lensfun camera model to another standard
        camera model.

        Args:
            target_cam_id: target camera model.
            data_row: calibration data for the lens.
            n_radii: number of radii to sample in order to do the mapping.

        Returns:
            coeffs: (D_k,) distortion coefficients for the target camera model.
            valid: boolean array indicating if the mapping was successful
        """
        assert (
            target_cam_id in self.CAM_MAPS
        ), f"{target_cam_id} not implemented. Implemented models are: {self.CAM_MAPS}."

        cam = CameraFactory.create_from_id(target_cam_id)
        # Assume width is the longer side, i.e: aspect_ratio = width / height
        sensor_diag_mm = self.ref_sensor_diag_mm / data_row.crop_factor
        h = sensor_diag_mm * sqrt(1 / (data_row.aspect_ratio**2 + 1))
        w = h * data_row.aspect_ratio
        # params0 = torch.cat((torch.tensor((1, 1, 0 * w, 0 * h)), coeffs)).float()
        # image coordinates
        u = np.linspace(0.1 * w, 0.9 * w, grid_side)
        v = np.linspace(0.1 * w, 0.9 * h, int(grid_side / data_row.aspect_ratio))
        uv = np.stack(np.meshgrid(u, v, indexing="xy"), axis=-1).reshape(-1, 2)
        uvc = uv - [0.5 * w, 0.5 * h]
        # undistort sensor radii
        r = np.linalg.norm(uvc, axis=-1)
        ru, converged = self.undistort_radii(r, data_row)
        if not converged.all():
            cam_spec = f"Cam: {data_row.maker} {data_row.model} {data_row.type}."
            print(
                f"WARNING: {(~converged).sum()} radii undistortions failed for {cam_spec}"
            )
            r = r[converged]
            ru = ru[converged]
            uvc = uvc[converged]
            uv = uv[converged]
        # filter out points with too high polar angle (e.g. > 90 deg). This is specially
        # needed for full-frame sensors where the light does not hit every pixel.
        theta = self.unproject_undistorted_radii(ru, data_row.type, data_row.focal)
        ang_mask = theta < max_polar_angle * DEG2RAD
        if not ang_mask.all():
            cam_spec = f"Cam: {data_row.maker} {data_row.model} {data_row.type}."
            print(
                f"WARNING: {(~ang_mask).sum()} points with θ > {max_polar_angle}° for {cam_spec}"
            )
            r = r[ang_mask]
            ru = ru[ang_mask]
            uvc = uvc[ang_mask]
            uv = uv[ang_mask]
            theta = theta[ang_mask]

        # get ray coordinates, using gt undistortion params
        phi = np.atan2(uvc[:, 1], uvc[:, 0])
        R = np.sin(theta)
        rays = np.stack((R * np.cos(phi), R * np.sin(phi), np.cos(theta)), axis=-1)
        # linear fit
        uv = torch.from_numpy(uv).float()
        rays = torch.from_numpy(rays).float()
        if linfit_only_coeffs:
            f = data_row.focal  # in mm
            r_norm = torch.from_numpy(r / f).float()
            R = torch.from_numpy(np.sin(theta)).float()
            Z = torch.from_numpy(np.cos(theta)).float()
            coeffs, info = cam.fit_dist_from_radii(r_norm, R, Z)
            params0 = torch.cat(
                (torch.tensor((f, f, 0.5 * w, 0.5 * h)), coeffs)
            ).float()
        else:
            params0, info = cam.fit(uv, rays)
            # assert (params0[:2] >= 0).all(), params0

        if not nonlin_opt:
            coeffs = cam.params_to_dict(params0)["k"]
            return coeffs.numpy(), True if info is None else (info == 0).item()  # type: ignore
        # nonlinear optimization
        if hasattr(cam, "safe_optim"):
            cam.safe_optim = safe_optim  # type: ignore

        optimizer = self.gn_optimizer if optimizer == "gn" else self.lm_optimizer
        params, cost0, final_cost, _ = optimizer(
            cam, params0, uv, rays, fix_params=fix_params
        )
        params = params if final_cost < cost0 else params0
        coeffs = cam.params_to_dict(params)["k"]
        return coeffs.numpy(), True if info is None else (info == 0).item()  # type: ignore

    def find_lens(self, maker: str, model: str, mount: str) -> pd.Series:
        """Find the calibration data for a lens."""
        raise NotImplementedError

    def estimate_fov(
        self, data_row: pd.Series, as_deg: bool = True, which: str = "diag"
    ) -> float:
        """Estimate the field of view of a lens."""
        sensor_diag_mm = self.ref_sensor_diag_mm / data_row.crop_factor
        # Assume width is the longer side, i.e: aspect_ratio = width / height
        h = sensor_diag_mm * sqrt(1 / (data_row.aspect_ratio**2 + 1))
        w = h * data_row.aspect_ratio

        if which == "diag":
            r_mm = np.array([0.5 * sensor_diag_mm])  # max sensor radius
        elif which == "vfov":
            r_mm = np.array([0.5 * h])  # max sensor radius
        elif which == "hfov":
            r_mm = np.array([0.5 * w])
        else:
            raise ValueError(f"Unknown fov type: {which}")

        # undistort according to Lensfun distortion model
        ru_max_mm, converged = self.undistort_radii(r_mm, data_row)
        assert converged, f"Undistortion failed for {data_row}"

        # polar angle (fov)
        theta = self.unproject_undistorted_radii(
            ru_max_mm, data_row.type, data_row.focal
        )[0]
        theta = 2 * (RAD2DEG * theta if as_deg else theta)
        return float(theta)

    def undistort_image(
        self,
        im: np.ndarray,
        data_row: pd.Series,
        target_proj: str = "rectilinear",
        scale: float = 1.0,
    ) -> np.ndarray:
        """Undistort an image using the calibration data of a lens."""
        assert (
            target_proj in self.PROJECTIONS
        ), f"Unknown projection: {target_proj}. Choose from {self.PROJECTIONS}."
        assert data_row.type in self.PROJECTIONS, f"Unknown projection: {data_row.type}"

        # get centered image coordinates
        h, w, c = im.shape
        assert c == 3
        assert (
            abs((max(h, w) / min(h, w)) - data_row.aspect_ratio) < 0.01
        ), f"Aspect ratio mismatch: {data_row}"
        im_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)  # fmt:skip
        im_coords_c = im_coords.copy().astype(float)
        im_coords_c[:, 0] -= 0.5 * w
        im_coords_c[:, 1] -= 0.5 * h
        # transform to mm
        sensor_diag_mm = self.ref_sensor_diag_mm / data_row.crop_factor
        sensor_diag_pix = sqrt((w + 1) ** 2 + (h + 1) ** 2)
        pix_to_mm = sensor_diag_mm / sensor_diag_pix
        im_coords_c_mm = im_coords_c * pix_to_mm
        sensor_radii = np.linalg.norm(im_coords_c_mm, axis=1)
        # get forward distortion map
        theta = self.unproject_undistorted_radii(
            sensor_radii, target_proj, data_row.focal * scale
        )
        ru = self.project_polar_angles(theta, data_row.type, data_row.focal)
        rd = {
            "poly3": self._apply_poly3_dist,
            "poly5": self._apply_poly5_dist,
            "ptlens": self._apply_ptlens_dist,
        }[data_row.dist_model](ru, self.rescale_coefficients(data_row))

        eps = np.finfo(sensor_radii.dtype).eps
        dist = np.where(sensor_radii < eps, 1, rd / sensor_radii.clip(eps))
        map_xy = im_coords_c_mm / pix_to_mm * dist[:, None]
        map_xy[:, 0] += 0.5 * w
        map_xy[:, 1] += 0.5 * h
        map_xy = map_xy.reshape(h, w, 2)
        im_u = cv2.remap(im, map_xy.astype(np.float32), None, cv2.INTER_LANCZOS4)  # type: ignore
        return im_u

    def undistort_image_with_mapped_coeffs(
        self,
        im: np.ndarray,
        data_row: pd.Series,
        target_cam_id: str,
        linfit_only_coeffs: bool = True,
        grid_side: int = 50,
        scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert target_cam_id in self.CAM_MAPS
        target_proj = "rectilinear"
        # get centered image coordinates
        h, w, _ = im.shape
        im_coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)  # fmt:skip
        im_coords_c = im_coords.copy().astype(float)
        im_coords_c[:, 0] -= 0.5 * w
        im_coords_c[:, 1] -= 0.5 * h
        # transform to mm
        sensor_diag_mm = self.ref_sensor_diag_mm / data_row.crop_factor
        sensor_diag_pix = sqrt(w**2 + h**2)
        pix_to_mm = sensor_diag_mm / sensor_diag_pix
        im_coords_c_mm = im_coords_c * pix_to_mm
        sensor_radii = np.linalg.norm(im_coords_c_mm, axis=1)
        # get forward distortion map
        theta = self.unproject_undistorted_radii(
            sensor_radii, target_proj, data_row.focal * scale
        )
        coeffs, valid = self.map_distortion_coefficients(
            target_cam_id,
            data_row,
            linfit_only_coeffs=linfit_only_coeffs,
            grid_side=grid_side,
        )
        assert valid, f"Mapping failed for {data_row}"
        rd = {
            "ucm": self._apply_ucm_dist,
            "eucm": self._apply_eucm_dist,
            "division:1": self._apply_division1_dist,
            "division:2": self._apply_division2_dist,
        }[target_cam_id](theta, coeffs)
        rd = rd * data_row.focal

        eps = np.finfo(sensor_radii.dtype).eps
        dist = np.where(sensor_radii < eps, 1, rd / sensor_radii.clip(eps))
        map_xy = im_coords_c * dist[:, None]
        map_xy[:, 0] += 0.5 * w
        map_xy[:, 1] += 0.5 * h
        map_xy = map_xy.reshape(h, w, 2)
        im_u = cv2.remap(im, map_xy.astype(np.float32), None, cv2.INTER_LANCZOS4)  # type: ignore
        return im_u, coeffs

    def unproject_undistorted_radii(
        self, ru: np.ndarray, target_proj: str, focal: float
    ) -> np.ndarray:
        """Unproject points from the sensor plane (in mm) to their corresponding polar
        angle.

        The different projections that Lensfun uses are covered in:
        https://lensfun.github.io/manual/v0.3.2/corrections.html#changeofprojection

        Args:
            ru: (..., n) undistorted radii in mm.
            target_proj: target projection.
            focal: focal length in mm.

        Returns:
            theta: (..., n) polar angles in radians.
        """
        f = focal
        # assert target_proj in self.PROJECTIONS, f"Unknown projection: {target_proj}"
        assert (ru >= 0).all(), ru.min()
        assert f >= 0
        if target_proj == "rectilinear":
            theta = np.arctan((ru / f).clip(0))
        elif target_proj == "equisolid":
            theta = 2 * np.arcsin((0.5 * ru / f).clip(0, 1))
        elif target_proj == "fisheye":  # equidistant
            theta = (ru / f).clip(0)
        elif target_proj == "stereographic":
            theta = 2 * np.arctan((0.5 * ru / f).clip(0))
        elif target_proj == "orthographic":
            theta = np.arcsin((ru / f).clip(0, 1))
        else:
            raise ValueError(f"Unknown projection: {target_proj}")

        return theta

    def project_polar_angles(
        self, theta: np.ndarray, target_proj: str, focal: float
    ) -> np.ndarray:
        """Map polar angles to (undistorted) sensor plane radii (in mm) according to
        the lens projection.

        The different projections that Lensfun uses are covered in:
        https://lensfun.github.io/manual/v0.3.2/corrections.html#changeofprojection

        Args:
            theta: (n,) polar angles in radians.
            target_proj: target projection.
            focal: focal length in mm.

        Returns:
            ru: (n,) undistorted radii in mm.
        """
        f = focal
        assert target_proj in self.PROJECTIONS

        if target_proj == "rectilinear":
            ru = f * np.tan(theta)
        elif target_proj == "equisolid":
            ru = 2 * f * np.sin(0.5 * theta)
        elif target_proj == "fisheye":
            ru = f * theta
        elif target_proj == "stereographic":
            ru = 2 * f * np.tan(0.5 * theta)
        elif target_proj == "orthographic":
            ru = f * np.sin(theta)
        else:
            raise ValueError(f"Unknown projection: {target_proj}")

        return ru.clip(0)

    @staticmethod
    def _apply_poly3_dist(ru, k):
        return ru * (1 + k[0] * ru**2)

    @staticmethod
    def _poly3(ru, k, rd):
        return ru * (1 + k[0] * ru**2) - rd

    @staticmethod
    def _dpoly3_dru(ru, k, rd):
        return 1 + 3 * k[0] * ru**2

    @staticmethod
    def _apply_poly5_dist(ru, k):
        return ru * (1 + k[0] * ru**2 + k[1] * ru**4)

    @staticmethod
    def _poly5(ru, k, rd):
        return ru * (1 + k[0] * ru**2 + k[1] * ru**4) - rd

    @staticmethod
    def _dpoly5_dru(ru, k, rd):
        return 1 + 3 * k[0] * ru**2 + 5 * k[1] * ru**4

    @staticmethod
    def _apply_ptlens_dist(ru, k):
        return ru * (1 + k[2] * ru + k[1] * ru**2 + k[0] * ru**3)

    @staticmethod
    def _ptlens(ru, k, rd):
        return ru * (1 + k[2] * ru + k[1] * ru**2 + k[0] * ru**3) - rd

    @staticmethod
    def _dptlens_dru(ru, k, rd):
        return 1 + 2 * k[2] * ru + 3 * k[1] * ru**2 + 4 * k[0] * ru**3

    @staticmethod
    def _apply_ucm_dist(theta, k):
        assert k.shape == (1,)
        return np.sin(theta) * (k[0] + 1) / (k[0] + np.cos(theta))

    @staticmethod
    def _apply_eucm_dist(theta, k):
        assert k.shape == (2,)
        R, Z = np.sin(theta), np.cos(theta)
        d = np.sqrt(k[1] * R**2 + Z**2)
        return R / (k[0] * d + (1 - k[0]) * Z)

    @staticmethod
    def _apply_division1_dist(theta, k):
        assert k.shape == (1,)
        eps = np.finfo(theta.dtype).eps
        R, Z = np.sin(theta), np.cos(theta)
        disc = Z**2 - 4 * k[0] * R**2
        den = Z + np.sqrt(disc.clip(0))  # use + to get smallest root
        assert ((disc > -eps) & (den > eps)).all()
        return 2 * R / den

    @staticmethod
    def _apply_division2_dist(theta, k):
        assert k.shape == (2,)
        R = torch.from_numpy(np.sin(theta))
        Z = torch.from_numpy(np.cos(theta))
        r, valid = radii_via_companion(R, Z, torch.from_numpy(k))
        assert valid.sum() > 0.99 * len(valid)
        return r.numpy()

