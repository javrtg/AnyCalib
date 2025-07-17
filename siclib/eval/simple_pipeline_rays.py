import logging
import resource
from collections import defaultdict
from math import pi
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from anycalib.cameras import CameraFactory
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3
from anycalib.visualization.viz_batch import make_batch_figures
from siclib.datasets import get_dataset
from siclib.eval.eval_pipeline import EvalPipeline
from siclib.eval.io import load_model
from siclib.eval.utils import download_and_extract_benchmark, plot_scatter_grid
from siclib.models.cache_loader import CacheLoader
from siclib.utils.export_predictions import export_predictions
from siclib.utils.tools import AUCMetric

# flake8: noqa
# mypy: ignore-errors

logger = logging.getLogger(__name__)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.set_grad_enabled(False)

RAD2DEG = 180 / pi


def compute_metrics(pred: dict, data: dict, cam: BaseCamera, thresholds: list[float]):
    intrins = cam.reverse_scale_and_shift(pred["intrinsics"], data["scale_xy"], data["shift_xy"])
    cam_gt = CameraFactory.create_from_id(data["cam_id_gt"])
    intrins_gt = data["intrinsics_gt"][: cam_gt.nparams]  # intrins_gt may be padded to be stackable

    results = {}
    w_cxcy_prior = data.get("cxcy", None) is not None
    w, h = data["original_image_size"].tolist()

    if cam.id == cam_gt.id:
        # intrinsics absolute and relative errors
        abs_errors = torch.abs(intrins - intrins_gt)
        rel_errors = abs_errors / intrins_gt.abs()
        for abs_e, rel_e, p_str in zip(abs_errors, rel_errors, cam.PARAMS_IDX.keys()):
            if p_str in ("cx", "cy") and w_cxcy_prior:
                # skip cx, cy if they are not predicted
                continue
            results[f"{p_str}_error"] = abs_e.item()
            results[f"rel_{p_str}_error"] = rel_e.item()

        if "fx" in cam.PARAMS_IDX:
            # compute max error between fx, fy
            rel_fx = rel_errors[cam.PARAMS_IDX["fx"]]
            rel_fy = rel_errors[cam.PARAMS_IDX["fy"]]
            results["max_rel_f_error"] = max(rel_fx, rel_fy).item()

        # compute max error between cx, cy
        abs_cx = abs_errors[cam.PARAMS_IDX["cx"]]
        abs_cy = abs_errors[cam.PARAMS_IDX["cy"]]
        results["max_rel_c_error"] = 2 * max(abs_cx / w, abs_cy / h).item()

    # compute vfov error
    vfov_gt, valid_vfov_gt = cam_gt.get_vfov(intrins_gt, h)
    vfov, valid_vfov = cam.get_vfov(intrins, h)
    vfov_gt = vfov_gt * RAD2DEG
    vfov = vfov * RAD2DEG
    results["vfov_gt"] = vfov_gt.item()
    results["vfov"] = vfov.item()
    if valid_vfov and valid_vfov_gt:
        results["vfov_error"] = torch.abs(vfov - vfov_gt).item()
    else:
        results["vfov_error"] = float("nan")
    # hfov error
    hfov_gt, valid_hfov_gt = cam_gt.get_hfov(intrins_gt, w)
    hfov, valid_hfov = cam.get_hfov(intrins, w)
    hfov_gt = hfov_gt * RAD2DEG
    hfov = hfov * RAD2DEG
    results["hfov_gt"] = hfov_gt.item()
    results["hfov"] = hfov.item()
    if valid_hfov and valid_hfov_gt:
        results["hfov_error"] = torch.abs(hfov - hfov_gt).item()
    else:
        results["hfov_error"] = float("nan")

    if getattr(cam, "NUM_F", 0) > 0 and getattr(cam_gt, "NUM_F", 0) > 0:
        # vfov error assuming pinhole model
        vfov_pin_gt = cam_gt.get_pinhole_vfov(intrins_gt, h) * RAD2DEG
        vfov_pin = cam.get_pinhole_vfov(intrins, h) * RAD2DEG
        results["vfov_pin_error"] = torch.abs(vfov_pin - vfov_pin_gt).item()

    # mean reprojection error
    im_coords_grid = cam.pixel_grid_coords(h, w, intrins_gt, offset=0.5)  # (H, W, 2)
    bearings, valid_u = cam_gt.unproject(intrins_gt, im_coords_grid)  # (H, W, 3)
    im_coords_pred, valid_p = cam.project(intrins, bearings)  # (H, W, 2)
    assert im_coords_grid.shape == (h, w, 2) == im_coords_pred.shape
    assert bearings.shape == (h, w, 3)
    errors = torch.linalg.norm(im_coords_grid - im_coords_pred, dim=-1)
    if valid_u is not None and valid_p is not None:
        errors = errors[valid_u & valid_p]
    elif valid_u is not None:
        errors = errors[valid_u]
    elif valid_p is not None:
        errors = errors[valid_p]
    results["reproj_error"] = errors.mean().item()
    for th in thresholds:
        results[f"reproj_error@{th}"] = (errors < th).float().mean().item()

    # mean angular error
    bearings_gt, valid_gt = cam_gt.unproject(intrins_gt, im_coords_grid)
    bearings, valid = cam.unproject(intrins, im_coords_grid)
    ang_errors = RAD2DEG * Unit3.distance(bearings_gt, bearings)
    if valid_gt is not None and valid is not None:
        ang_errors = ang_errors[valid & valid_gt]
    elif valid is not None:
        ang_errors = ang_errors[valid]
    elif valid_gt is not None:
        ang_errors = ang_errors[valid_gt]
    results["ang_error"] = ang_errors.mean().item()
    results["ang_med_error"] = ang_errors.median().item()

    # compute pixel distortion error as in GeoCalib's paper
    if "radial" in cam.id and "radial" in cam_gt.id:
        # 1) unproject without distortion and using gt intrinsics
        # 2) project with both gt and pred intrinsics but using gt focal and cxcy
        # 3) compare pixel distances
        # 1)
        k1_idx_gt = cam_gt.PARAMS_IDX["k1"]
        cam_pin = CameraFactory.create_from_id("pinhole")
        bearings_pin, _ = cam_pin.unproject(intrins_gt[:k1_idx_gt], im_coords_grid)
        # 2)
        k1_idx = cam.PARAMS_IDX["k1"]
        im_gt, v1 = cam_gt.project(intrins_gt, bearings_pin)
        im_pred, v2 = CameraFactory.create_from_id(f"{cam_gt.id[:-1]}{cam.num_k}").project(  # type: ignore
            torch.cat((intrins_gt[:k1_idx_gt], intrins[k1_idx:])), bearings_pin
        )  # fxfy and cxcy from GT cam. distortion coeffs from predicted cam
        # 3)
        assert v1 is not None and v2 is not None
        err = torch.linalg.norm(im_gt - im_pred, dim=-1)[v1 & v2]
        for th in thresholds:
            results[f"pixel_distortion_error@{th}"] = (err < th).float().mean().item()

        if cam.id == cam_gt.id:
            # compute Mahalanobis distances for 1st distortion term
            J = cam_gt.jac_bearings_wrt_params(
                intrins_gt, bearings.reshape(-1, 3), im_coords_grid.reshape(-1, 2)
            ).flatten(0, 1)  # (H*W*3, nparams) # fmt: skip
            inv_cov = J.transpose(-1, -2) @ J  # (nparams, nparams)
            err = intrins - intrins_gt
            results["maha_error"] = (err[None] @ inv_cov @ err[:, None]).item()

    return results


class SimplePipeline(EvalPipeline):
    default_conf = {
        "data": {},
        "model": {},
        "eval": {
            "thresholds": [1, 5, 10],
            "pixel_thresholds": [1, 3, 5],
            "num_vis": 5,
            "verbose": True,
            "delete_cache": False,
            "delete_also_summaries": True,
            "eval_on_edit": False,
        },
        "url": None,  # url to benchmark.zip
    }

    export_keys = ["intrinsics"]

    optional_export_keys = [
        # "intrinsics_uncertainty",
        # "rays",
        # "log_covs",
    ]

    def _init(self, conf):
        self.verbose = conf.eval.verbose
        self.num_vis = self.conf.eval.num_vis

        if conf.eval.eval_on_edit:
            logger.info("Evaluating on EDITED dataset")
            conf.data.dataset_dir = conf.data.dataset_dir + "_edit"
            conf.data.test_img_dir = conf.data.dataset_dir + "/images"
            conf.data.test_h5 = conf.data.dataset_dir + "/images.h5"
            conf.url = conf.url_edit

        if conf.url is not None:
            ds_dir = Path(conf.data.dataset_dir)
            download_and_extract_benchmark(ds_dir.name, conf.url, ds_dir.parent)

    @classmethod
    def get_dataloader(cls, data_conf=None, batch_size=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf or cls.default_conf["data"]

        if batch_size is not None:
            data_conf["test_batch_size"] = batch_size

        do_shuffle = data_conf["test_batch_size"] > 1
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test", shuffle=do_shuffle)

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
        # set_seed(0)
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,  # type: ignore
                optional_keys=self.optional_export_keys,
                verbose=self.verbose,
            )
        return pred_file

    def get_figures(self, results):
        figures = {}
        if self.num_vis == 0:
            return figures

        ray_metrics = ["ang_rays"]
        cam_metrics = ["vfov"]

        if all(k in results for k in cam_metrics):
            x_keys = [f"{k}_gt" for k in cam_metrics]

            # gt vs error
            y_keys = [f"{k}_error" for k in cam_metrics]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, show_means=False)
            figures |= {"cam": fig}

            # gt vs pred
            y_keys = [f"{k}" for k in cam_metrics]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, diag=True, show_means=False)
            figures |= {"cam": fig}

        if all(f"{k}_error" in results for k in ray_metrics):
            x_keys = [f"{k}_gt" for k in cam_metrics]
            y_keys = [f"{k}_error" for k in ray_metrics]
            fig, _ = plot_scatter_grid(results, x_keys, y_keys, show_means=False)
            figures |= {"rays_gt_error": fig}

        return figures

    def run_eval(self, loader, pred_file):
        conf = self.conf.eval
        results = defaultdict(list)

        save_to = Path(pred_file).parent / "figures"
        if not save_to.exists() and self.num_vis > 0:
            save_to.mkdir()

        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()

        if not self.verbose:
            logger.info(f"Evaluating {pred_file}")

        for i, data in enumerate(
            tqdm(loader, desc="Evaluating", total=len(loader), ncols=80, disable=not self.verbose)
        ):
            # NOTE: data is batched but pred is not
            pred = cache_loader(data)
            data = {
                k: (v[0] if k != "image" and v is not None else v) for k, v in data.items()
            }  # eliminate batch dimension except image (for plotting reasons)

            # store pred intrins
            intrins: Tensor = pred["intrinsics"]
            results["intrinsics"] = intrins.tolist()

            # data and gt
            results["names"].append(data["name"])
            results["cam_id"].append(data["cam_id"])
            results["im_size"].append(tuple(data["image"].shape[-2:]))
            results["intrinsics_gt"].append(data["intrinsics"])

            cam = CameraFactory.create_from_id(data["cam_id"])
            # add error metrics
            camera_metrics = compute_metrics(pred, data, cam, conf.pixel_thresholds)
            ang_med_error = camera_metrics["ang_med_error"]
            for k, v in camera_metrics.items():
                results[k].append(v)

            if "intrinsics_uncertainty" in pred:
                results["intrinsics_uncertainty"].append(pred["intrinsics_uncertainty"].tolist())


            if i < self.num_vis:
                if "rays" not in pred:
                    h, w = data["image"].shape[-2:]
                    # w, h = data["original_image_size"].tolist()
                    pred["rays"] = cam.ray_grid(h, w, pred["intrinsics"])[0].view(h * w, 3)

                figs = make_batch_figures(pred, data)
                for n, fig in figs.items():
                    fig.savefig(
                        save_to / f"rays-{n}-{i}-{ang_med_error:.3f}.jpg",
                        bbox_inches="tight",
                        pad_inches=0,
                    )
                    plt.close(fig)

        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(np.array(v).dtype, np.number):
                continue

            if k.endswith("_error") or any(kw in k for kw in ("recall", "pixel", "reproj")):
                summaries[f"mean_{k}"] = round(np.nanmean(arr), 3)
                summaries[f"median_{k}"] = round(np.nanmedian(arr), 3)
                if k == "vfov_error" or k == "hfov_error":
                    summaries[f"nans_{k}"] = np.sum(np.isnan(arr))
                    summaries[f"nans%_{k}"] = round(np.sum(np.isnan(arr)) / len(arr) * 100, 3)

                if any(keyword in k for keyword in ["max_rel_f", "max_rel_c", "vfov", "hfov"]):
                    if not conf.thresholds:
                        continue

                    auc = AUCMetric(
                        elements=arr, thresholds=list(conf.thresholds), min_error=1
                    ).compute()
                    for i, t in enumerate(conf.thresholds):
                        summaries[f"auc_{k}@{t}"] = round(auc[i], 3)  # type: ignore

        return summaries, self.get_figures(results), results


if __name__ == "__main__":
    import pprint

    from omegaconf import OmegaConf

    from siclib.eval.io import get_eval_parser, parse_eval_args
    from siclib.settings import EVAL_PATH  # type: ignore

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(SimplePipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)  # type: ignore
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(dataset_name, args, "configs/", default_conf)

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = SimplePipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    pprint.pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
