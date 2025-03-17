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

RAD2DEG = 180 / pi


class ResultsPlotter:

    def run(self, experiment_dir, model=None, overwrite=False, overwrite_eval=False):
        """Run export+eval loop"""
        self.save_conf(experiment_dir, overwrite=overwrite, overwrite_eval=overwrite_eval)
        pred_file = self.get_predictions(experiment_dir, model=model, overwrite=overwrite)
        # pred_file = experiment_dir / "predictions.h5"

        f = {}
        if not exists_eval(experiment_dir) or overwrite_eval or overwrite:
            s, f, r = self.run_eval(self.get_dataloader(self.conf.data, 1), pred_file)
            save_eval(experiment_dir, s, f, r)
        s, r = load_eval(experiment_dir)
        if self.conf.eval.get("delete_cache", False):
            for file in ("results.h5", "predictions.h5", "summaries.json", "conf.yaml"):
                (experiment_dir / file).unlink()
            experiment_dir.rmdir()
        return s, f, r


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
    output_dir = Path(EVAL_PATH, dataset_name)
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
