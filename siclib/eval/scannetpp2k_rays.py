import logging
import resource
import shutil
from pathlib import Path
from pprint import pprint

import h5py
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from siclib.eval.io import get_eval_parser, parse_eval_args
from siclib.eval.simple_pipeline_rays import SimplePipeline
from siclib.settings import EVAL_PATH

logger = logging.getLogger(__name__)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

torch.set_grad_enabled(False)


def prepare_benchmark(bench_dir: Path, scannetpp_dir: Path):
    if bench_dir.exists():
        logger.info(f"Benchmark ScanNet++ already exists at {bench_dir}, skipping preparation.")
        return
    if not scannetpp_dir.exists():
        raise FileNotFoundError(f"ScanNet++ directory {scannetpp_dir} does not exist.")

    bench_im_dir = bench_dir / "images"
    bench_im_dir.mkdir(parents=True)

    # copy eval data to benchmark dir
    eval_data = Path(__file__).parent / "scannetpp2k_images.h5"
    shutil.copy(eval_data, bench_dir / "images.h5")

    # copy eval images from ScanNet++ to benchmark dir
    logger.info(f"Copying evaluation images from {scannetpp_dir} to {bench_im_dir}.")
    base_dir = scannetpp_dir / "data"
    rel_im_dir = Path("dslr") / "resized_images"
    with h5py.File(eval_data, "r") as f:
        scene_im_names = list(f.keys())  # format name: "<scene>_<im_name>.JPG"
    for scene_im_name in tqdm(scene_im_names):
        scene, im_name = scene_im_name.split("_")
        im_path = base_dir / scene / rel_im_dir / im_name
        shutil.copy(im_path, bench_im_dir / scene_im_name)


class ScanNetpp2k(SimplePipeline):
    default_conf = {
        "data": {
            "name": "simple_dataset_rays",
            "dataset_dir": "data/scannetpp2k",
            "test_img_dir": "${.dataset_dir}/images",
            "test_csv": "${.dataset_dir}/images.csv",
            "test_h5": "${.dataset_dir}/images.h5",
            "augmentations": {"name": "identity"},
            "preprocessing": {"resize": None, "edge_divisible_by": None},
            "test_batch_size": 1,
            "cam_id": None,
        },
        "model": {},
        "eval": {
            "thresholds": [1, 5, 10],
            "pixel_thresholds": [0.5, 1, 3, 5],
            "num_vis": 10,
            "verbose": True,
            "eval_on_edit": False,
        },
        "url": None,
        "scannetpp_root": "data/scannetpp",
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

        prepare_benchmark(Path(conf.data.dataset_dir), Path(conf.scannetpp_root))


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(ScanNetpp2k.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)  # type: ignore
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name, args, "configs/", default_conf, only_custom_model=False
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = ScanNetpp2k(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
