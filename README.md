<div align="center">
    <h1>AnyCalib:<br>
    On-Manifold Learning for Model-Agnostic Single-View Camera Calibration</h1>
    <p>Javier Tirado-Garín &emsp;&emsp; Javier Civera<br>
    I3A, University of Zaragoza</p>
    <img width="99%" src="https://github.com/javrtg/AnyCalib/blob/main/assets/method_dark.png?raw=true">
    <p><strong>Camera calibration from a single perspective/edited/distorted image using a freely chosen camera model</strong></p>
    <p><a href="https://arxiv.org/abs/2503.12701">arXiv</a></p>
</div>


## Usage (pretrained models)

The only requirements are Python (≥3.10) and PyTorch.
The project, in development mode, can be installed with:
```shell
git clone https://github.com/javrtg/AnyCalib.git && cd AnyCalib
pip install -e .
```
Alternatively, and optionally, a compatible version of [`xformers`](https://github.com/facebookresearch/xformers) can also be installed for better efficiency by running the following instead of `pip install -e .`:
```shell
pip install -e .[eff]
```


### Minimal usage example
```python
import numpy as np
import torch
from PIL import Image  # the library of choice to load images

from anycalib import AnyCalib


dev = torch.device("cuda")

# load input image and convert it to a (3, H, W) tensor with RGB values in [0, 1]
image = np.array(Image.open("path/to/image.jpg").convert("RGB"))
image = torch.tensor(image, dtype=torch.float32, device=dev).permute(2, 0, 1) / 255

# instantiate AnyCalib according to the desired model_id. Options:
# "anycalib_pinhole": model trained with *only* perspective (pinhole) images,
# "anycalib_gen": trained with perspective, distorted and strongly distorted images,
# "anycalib_dist": trained with distorted and strongly distorted images,
# "anycalib_edit": Trained on edited (stretched and cropped) perspective images.
model = AnyCalib(model_id="anycalib_pinhole").to(dev)

# predict according to the desired camera model. Implemented camera models are detailed further below.
output = model.predict(image, cam_id="pinhole")
# output is a dictionary with the following key-value pairs:
# {
#      "intrinsics": (D,) tensor with the estimated intrinsics for the selected camera model,
#      "fov_field": (N, 2) tensor with the regressed FoV field by the network. N≈320^2 (resolution close to the one seen during training),
#      "tangent_coords": alias for "fov_field",
#      "rays": (N, 3) tensor with the corresponding (via the exponential map) ray directions in the camera frame (x right, y down, z forward),
#      "pred_size": (H, W) tuple with the image size used by the network. It can be used e.g. for resizing the FoV/ray fields to the original image size.
# }
```
The weights of the selected `model_id`, if not already downloaded, will be automatically downloaded to the Torch Hub cache directory: `torch.hub.get_dir()`. <br> 
Additional configuration options are indicated in the docstring of `AnyCalib`: 
<details>
<summary> <code>help(AnyCalib)</code> </summary>

```python
    """AnyCalib class.

    Args for instantiation:
        model_id: one of {'anycalib_pinhole', 'anycalib_gen', 'anycalib_dist', 'anycalib_edit'}.
            Each model differes in the type of images they seen during training:
                * 'anycalib_pinhole': Perspective (pinhole) images,
                * 'anycalib_gen': General images, including perspective, distorted and
                    strongly distorted images, and
                * 'anycalib_dist': Distorted images using the Brown-Conrady camera model
                    and strongly distorted images, using the EUCM camera model,
                * 'anycalib_edit': Trained on edited (stretched and cropped) perspective
                    images.
            Default: 'anycalib_pinhole'.
        nonlin_opt_method: nonlinear optimization method: 'gauss_newton' or 'lev_mar'.
            Default: 'gauss_newton'
        nonlin_opt_conf: nonlinear optimization configuration.
            This config can be used to control the number of iterations and the space
            where the residuals are minimized. See the classes `GaussNewtonCalib` or
            `LevMarCalib` under anycalib/optim for details. Default: None.
        init_with_sac: use RANSAC instead of nonminimal fit for initializating the
            intrinsics. Default: False.
        fallback_to_sac: use RANSAC if nonminimal fit fails. Default: True.
        ransac_conf: RANSAC configuration. This config can be used to control e.g. the
            inlier threshold or the number of minimal samples to try. See the class
            `RANSAC` in anycalib/ransac.py for details. Default: None.
        rm_borders: border size of the dense FoV fields to ignore during fitting.
            Default: 0.
        sample_size: approximate number of 2D-3D correspondences to use for fitting the
            intrinsics. Negative value -> no subsampling. Default: -1.
    """
```
</details>

### Minimal batched example
AnyCalib can also be executed in batch and using possibly different camera models for each image. For example:
```python
images = ... # (B, 3, H, W)
# NOTE: if cam_ids is a list, then len(cam_ids) must be equal to B
cam_ids = ["pinhole", "radial:1", "kb:4"]  # different camera models for each image
cam_ids = "pinhole"  # same camera model across images
output = model.predict(images, cam_id=cam_ids)
# corresponding batched output dictionary:
# {
#      "intrinsics": List[(D_i,) tensors] for each camera model "i",
#      "fov_field": (B, N, 2) tensor,
#      "tangent_coords": alias for "fov_field",
#      "rays": (B, N, 3) tensor,
#      "pred_size": (H, W).
# }
```

### Currently implemented camera models
* `cam_id` represents the camera model identifier(s) that can be used in the `predict` method. <br> 
* `D` corresponds to the number of intrinsics of the camera model. It determines the length of each `intrinsics` tensor in the output dictionary.

| `cam_id` | Description | `D` | Intrinsics |
|:--|:--|:-:|:--|
| `pinhole` | Pinhole camera model | 4 | $f_x,~f_y,~c_x,~c_y$ |
| `simple_pinhole` | `pinhole` with one focal length | 3 | $f,~c_x,~c_y$ |
| `radial:k` | Radial (Brown-Conrady) [[1]](#1) camera model with `k` $\in$ [1, 4] distortion coefficients | 4+`k` | $f_x,~f_y,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |
| `simple_radial:k` | `radial:k` with one focal length | 3+`k` | $f,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |
| `kb:k` | Kannala-Brandt [[2]](#2) camera model with `k` $\in$ [1, 4] distortion coefficients | 4+`k` | $f_x,~f_y,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |
| `simple_kb:k` | `kb:k` with one focal length | 3+`k` | $f,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |
| `ucm` | Unified Camera Model [[3]](#3) | 5 | $f_x,~f_y,~c_x,~c_y$ <br> $k$ |
| `simple_ucm` | `ucm` with one focal length | 4 | $f,~c_x,~c_y$ <br> $k$ |
| `eucm` | Enhanced Unified Camera Model [[4]](#4) | 6 | $f_x,~f_y,~c_x,~c_y$ <br> $k_1,~k_2$ |
| `simple_eucm` | `eucm` with one focal length | 5 | $f,~c_x,~c_y$ <br> $k_1,~k_2$ |
| `division:k` | Division camera model [[5]](#5) with `k` $\in$ [1, 4] distortion coefficients | 4+`k` | $f_x,~f_y,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |
| `simple_division:k` | `division:k` with one focal length | 3+`k` | $f,~c_x,~c_y$ <br> $k_1[,~k_2[,~k_3[,~k_4]]]$ |

In addition to the original works, we recommend the works of Usenko et al. [[6]](#6) and Lochman et al. [[7]](#7) for a comprehensive comparison of the different camera models.


## Evaluation
The evaluation and training code is built upon the [`siclib`](siclib) library from [GeoCalib](https://github.com/cvg/GeoCalib), which can be installed as:
```shell
pip install -e siclib
```
Running the evaluation commands will write the results to `outputs/results/`.

### LaMAR
Running the evaluation commands will download the dataset to `data/lamar2k` which will take around 400 MB of disk space.

AnyCalib trained on $\mathrm{OP_{p}}$: 
```shell
python -m siclib.eval.lamar2k_rays --conf anycalib_pretrained --tag anycalib_p --overwrite
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.lamar2k_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen
```

### MegaDepth (pinhole)
Running the evaluation commands will download the dataset to `data/megadepth2k` which will take around 2 GB of disk space.

AnyCalib trained on $\mathrm{OP_{p}}$: 
```shell
python -m siclib.eval.megadepth2k_rays --conf anycalib_pretrained --tag anycalib_p --overwrite
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.megadepth2k_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen
```

### TartanAir
Running the evaluation commands will download the dataset to `data/tartanair` which will take around 1.7 GB of disk space.

AnyCalib trained on $\mathrm{OP_{p}}$: 
```shell
python -m siclib.eval.tartanair_rays --conf anycalib_pretrained --tag anycalib_p --overwrite
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.tartanair_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen
```

### Stanford2D3D
Running the evaluation commands will download the dataset to `data/stanford2d3d` which will take around 844 MB of disk space.

AnyCalib trained on $\mathrm{OP_{p}}$: 
```shell
python -m siclib.eval.stanford2d3d_rays --conf anycalib_pretrained --tag anycalib_p --overwrite
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.stanford2d3d_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen
```

### MegaDepth (radial)
Running the evaluation commands will download the dataset to `data/megadepth2k-radial` which will take around 1.4 GB of disk space.

AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.megadepth2k_radial_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen
```

### Mono
Running the evaluation commands will download the dataset to `data/monovo2k` which will take around 445 MB of disk space.

AnyCalib trained on $\mathrm{OP_{d}}$: 
```shell
python -m siclib.eval.monovo2k_rays --conf anycalib_pretrained --tag anycalib_d --overwrite model.model_id=anycalib_dist data.cam_id=ucm
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.monovo2k_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen data.cam_id=ucm
```

### ScanNet++
To comply with ScanNet++ license, we cannot directly share its data. 
Please download the ScanNet++ dataset following the [official instructions](https://kaldir.vc.in.tum.de/scannetpp/#:~:text=the%20data%20now.-,Download%20the%20data,-To%20download%20the) and indicate the path to the root of the dataset in the following evaluation command. <br>
This needs to be provided only the first time the evaluation is run. This first time, the command will automatically copy the evaluation images under `data/scannetpp2k` which will take around 760 MB of disk space.

AnyCalib trained on $\mathrm{OP_{d}}$: 
```shell
python -m siclib.eval.scannetpp2k_rays --conf anycalib_pretrained --tag anycalib_d --overwrite model.model_id=anycalib_dist scannetpp_root=<path_to_scannetpp>
```
AnyCalib trained on $\mathrm{OP_{g}}$: 
```shell
python -m siclib.eval.scannetpp2k_rays --conf anycalib_pretrained --tag anycalib_g --overwrite model.model_id=anycalib_gen scannetpp_root=<path_to_scannetpp>
```

### LaMAR (edited)
Running the evaluation commands will download the dataset to `data/lamar2k_edit` which will take around 224 MB of disk space.

AnyCalib trained following WildCam [[8]](#8) training protocol: 
```shell
python -m siclib.eval.lamar2k_rays --conf anycalib_pretrained --tag anycalib_e --overwrite model.model_id=anycalib_edit eval.eval_on_edit=True
```

### Tartanair (edited)
Running the evaluation commands will download the dataset to `data/tartanair_edit` which will take around 488 MB of disk space.

AnyCalib trained following WildCam [[8]](#8) training protocol: 
```shell
python -m siclib.eval.tartanair_rays --conf anycalib_pretrained --tag anycalib_e --overwrite model.model_id=anycalib_edit eval.eval_on_edit=True
```

### Stanford2D3D (edited)
Running the evaluation commands will download the dataset to `data/stanford2d3d_edit` which will take around 420 MB of disk space.

AnyCalib trained on $\mathrm{OP_{p}}$, following WildCam [[8]](#8) training protocol: 
```shell
python -m siclib.eval.stanford2d3d_rays --conf anycalib_pretrained --tag anycalib_e --overwrite model.model_id=anycalib_edit eval.eval_on_edit=True
```

## Extended OpenPano Dataset
We extend the OpenPano dataset from [GeoCalib](https://github.com/cvg/GeoCalib?tab=readme-ov-file#openpano-dataset) with panoramas that not need to be aligned with the gravity direction. This extended version consists of tonemapped panoramas from [The Laval Photometric Indoor HDR Dataset](http://hdrdb.com/indoor-hdr-photometric/), [PolyHaven](https://polyhaven.com/hdris), [HDRMaps](https://hdrmaps.com/freebies/free-hdris/), [AmbientCG](https://ambientcg.com/list?type=hdri&sort=popular) and [BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree:hdr).

Before sampling images from the panoramas, first download the Laval dataset following the instructions on the [corresponding project page](http://hdrdb.com/indoor-hdr-photometric/#:~:text=HDR%20Dataset.-,Download,-To%20obtain%20the) and place the panoramas in `data/indoorDatasetCalibrated`. Then, tonemap the HDR images using the following command:
```shell
python -m siclib.datasets.utils.tonemapping --hdr_dir data/indoorDatasetCalibrated --out_dir data/laval-tonemap
```

To download the rest of the panoramas and organize all the panoramas in their corresponding splits `data/openpano_v2/panoramas/{split}`, execute:
```shell
python -m siclib.datasets.utils.download_openpano --name openpano_v2 --laval_dir data/laval-tonemap
```
The panoramas from PolyHaven, HDRMaps, AmbientCG and BlenderKit can be alternatively manually downloaded from [here](https://drive.google.com/drive/folders/1HSXKNrleJKas4cRLd1C8SqR9J1nU1-Z_?usp=sharing).

Afterwards, the different training datasets mentioned in the paper: $\mathrm{OP_{p}}$, $\mathrm{OP_{g}}$, $\mathrm{OP_{r}}$ and $\mathrm{OP_{d}}$ can be created by running the following commands. We recommend running them with the flag `device=cuda` as this significantly speeds up the creation of the datasets, but if no GPU is available, the flag can be omitted.

$\mathrm{OP_{p}}$ (will be stored under `data/openpano_v2/openpano_v2`):
```shell
python -m siclib.datasets.create_dataset_from_pano --config-name openpano_v2 device=cuda
```
$\mathrm{OP_{g}}$ (will be stored under `data/openpano_v2/openpano_v2_gen`):
```shell
python -m siclib.datasets.create_dataset_from_pano_rays --config-name openpano_v2_gen device=cuda
```
$\mathrm{OP_{r}}$ (will be stored under `data/openpano_v2/openpano_v2_radial`):
```shell
python -m siclib.datasets.create_dataset_from_pano_rays --config-name openpano_v2_radial device=cuda
```
$\mathrm{OP_{d}}$ (will be stored under `data/openpano_v2/openpano_v2_dist`):
```shell
python -m siclib.datasets.create_dataset_from_pano_rays --config-name openpano_v2_dist device=cuda
```

## Training
As with the evaluation, the training code is built upon the [`siclib`](siclib) library from [GeoCalib](https://github.com/cvg/GeoCalib). Here we adapt their instructions to AnyCalib. `siclib` can be installed executing:
```shell
pip install -e siclib
```
Once (at least one of) the [extended OpenPano Dataset](#Extended-OpenPano-Dataset) (`openpano_v2`) has been downloaded and prepared, we can train AnyCalib with it.

For training with $\mathrm{OP_{p}}$ (default):
```shell
python -m siclib.train anycalib_op_p --conf anycalib --distributed
```
Feel free to use any other experiment name. By default, the checkpoints will be written to `outputs/training/`. The default batch size is 24 which requires at least 1 NVIDIA Tesla V100 GPU with 32GB of VRAM. If only one GPU is used, the flag `--distributed` can be omitted. Configurations are managed by [Hydra](https://hydra.cc/) and can be overwritten from the command line. 

For example, for training with $\mathrm{OP_{g}}$:
```shell
python -m siclib.train anycalib_op_g --conf anycalib --distributed data.dataset_dir='data/openpano_v2/openpano_v2_gen'
```

For training with $\mathrm{OP_{d}}$:
```shell
python -m siclib.train anycalib_op_d --conf anycalib --distributed data.dataset_dir='data/openpano_v2/openpano_v2_dist'
```

For training with $\mathrm{OP_{r}}$:
```shell
python -m siclib.train anycalib_op_r --conf anycalib --distributed data.dataset_dir='data/openpano_v2/openpano_v2_radial'
```

For training with $\mathrm{OP_{p}}$ on edited (stretched and cropped) images, following the training protocol of WildCam [[8]](#8):
```shell
python -m siclib.train anycalib_op_e --conf anycalib --distributed \
data.dataset_dir='data/openpano_v2/openpano_v2' \
data.im_geom_transform.change_pixel_ar=true \
data.im_geom_transform.crop=0.5 
```

After training, the model can be evaluated using its experiment name:
```shell
python -m siclib.eval.<benchmark> --checkpoint <experiment_name> --tag <experiment_tag> --conf anycalib
```


## Acknowledgements
Thanks to the authors of [GeoCalib](https://github.com/cvg/GeoCalib) for open-sourcing the comprehensive and easy-to-use [`siclib`](https://github.com/cvg/GeoCalib/tree/main/siclib) which we use as the base of our evaluation and training code. <br>
Thanks to the authors of the [The Laval Photometric Indoor HDR Dataset](http://hdrdb.com/indoor-hdr-photometric/) for allowing us to release the weights of AnyCalib under a permissive license. <br>
Thanks also to the authors of [The Laval Photometric Indoor HDR Dataset](http://hdrdb.com/indoor-hdr-photometric/), [PolyHaven](https://polyhaven.com/hdris), [HDRMaps](https://hdrmaps.com/freebies/free-hdris/), [AmbientCG](https://ambientcg.com/list?type=hdri&sort=popular) and [BlenderKit](https://www.blenderkit.com/asset-gallery?query=category_subtree:hdr) for providing high-quality freely-available panoramas that made the training of AnyCalib possible.

## BibTex citation
If you use any ideas from the paper or code from this repo, please consider citing:
```bibtex
@article{tirado2025anycalib,
  title={{AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration}},
  author={Javier Tirado-Garín and Javier Civera},
  journal={arXiv preprint arXiv:2503.12701},
  year={2025}
}
```

## License
Code and weights are provided under the [Apache 2.0 license](LICENSE). 


## References
<a id="1">[1]</a>
Close-Range Camera Calibration. D.C. Brown, 1971.

<a id="2">[2]</a>
A Generic Camera Model and Calibration Method for Conventional, Wide-Angle, and Fish-Eye Lenses. J. Kannala, S.S. Brandt, TPAMI 2006.

<a id="3">[3]</a>
Single View Point Omnidirectional Camera Calibration from Planar Grids. C. Mei, P. Rives, ICRA, 2007.

<a id="4">[4]</a>
An Enhanced Unified Camera Model. B. Khomutenko, at al., IEEE RA-L, 2016.

<a id="5">[5]</a>
Simultaneous Linear Estimation of Multiple View Geometry and Lens Distortion. A.W. Fitzgibbon, CVPR, 2001.

<a id="6">[6]</a>
The Double Sphere Camera Model. V. Usenko, et al., 3DV, 2018.

<a id="7">[7]</a>
BabelCalib: A Universal Approach to Calibrating Central Cameras. Y. Lochman, et al., ICCV, 2021.

<a id="8">[8]</a>
Tame a Wild Camera: In-the-Wild Monocular Camera Calibration. S. Zhu, et al., NeurIPS, 2023.
