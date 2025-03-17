from pprint import pprint

import torch
from torch import Tensor

from anycalib.cameras import (
    EUCM,
    FOV,
    UCM,
    Division,
    KannalaBrandt,
    Pinhole,
    Radial,
    SimpleDivision,
    SimpleEUCM,
    SimpleKannalaBrandt,
    SimplePinhole,
    SimpleRadial,
    SimpleUCM,
)
from anycalib.cameras.base import BaseCamera
from anycalib.experiments.datasets import ControlledSyntheticData
from anycalib.ransac import RANSAC

PARAMS: dict[str, Tensor] = {
    "pinhole": torch.tensor([800.0, 700, 320, 240]),  # fx, fy, cx, cy
    "simple_pinhole": torch.tensor([750.0, 320, 240]),  # f, cx, cy
    "radial": torch.tensor([800.0, 700, 320, 240, -0.1]),  # fx, fy, cx, cy, k1, k2
    "simple_radial": torch.tensor([750.0, 320, 240, -0.1]),  # f, cx, cy, k1, k2
    "kb": torch.tensor(  # fx, fy, cx, cy, k1, k2, k3, k4
        [1900, 1800, 2150, 1450, -4.5e-2, -4.6e-3, -5.1e-4, -9.1e-5]
        # [1900, 1800, 2150, 1450, -4.5e-2]
    ),
    "simple_kb": torch.tensor(  # f, cx, cy, k1, k2, k3, k4
        [1900, 2150, 1450, -4.5e-2, -4.6e-3, -5.1e-4, -9.1e-5]
        # [1900, 2150, 1450, -4.5e-2]
    ),
    "ucm": torch.tensor([700.0, 800, 637, 511, 1.87]),
    "simple_ucm": torch.tensor([700.0, 637, 511, 1.87]),
    "eucm": torch.tensor([700.0, 800, 637, 511, 0.57, 1.11]),
    "simple_eucm": torch.tensor([700.0, 637, 511, 0.57, 1.11]),
    # "division": torch.tensor([700.0, 800, 637, 511, -0.4]),
    "division": torch.tensor([700.0, 800, 637, 511, -0.4, -0.05]),
    "simple_division": torch.tensor([700.0, 637, 511, -0.4, -0.05]),
    "fov": torch.tensor([700.0, 800, 637, 511, 0.9]),
}

CAMS: dict[str, BaseCamera] = {
    "pinhole": Pinhole(),
    "simple_pinhole": SimplePinhole(),
    "radial": Radial(num_k=PARAMS["radial"].shape[-1] - 4),
    "simple_radial": SimpleRadial(num_k=PARAMS["simple_radial"].shape[-1] - 3),
    "kb": KannalaBrandt(num_k=PARAMS["kb"].shape[-1] - 4),
    "simple_kb": SimpleKannalaBrandt(num_k=PARAMS["simple_kb"].shape[-1] - 3),
    "ucm": UCM(),
    "simple_ucm": SimpleUCM(),
    "eucm": EUCM(),
    "simple_eucm": SimpleEUCM(),
    # "division": Division(num_k=1, complex_tol=1e-4),
    "division": Division(num_k=2, complex_tol=1e-4),
    "simple_division": SimpleDivision(num_k=2, complex_tol=1e-4),
    "fov": FOV(),
}

DATA_FOV: dict[str, float] = {
    "pinhole": 100,
    "simple_pinhole": 100,
    "radial": 100,
    "simple_radial": 100,
    "kb": 130,
    "simple_kb": 130,
    "ucm": 120,
    "simple_ucm": 120,
    "eucm": 120,
    "simple_eucm": 120,
    "division": 100,
    "simple_division": 100,
    "fov": 120,
}


def get_2d3d_correspondences(
    cam_id: str,
    params: Tensor,
    n: int = 25,
    noise: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate 2D-3D correspondences."""
    dset = ControlledSyntheticData(device="cpu", seed=0)
    data = dset(
        cam_id,
        params,
        n_rays=n,
        fov=DATA_FOV[cam_id],
        std_noise=noise,
        force_centered_cxcy=True,
    )
    im_coords = data["im_coords"]
    unit_bearings = data["bearings"]
    new_params = data["intrinsics"]
    return im_coords, unit_bearings, new_params


def to_device(device: str, *args):
    """Move tensors to device."""
    return tuple(arg.to(device) for arg in args)


########################################################################################


def is_cyclic_project_unproject(cam_id: str, **kwargs) -> bool:
    """Assess cycle consistency of projection and unprojection."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)
    # test
    unprojs, valid_u = cam.unproject(params, im_coords, **kwargs)
    projs, valid_p = cam.project(params, unprojs, **kwargs)
    if valid_u is not None:
        assert valid_u.all()
    if valid_p is not None:
        assert valid_p.all()
    # diff = (unit_bearings - unprojs).abs()
    # print(f"{cam_id + ':':<16} Max: {diff.max():.2e}\tMedian: {diff.median():.2e}")
    diff = (im_coords - projs).abs().ravel().sort().values
    print(
        f"{cam_id + ':':<16} Max: {diff.max():.2e}\tMedian: {diff.median():.2e}\t"
        f"Min: {diff.min():.2e}\t"
        f"95%: {diff[int(0.95 * len(diff))]:.2e}"
    )
    return torch.allclose(unit_bearings, unprojs) and torch.allclose(im_coords, projs)


def test_cyclic_projection_unprojection():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
        "kb": {},
        "simple_kb": {},
        "ucm": {},
        "simple_ucm": {},
        "eucm": {},
        "simple_eucm": {},
        "division": {},
        "simple_division": {},
        "fov": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = is_cyclic_project_unproject(cam_id, **kwargs)
    assert all(checks.values()), checks


########################################################################################


def fit(cam_id: str, **kwargs) -> bool:
    """Fit a camera model to a set of 2D-3D correspondences."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params, 100)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)
    # fit camera model
    im_coords, unit_bearings = to_device("cpu", im_coords, unit_bearings)
    params_hat, _ = cam.fit(im_coords, unit_bearings, **kwargs)
    params_hat = params_hat.to("cpu")
    # error
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    # raise
    assert params.shape == params_hat.shape
    return torch.allclose(params, params_hat, atol=1e-4, rtol=1e-4)


def test_fit():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
        "kb": {},
        "simple_kb": {},
        "ucm": {},
        "simple_ucm": {},
        "eucm": {},
        "simple_eucm": {},
        "division": {},
        "simple_division": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit(cam_id, **kwargs)
    assert all(checks.values()), checks


#######################################################################################


def fit_with_covs(cam_id: str, *args, **kwargs) -> bool:
    """Fit a camera model to a set of 2D-3D correspondences with covariances."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params)
    # only *diagonal* elements of the covariances
    covs = torch.ones((im_coords.shape[0], 2), dtype=params.dtype)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)
    covs = covs[None].expand(b, -1, -1)
    # fit camera model
    params_hat, _ = cam.fit(
        im_coords, unit_bearings, covs=covs, params0=params, **kwargs
    )
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    return torch.allclose(params, params_hat, atol=1e-4, rtol=1e-4)


def test_fit_with_covs():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit_with_covs(cam_id, **kwargs)
    assert all(checks.values()), checks


########################################################################################


def fit_minimal(cam_id: str, *args, **kwargs) -> bool:
    """Estimation of intrinsics with a minimal sample."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    # solve with ransac
    ransac = RANSAC()
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params, 1000)
    params_hat, _ = ransac(cam, im_coords, unit_bearings)
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    atol = rtol = 1e-4
    if cam_id in ["simple_kb"]:
        atol = rtol = 1e-2
    # raise
    return torch.allclose(params, params_hat, atol=atol, rtol=rtol)


def test_fit_minimal():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
        "kb": {},
        "simple_kb": {},
        "ucm": {},
        "simple_ucm": {},
        "eucm": {},
        "simple_eucm": {},
        "division": {},
        "simple_division": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit_minimal(cam_id, **kwargs)
    assert all(checks.values()), pprint(checks)


########################################################################################


def fit_minimal_with_cxcy(cam_id: str, *args, **kwargs) -> bool:
    """Estimation of intrinsics with a minimal sample."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    # solve with ransac
    ransac = RANSAC()
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params, 1000)
    cxcy = cam.params_to_dict(params)["c"]
    params_hat, _ = ransac(cam, im_coords, unit_bearings, cxcy=cxcy)
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    return torch.allclose(params, params_hat, atol=1e-1, rtol=1e-1)


def test_fit_minimal_with_cxcy():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
        "kb": {},
        "simple_kb": {},
        "ucm": {},
        "simple_ucm": {},
        "eucm": {},
        "simple_eucm": {},
        "division": {},
        "simple_division": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit_minimal_with_cxcy(cam_id, **kwargs)
    assert all(checks.values()), pprint(checks)


########################################################################################


def fit_with_cxcy(cam_id: str, **kwargs) -> bool:
    """Fit a camera model to a set of 2D-3D correspondences."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params)
    cxcy = cam.params_to_dict(params)["c"]
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)
    cxcy = cxcy[None].expand(b, -1)
    # fit camera model
    params_hat, _ = cam.fit(im_coords, unit_bearings, cxcy=cxcy)
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    return torch.allclose(params, params_hat, atol=1e-4, rtol=1e-4)


def test_fit_with_cxcy():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
        "kb": {},
        "simple_kb": {},
        "ucm": {},
        "simple_ucm": {},
        "eucm": {},
        "simple_eucm": {},
        "division": {},
        "simple_division": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit_with_cxcy(cam_id, **kwargs)
    assert all(checks.values()), checks


########################################################################################


def fit_with_covs_cxcy(cam_id: str, *args, **kwargs) -> bool:
    """Fit a camera model to a set of 2D-3D correspondences with covariances."""
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params)
    cxcy = cam.params_to_dict(params)["c"]
    # only *diagonal* elements of the covariances
    covs = torch.ones((im_coords.shape[0], 2), dtype=params.dtype)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)
    cxcy = cxcy[None].expand(b, -1)
    covs = covs[None].expand(b, -1, -1)
    # fit camera model
    params_hat, _ = cam.fit(
        im_coords, unit_bearings, cxcy=cxcy, covs=covs, params0=params
    )
    diff = (params - params_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{params}\n{params_hat}")
    assert params.shape == params_hat.shape
    return torch.allclose(params, params_hat, atol=1e-4, rtol=1e-4)


def test_fit_with_covs_cxcy():
    cam_cfgs = {
        "pinhole": {},
        "simple_pinhole": {},
        "radial": {},
        "simple_radial": {},
    }
    checks = {}
    for cam_id, kwargs in cam_cfgs.items():
        checks[cam_id] = fit_with_covs_cxcy(cam_id, **kwargs)
    assert all(checks.values()), checks


########################################################################################


def jac_bearings_wrt_params(cam_id: str) -> bool:
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)

    def wrap_unproject(params, im_coords):
        """Wrapper for unproject method to not return the integer/boolean tensor."""
        bearings, _ = cam.unproject(params, im_coords)
        return bearings

    # jacobian
    idx = torch.arange(b)
    jac_autograd = torch.autograd.functional.jacobian(
        wrap_unproject, (params, im_coords)
    )[0]
    jac_autograd = jac_autograd[idx, ..., idx, :]
    jac_analytic = cam.jac_bearings_wrt_params(params, unit_bearings, im_coords)
    diff = (jac_autograd - jac_analytic).abs()
    print(f"{cam_id}: {diff.max():.2e} {diff.median():.2e} {(diff > 1e-4).sum()}")
    print(f"\n{jac_autograd}\n\n{jac_analytic}")
    print(f"\n{jac_autograd.shape=}\n{jac_analytic.shape=}")
    return torch.allclose(jac_autograd, jac_analytic, rtol=1e-4)


def test_jac_bearings_wrt_params():
    cams = (
        "pinhole",
        "simple_pinhole",
        "radial",
        "simple_radial",
        "kb",
        "simple_kb",
        "ucm",
        "simple_ucm",
        "eucm",
        "simple_eucm",
        "division",
        "simple_division",
    )
    checks = {}
    for cam_id in cams:
        checks[cam_id] = jac_bearings_wrt_params(cam_id)
    assert all(checks.values()), checks


########################################################################################


def jac_bearings_wrt_im(cam_id: str) -> bool:
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params, 100)
    # batchify
    b = 2
    im_coords = im_coords[None].expand(b, -1, -1)
    unit_bearings = unit_bearings[None].expand(b, -1, -1)
    params = params[None].expand(b, -1)

    def wrap_unproject(params, im_coords):
        """Wrapper for unproject method to not return the integer/boolean tensor."""
        bearings, _ = cam.unproject(params, im_coords)
        return bearings

    # jacobian
    idx_b = torch.arange(b)
    idx_n = torch.arange(im_coords.shape[-2])
    jac_autograd = torch.autograd.functional.jacobian(
        wrap_unproject, (params, im_coords)
    )[1]
    jac_autograd = jac_autograd[:, idx_n, ..., idx_n, :][:, idx_b, :, idx_b]
    jac_analytic = cam.jac_bearings_wrt_imcoords(params, unit_bearings, im_coords)
    diff = (jac_autograd - jac_analytic).abs()
    print(f"{cam_id}: {diff.max():.2e} {diff.median():.2e} {(diff > 1e-4).sum()}")
    print(f"\n{jac_autograd}\n\n{jac_analytic}")
    return torch.allclose(jac_autograd, jac_analytic, rtol=1e-4)


def test_jac_bearings_wrt_im():
    cams = (
        "pinhole",
        "simple_pinhole",
        "radial",
        "simple_radial",
        "kb",
        "simple_kb",
        "ucm",
        "simple_ucm",
        "eucm",
        "simple_eucm",
        "division",
        "simple_division",
    )
    checks = {}
    for cam_id in cams:
        checks[cam_id] = jac_bearings_wrt_im(cam_id)
    assert all(checks.values()), checks


########################################################################################


def get_sensor_radii(cam: BaseCamera, params: Tensor, im_coords: Tensor) -> Tensor:
    """Compute the radii of the sensor plane."""
    p_dict = cam.params_to_dict(params)
    f, c = p_dict["f"], p_dict["c"]
    sensor_coords = (im_coords - c[..., None, :]) / f[..., None, :]
    return torch.linalg.norm(sensor_coords, dim=-1)


def fit_from_radii(cam_id: str) -> bool:
    cam = CAMS[cam_id]
    params = PARAMS[cam_id]
    im_coords, unit_bearings, params = get_2d3d_correspondences(cam_id, params, 100)
    # batchify
    # b = 2
    # im_coords = im_coords[None].expand(b, -1, -1)
    # unit_bearings = unit_bearings[None].expand(b, -1, -1)
    # params = params[None].expand(b, -1)
    # fit distortion params given radii
    r = get_sensor_radii(cam, params, im_coords)
    R = torch.linalg.norm(unit_bearings[..., :2], dim=-1)
    k = cam.params_to_dict(params)["k"]  # GT
    k_hat, _ = cam.fit_dist_from_radii(r, R, unit_bearings[..., 2])
    # error
    diff = (k - k_hat).abs().max()
    print(f"{cam_id}: {diff:.2e}\n{k}\n{k_hat}")
    assert k.shape == k_hat.shape
    return torch.allclose(k, k_hat, atol=1e-4, rtol=1e-4)


def test_fit_from_radii():
    cams = (
        # "pinhole",
        # "simple_pinhole",
        # "radial",
        # "simple_radial",
        # "kb",
        # "simple_kb",
        "ucm",
        "simple_ucm",
        "eucm",
        "simple_eucm",
        "division",
        "simple_division",
    )
    checks = {}
    for cam_id in cams:
        checks[cam_id] = fit_from_radii(cam_id)
    assert all(checks.values()), checks
