from math import pi

import torch
import torch.nn.functional as F

from anycalib.manifolds import Unit3

N = 100


def sample_unit_vecs(n: int = 100, seed: int = 0) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    x = 2 * torch.rand((n, 3), generator=rng) - 1
    x /= x.norm(dim=-1, keepdim=True)
    return x


def test_cyclic_explog():
    # sample data
    n = 100
    x = sample_unit_vecs(n, seed=0)
    y = sample_unit_vecs(n, seed=1)
    # test cycle consistency
    tangent_vecs = Unit3.logmap(x, y)
    manif_vecs = Unit3.expmap(x, tangent_vecs)
    diff = (y - manif_vecs).abs().max()
    assert torch.allclose(y, manif_vecs, atol=1e-5), diff


def test_cyclic_explog_close_to_parallel():
    # sample data
    n = 30
    x = sample_unit_vecs(n=n, seed=0)
    # add perturbations [0, 1e-30 to 0.1]
    zero = torch.zeros(1, dtype=x.dtype)
    perturb = torch.cat((zero, torch.logspace(-30, -1, n - 1, dtype=x.dtype)))
    y = x + perturb[:, None]
    y /= y.norm(dim=-1, keepdim=True)
    # test cycle consistency
    tangent_vecs = Unit3.logmap(x, y)
    manif_vecs = Unit3.expmap(x, tangent_vecs)
    diff = (y - manif_vecs).abs().max()
    assert torch.allclose(y, manif_vecs), diff


def test_gradcheck():
    def jac_logmap_wrt_refvecs_wrap(x, y):
        jac, _ = Unit3.jac_logmap_wrt_refvecs(x, y)
        return jac

    # sample data
    n = 10
    x = sample_unit_vecs(n=n, seed=0).to(dtype=torch.float64)
    y = sample_unit_vecs(n=n, seed=1).to(dtype=torch.float64)
    x.requires_grad_()
    y.requires_grad_()
    local_coords = Unit3.logmap(x.detach(), y.detach())
    # methods to check
    methods = {
        "expmap": (Unit3.expmap, (x, local_coords)),
        "logmap": (Unit3.logmap, (x, y)),
        "jac_logmap_wrt_vecs": (Unit3.jac_logmap_wrt_vecs, (x, y)),
        "jac_logmap_wrt_refvecs": (jac_logmap_wrt_refvecs_wrap, (x, y)),
    }
    # test gradcheck
    checks = {
        name: torch.autograd.gradcheck(func, args)
        for name, (func, args) in methods.items()
    }
    assert all(checks.values()), checks


def test_jac_logmap_wrt_vecs():
    # sample data
    n = 100
    x = sample_unit_vecs(n=n, seed=0)
    # restrict y to distances below pi/2 to reflect expected distances
    torch.manual_seed(123)
    angles = 0.5 * pi * torch.rand(n, 1)
    local_coords = angles * F.normalize(torch.rand(n, 2) - 0.5, dim=-1)
    y = Unit3.expmap(x, local_coords)

    idx = torch.arange(n)
    jac_autograd = torch.autograd.functional.jacobian(Unit3.logmap, (x, y))[1]
    jac_autograd = jac_autograd[idx, :, idx]
    jac_analytic = Unit3.jac_logmap_wrt_vecs(x, y)

    diff = (jac_autograd - jac_analytic).abs()
    print(jac_autograd.isnan().sum(), jac_analytic.isnan().sum())
    print(diff.max(), diff.median(), (diff > 1e-4).sum())
    assert torch.allclose(jac_autograd, jac_analytic, atol=1e-3, rtol=1e-4)


def test_jac_logmap_wrt_vecs_at_z1():
    # sample data
    n = 100
    z1 = torch.tensor([0.0, 0.0, 1.0])
    # restrict y to distances below pi/2 to reflect expected distances
    torch.manual_seed(123)
    angles = 0.5 * pi * torch.rand(n, 1)
    local_coords = angles * F.normalize(torch.rand(n, 2) - 0.5, dim=-1)
    y = Unit3.expmap(z1, local_coords)

    idx = torch.arange(n)
    jac_autograd = torch.autograd.functional.jacobian(Unit3.logmap_at_z1, y)
    jac_autograd = jac_autograd[idx, :, idx]  # type: ignore
    jac_analytic = Unit3.jac_logmap_wrt_vecs_at_z1(y)

    diff = (jac_autograd - jac_analytic).abs()
    print(jac_autograd.isnan().sum(), jac_analytic.isnan().sum())
    print(diff.max(), diff.median(), (diff > 1e-4).sum())
    assert torch.allclose(jac_autograd, jac_analytic, atol=1e-3, rtol=1e-4)


def test_jac_logmap_wrt_refvecs():
    # sample data
    n = 100
    x = sample_unit_vecs(n=n, seed=0)
    # restrict y to distances below pi/2 to reflect expected distances
    torch.manual_seed(123)
    angles = 0.5 * pi * torch.rand(n, 1)
    local_coords = angles * F.normalize(torch.rand(n, 2) - 0.5, dim=-1)
    y = Unit3.expmap(x, local_coords)

    idx = torch.arange(n)
    jac_autograd = torch.autograd.functional.jacobian(Unit3.logmap, (x, y))[0]
    jac_autograd = jac_autograd[idx, :, idx]
    jac_analytic, _ = Unit3.jac_logmap_wrt_refvecs(x, y)

    diff = (jac_autograd - jac_analytic).abs()
    print(jac_autograd.isnan().sum(), jac_analytic.isnan().sum())
    print(diff.min(), diff.max(), diff.median(), (diff > 1e-4).sum())
    print(f"{jac_autograd}\n\n{jac_analytic}")
    assert torch.allclose(jac_autograd, jac_analytic, atol=1e-3, rtol=1e-4)
