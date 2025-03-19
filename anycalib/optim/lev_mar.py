import torch
from torch import Tensor

from anycalib.cameras.base import BaseCamera
from anycalib.optim.gauss_newton import GaussNewtonCalib


def expand_updates(delta: Tensor, optim_idx: list[int] | None, dim: int) -> Tensor:
    """"""
    if optim_idx is None:
        return delta
    updates = delta.new_zeros(delta.shape[:-1] + (dim,))
    updates[..., optim_idx] = delta
    return updates


class LevMarCalib(GaussNewtonCalib):
    """Levenberg-Marquardt nonlinear optimization for single-view camera calibration."""

    DEFAULT_CONF = {
        "max_iters": 10,
        "res_tangent": "fitted",
        "solver": "normal",
    }

    def __init__(self, cfg: dict | None = None):
        cfg = self.DEFAULT_CONF | (cfg or {})
        max_iters = cfg["max_iters"]
        res_tangent = cfg["res_tangent"]

        assert max_iters >= 0, "max_iters must be non-negative"
        if res_tangent == "observed":
            self.res_jac_fun = self.res_and_jac_in_obs_tangent
        elif res_tangent == "fitted":
            self.res_jac_fun = self.res_and_jac_in_fitted_tangent
        elif res_tangent == "z1":
            self.res_jac_fun = self.res_and_jac_in_z1_tangent
        else:
            raise ValueError(
                "`res_tangent` must be 'observed', 'fitted' or 'z1'. However, got: "
                f"'{res_tangent=}'."
            )
        self.res_tangent = res_tangent
        self.max_iters = max_iters

        self.solver = "normal"  # cfg["solver"]
        self.compute_update = self.solve_normal_eqs

    def __call__(
        self,
        cam: BaseCamera,
        params0: Tensor,
        im_coords: Tensor,
        observations: Tensor,
        weights: None | Tensor = None,
        fix_cxcy: bool = False,
        fix_params: None | tuple[str, ...] | list[str] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Iterative refinement of intrinsic parameters using Gauss-Newton

        Args:
            cam: camera model.
            params0: (..., D) initial guess of intrinsic parameters.
            im_coords: (..., N, 2) image coordinates of observed points.
            observations: (..., N, D) observations. D=3 if they represent unit bearing vectors.
                D=2 if they represent 2D points expressed in a tangent plane.
            weights: (..., N, 2) diagonal elements of the *inverse* bearing covariances
                expressed in the tangent space of the bearing mean (to be estimated).
            fix_cxcy: whether to mantain fixed the principal point or optimize it.
            fix_idx: parameter names to maintain fixed during the optimization.
                If None, all parameters are optimized unless `fix_cxcy` is True. In which
                case, the principal point is fixed.

        Returns:
            (..., D) refined intrinsic parameters.
            (...,) initial cost.
            (...,) final cost.
            (..., D, D) approximate covariance inverse of the solution.
        """
        assert observations.shape[-1] in (2, 3), f"Invalid {observations.shape[-1]=}"
        params = params0
        weights_ = None if weights is None else weights[..., None]  # (..., N, 2, 1)

        # obtain indexes of parameters to be optimized
        if fix_cxcy:
            fix_params = (
                ["cx", "cy"] if fix_params is None else list(fix_params) + ["cx", "cy"]
            )
        dim = params0.shape[-1]
        optim_idx = (
            None
            if fix_params is None
            else [
                idx
                for (n, idx), _ in zip(cam.PARAMS_IDX.items(), range(dim))
                if n not in fix_params
            ]
        )

        # current estimates
        residuals, jac, valid = self.res_jac_fun(cam, observations, params, im_coords)
        jac = cam.get_optim_jac(jac, params)  # Jacobian needed during optimization
        jac = jac if optim_idx is None else jac[..., optim_idx]
        cost0 = cost = self.compute_cost(residuals, weights, valid)
        damping = self.init_damping(jac)

        for _ in range(self.max_iters):
            # solve normal equations J^T W J delta = -J^T W r and update state
            delta = self.compute_update(jac, -residuals, damping, weights_, valid)  # (..., D) # fmt: skip
            params = cam.get_optim_update(params, expand_updates(delta, optim_idx, dim))
            residuals, jac, valid = self.res_jac_fun(
                cam, observations, params, im_coords
            )
            jac = cam.get_optim_jac(jac, params)
            jac = jac if optim_idx is None else jac[..., optim_idx]
            # update damping term
            new_cost = self.compute_cost(residuals, weights, valid)
            damping = damping * torch.where((new_cost < cost).unsqueeze(-1), 0.1, 10)
            cost = new_cost

        icovs = self.estimate_inverse_covariance(jac, valid, weights_)
        return params, cost0, cost, icovs

    @staticmethod
    def init_damping(jac: Tensor) -> Tensor:
        """Initialize damping term

        Args:
            jac: (..., N, 2, D) Jacobian of residuals w.r.t. intrinsic parameters.

        Returns:
            (..., 1) initial damping term.
        """
        # based on [Hartley, Zisserman, 2004]:  1e-3 * (trace(JTWJ) / D)
        return 1e-3 * (jac**2).sum((-3, -2, -1)) / jac.shape[-1]

    @staticmethod
    def solve_normal_eqs(
        Js: Tensor,
        neg_res: Tensor,
        damping: Tensor,
        Ws: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Solve normal equations with optional weights and mask:
            ( J^TWJ + μ*diag(J^TWJ) ) Δ = -J^TW r

        This method can actually be used for residuals of any dimensionality. However,
        for clarity, in this docstring, we assume 2D residuals (those expressed in the
        tangent space of each bearing vector).

        Args:
            Js: (..., N, 2, D) stacked Jacobian for each tangent-space 2D error.
            neg_res: (..., N, 2) *negative* residuals.
            damping: (..., 1) damping term.
            Ws: (..., N, 2, 1) *diagonal* of the weight matrices for each 2D error.
            mask: (..., N) boolean mask for valid observations.

        Returns:
            (..., D) Gauss-Newton step.
        """
        WJs = Js if Ws is None else Ws * Js  # (..., N, 2, D)
        WJs = WJs if mask is None else WJs * mask[..., None, None]
        JtW = WJs.flatten(-3, -2).transpose(-1, -2)  # (..., D, 2*N)
        JtWr = (JtW @ neg_res.flatten(-2, -1)[..., None]).squeeze(-1)  # (..., D)
        JtWJ = JtW @ Js.flatten(-3, -2)  # (..., D, D)
        # damping term
        JtWJ = JtWJ + (damping * JtWJ.diagonal(dim1=-2, dim2=-1)).diag_embed()
        # solve
        delta, info = torch.linalg.solve_ex(JtWJ, JtWr)  # (..., D)
        delta = torch.where((info == 0)[..., None], delta, 0)
        # assert (info == 0).all(), "LM failed to converge"
        return delta
