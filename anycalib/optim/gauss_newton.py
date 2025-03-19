import torch
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3


def expand_updates(delta: Tensor, optim_idx: list[int] | None, dim: int) -> Tensor:
    """"""
    if optim_idx is None:
        return delta
    updates = delta.new_zeros(delta.shape[:-1] + (dim,))
    updates[..., optim_idx] = delta
    return updates


class GaussNewtonCalib:
    """Gauss-Newton nonlinear optimization for single-view camera calibration."""

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
                "`res_tangent` must be 'observed', 'fitted', or 'z1'. However, got: "
                f"'{res_tangent=}'."
            )
        self.res_tangent = res_tangent
        self.max_iters = max_iters

        self.solver = cfg["solver"]
        if self.solver == "normal":
            # form and solve normal eqss with LU decomposition
            self.compute_update = self.solve_normal_eqs
        elif self.solver == "qr":
            # don't form normal eqs. and solve with QR decomposition
            self.compute_update = self.solve_qr
        elif self.solver == "pinv":
            # use pseudo-inverse. Useful for ill-conditioned problems in CUDA
            self.compute_update = self.solve_pinv
        else:
            raise ValueError(
                "`solver` must be 'normal' or 'qr'. However, got: " f"'{self.solver=}'."
            )

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
        cost0 = self.compute_cost(residuals, weights, valid)

        for _ in range(self.max_iters):
            # solve normal equations J^T W J delta = -J^T W r and update state
            delta = self.compute_update(jac, -residuals, weights_, valid)  # (..., D)
            params = cam.get_optim_update(params, expand_updates(delta, optim_idx, dim))
            residuals, jac, valid = self.res_jac_fun(
                cam, observations, params, im_coords
            )
            jac = cam.get_optim_jac(jac, params)
            jac = jac if optim_idx is None else jac[..., optim_idx]
        final_cost = self.compute_cost(residuals, weights, valid)
        icovs = self.estimate_inverse_covariance(jac, valid, weights_)
        return params, cost0, final_cost, icovs

    @staticmethod
    def res_and_jac_in_obs_tangent(
        cam: BaseCamera,
        observed_bearings: Tensor,
        params: Tensor,
        im_coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Residuals and Jacobian of residuals w.r.t. intrinsics, with residuals
        expressed in the tangent space of the observed (input) bearings.

        Args:
            cam: camera model.
            observed_bearings: (..., N, 3) observed unit bearing vectors.
            params: (..., D) intrinsic parameters.
            im_coords: (..., N, 2) image coordinates of observed points.

        Returns:
            (..., N, 2) residuals in the tangent space of each observed bearing.
            (..., N, 2, D) Jacobian of residuals w.r.t. intrinsic parameters.
            (..., N) None (if all valid) or boolean mask for valid observations.
        """
        # (..., N, 3) current best-fit (mean) bearing vectors.
        best_fit_bearings, valid = cam.unproject(params, im_coords)
        # residuals in the tangent space of each observed bearing
        residuals = Unit3.logmap(observed_bearings, best_fit_bearings)  # (.., N, 2)
        # dres/dbearing = dlog/dbearing
        jac_res_wrt_bearings = Unit3.jac_logmap_wrt_vecs(
            observed_bearings, best_fit_bearings
        )  # (..., N, 2, 3)
        # dres/dparams = dres/dbearing * dbearing/dparams
        jac_bearing_wrt_params = cam.jac_bearings_wrt_params(
            params, best_fit_bearings, im_coords
        )  # (..., N, 3, D)
        jac_res_wrt_params = ut.fast_small_matmul(
            jac_res_wrt_bearings, jac_bearing_wrt_params
        )
        return residuals, jac_res_wrt_params, valid

    @staticmethod
    def res_and_jac_in_fitted_tangent(
        cam: BaseCamera,
        observed_bearings: Tensor,
        params: Tensor,
        im_coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Residuals and Jacobian of residuals w.r.t. intrinsics, with residuals
        expressed in the tangent space of the bearings that are being iteratively fitted

        Args:
            cam: camera model.
            observed_bearings: (..., N, 3) observed unit bearing vectors.
            params: (..., D) intrinsic parameters.
            im_coords: (..., N, 2) image coordinates of observed points.

        Returns:
            (..., N, 2) residuals in the tangent space of each estimated bearing mean.
            (..., N, 2, D) Jacobian of residuals w.r.t. intrinsic parameters.
            (..., N) boolean mask for valid observations for fitting.
        """
        # (..., N, 3) current best-fit (mean) bearing vectors.
        best_fit_bearings, valid = cam.unproject(params, im_coords)
        # residuals in the tangent space of each estimated bearing mean
        residuals = Unit3.logmap(best_fit_bearings, observed_bearings)  # (.., N, 2)
        # dres/dbearing = dlog/dbearing
        jac_res_wrt_bearings, valid_jacs = Unit3.jac_logmap_wrt_refvecs(
            best_fit_bearings, observed_bearings
        )  # (..., N, 2, 3)
        # dres/dparams = dres/dbearing * dbearing/dparams
        jac_bearing_wrt_params = cam.jac_bearings_wrt_params(
            params, best_fit_bearings, im_coords
        )  # (..., N, 3, D)
        jac_res_wrt_params = ut.fast_small_matmul(
            jac_res_wrt_bearings, jac_bearing_wrt_params
        )
        # update valid observations for fitting
        valid = valid_jacs if valid is None else valid & valid_jacs
        return residuals, jac_res_wrt_params, valid

    @staticmethod
    def res_and_jac_in_z1_tangent(
        cam: BaseCamera,
        observed_tcoords: Tensor,
        params: Tensor,
        im_coords: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Residuals and Jacobian of residuals w.r.t. intrinsics, with residuals
        expressed in the tangent space at (0, 0, 1).

        Args:
            cam: camera model.
            observed_tcoords: (..., N, 2) observed tangent coordinates expressed in the
                tangent space at (0, 0, 1).
            params: (..., D) intrinsic parameters.
            im_coords: (..., N, 2) image coordinates of observed points.

        Returns:
            (..., N, 2) residuals in the tangent space of each estimated bearing mean.
            (..., N, 2, D) Jacobian of residuals w.r.t. intrinsic parameters.
            (..., N) updated boolean mask for valid observations for fitting.
        """
        # (..., N, 2) current best-fit (mean) tangent coordinates at (0, 0, 1).
        best_fit_bearings, valid = cam.unproject(params, im_coords)
        best_fit_tcoords = Unit3.logmap_at_z1(best_fit_bearings)
        # residuals in the tangent space at (0, 0, 1)
        residuals = best_fit_tcoords - observed_tcoords
        # dres/dbearing = dlog/dbearing (..., N, 2, 3)
        jac_res_wrt_bearings = Unit3.jac_logmap_wrt_vecs_at_z1(best_fit_bearings)
        # dres/dparams = dres/dbearing * dbearing/dparams
        jac_bearing_wrt_params = cam.jac_bearings_wrt_params(
            params, best_fit_bearings, im_coords
        )  # (..., N, 3, D)
        jac_res_wrt_params = ut.fast_small_matmul(
            jac_res_wrt_bearings, jac_bearing_wrt_params
        )
        return residuals, jac_res_wrt_params, valid

    @staticmethod
    def solve_normal_eqs(
        Js: Tensor,
        neg_res: Tensor,
        Ws: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Solve GN normal equations with optional weights and mask: J^TWJ Δ = -J^TW r

        This method can actually be used for residuals of any dimensionality. However,
        for clarity, in this docstring, we assume 2D residuals (those expressed in the
        tangent space of each bearing vector).

        Args:
            Js: (..., N, 2, D) stacked Jacobian for each tangent-space 2D error.
            neg_res: (..., N, 2) *negative* residuals.
            Ws: (..., N, 2, 1) *diagonal* of the weight matrices for each 2D error.
            mask: (..., N) boolean mask for valid observations.

        Returns:
            (..., D) Gauss-Newton step.
        """
        WJs = Js if Ws is None else Ws * Js  # (..., N, 2, D)
        WJs = WJs if mask is None else WJs * mask[..., None, None]
        JtW = WJs.flatten(-3, -2).transpose(-1, -2)  # (..., D, 2*N)
        JtWJ = JtW @ Js.flatten(-3, -2)  # (..., D, D)
        JtWr = JtW @ neg_res.flatten(-2, -1)[..., None]  # (..., D, 1)
        delta, info = torch.linalg.solve_ex(JtWJ, JtWr.squeeze(-1))  # (..., D)
        delta = torch.where((info == 0)[..., None], delta.nan_to_num(0, 0, 0), 0)
        # assert (info == 0).all(), "Gauss-Newton failed to converge"
        return delta

    @staticmethod
    def solve_qr(
        Js: Tensor,
        neg_res: Tensor,
        Ws: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute update with QR decomposition.

        Args:
            Js: (..., N, 2, D) stacked Jacobian for each tangent-space 2D error.
            neg_res: (..., N, 2) *negative* residuals.
            Ws: (..., N, 2, 1) *diagonal* of the weight matrices for each 2D error.
            mask: (..., N) boolean mask for valid observations.

        Returns:
            (..., D) Gauss-Newton step.
        """
        Ws = None if Ws is None else torch.sqrt(Ws)
        WJs = Js if Ws is None else Ws * Js  # (..., N, 2, D)
        WJs = WJs if mask is None else WJs * mask[..., None, None]
        Wr = neg_res if Ws is None else (Ws * neg_res[..., None]).squeeze(-1)
        Wr = Wr if mask is None else Wr * mask[..., None]
        results = torch.linalg.lstsq(
            WJs.flatten(-3, -2), Wr.flatten(-2, -1), driver="gels"
        )
        return results.solution

    @staticmethod
    def solve_pinv(
        Js: Tensor,
        neg_res: Tensor,
        Ws: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Compute update with pseudo-inverse.

        Args:
            Js: (..., N, 2, D) stacked Jacobian for each tangent-space 2D error.
            neg_res: (..., N, 2) *negative* residuals.
            Ws: (..., N, 2, 1) *diagonal* of the weight matrices for each 2D error.
            mask: (..., N) boolean mask for valid observations.

        Returns:
            (..., D) Gauss-Newton step.
        """
        WJs = Js if Ws is None else Ws * Js  # (..., N, 2, D)
        WJs = WJs if mask is None else WJs * mask[..., None, None]
        JtW = WJs.flatten(-3, -2).transpose(-1, -2)  # (..., D, 2*N)
        JtWJ = JtW @ Js.flatten(-3, -2)  # (..., D, D)
        JtWr = JtW @ neg_res.flatten(-2, -1)[..., None]  # (..., D, 1)
        delta = (torch.linalg.pinv(JtWJ) @ JtWr).squeeze(-1)  # (..., D)
        return delta

    @staticmethod
    def compute_cost(
        res: Tensor, weights: Tensor | None, mask: Tensor | None
    ) -> Tensor:
        """Compute cost with optional weights and mask.

        Args:
            res: (..., N, 2) residuals.
            weights: (..., N, 2) diagonal elements of the *inverse* weight matrices.
            mask: (..., N) boolean mask for valid observations.

        Returns:
            (...,) cost.
        """
        cost = (res**2).sum(-1) if weights is None else (res**2 * weights).sum(-1)
        cost = cost.mean(-1) if mask is None else (cost * mask).sum(-1) / mask.sum(-1)
        return cost

    @staticmethod
    def estimate_inverse_covariance(
        Js: Tensor, mask: Tensor | None = None, Ws: Tensor | None = None
    ) -> Tensor:
        """Aproximate the covariance inverse of the solution as J^T W J.

        This aproximation is appropriate for residuals that follow a Gaussian distribution
        whose covariance matrices are given by inv(Ws) and when they are evaluated at the
        MLE of the parameters.

        Args:
            jac: (..., N, 2, D) Jacobian of residuals w.r.t. intrinsic parameters.
            mask: (..., N) boolean mask for valid observations
            Ws: (..., N, 2, 1) *diagonal* of the weight matrices for each 2D error.

        Returns:
            (..., D, D) covariance inverse of the solution.
        """
        WJs = Js if Ws is None else Ws * Js  # (..., N, 2, D)
        WJs = WJs if mask is None else WJs * mask[..., None, None]
        JtW = WJs.flatten(-3, -2).transpose(-1, -2)  # (..., D, 2*N)
        icovs = JtW @ Js.flatten(-3, -2)  # (..., D, D)
        return icovs
