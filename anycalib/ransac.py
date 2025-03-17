from math import comb, pi

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3


class RANSAC:
    """Simple parallel RANSAC

    Args:
        cfg: Configuration dictionary with the following key-value pairs:
            - max_reproj_error (float): Maximum reprojection error in pixels to consider
                a correspondence an inlier. Used for camera models *not* specified in
                the class variable `ANG_ERROR_CAMS`.
            - max_ang_error (float): Maximum angular error in radians to consider a
                correspondence an inlier. Only used for camera models specified in
                the class variable `ANG_ERROR_CAMS`.
            - n_samples (int): Maximum number of models to try (or equivalently, number
                of random samples to draw from the inputs).
            - chunk_size (int): Number of models to process in parallel.
            - max_correspondences (int): Maximum number of correspondences to consider.
                If the number of correspondences is larger than this value, a random
                subset of this size will be used.
    """

    __slots__ = "cfg"

    DEFAULT_CONF = {
        "th_reproj_error": 5**2,  # pixels^2
        "th_ang_error": 1 * pi / 180,  # radians
        "n_samples": 2_048,
        "chunk_size": 1_024,
        "max_correspondences": 20_000,
    }

    # cam ids for which msac scores will correspond to angular errors. Recommended for
    # cameras whose unprojection is more efficient than their projection function.
    ANG_ERROR_CAMS = {
        "division:2",
        "division:3",
        "division:4",
        "simple_division:2",
        "simple_division:3",
        "simple_division:4",
    }

    def __init__(self, cfg: dict[str, float] | None = None):
        self.cfg = self.DEFAULT_CONF | (cfg or {})

    def __call__(
        self,
        cam_sac: BaseCamera,
        im_coords: Tensor,
        bearings: Tensor,
        probs: Tensor | None = None,
        cxcy: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Estimate camera intrinsics using RANSAC.

        Args:
            cam_sac: Camera model to use for fitting.
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) normalized bearings / unit vectors.
            probs: (..., N) optional probabilities/weights for sampling. Must be
                non-negative and may not sum to 1.
            cxcy: (..., 2) already known principal point.

        Returns:
            intrinsics: (..., D) estimated intrinsics.
            inliers: (..., N) boolean mask of inliers.
        """
        sample_dim = cam_sac.get_min_sample_size(cxcy is not None)

        if cam_sac.id in self.ANG_ERROR_CAMS:
            th = self.cfg["th_ang_error"]
            err_fun = self._ang_error
        else:
            th = self.cfg["th_reproj_error"]
            err_fun = self._sq_reproj_error

        n_samples = self.cfg["n_samples"]
        chunk_size = self.cfg["chunk_size"]
        max_c = self.cfg["max_correspondences"]

        if max_c is not None and max_c < bearings.shape[-2]:
            idx = torch.randperm(bearings.shape[-2], device=bearings.device)[:max_c]
            bearings = bearings[..., idx, :]
            im_coords = im_coords[..., idx, :]
            probs = None if probs is None else probs[..., idx]

        # limit the number of samples to the number of possible combinations
        n_samples = min(n_samples, comb(bearings.shape[-2], sample_dim))

        batch_size = im_coords.shape[:-2]
        # draw sample indices (..., n_samples, sample_dim)
        indices = self.sampler(bearings[..., 0], n_samples, sample_dim, probs).view(
            *batch_size, -1, 1
        )
        im_coords_ = im_coords.take_along_dim(indices, dim=-2).view(
            *batch_size, n_samples, sample_dim, 2
        )
        bearings_ = bearings.take_along_dim(indices, dim=-2).view(
            *batch_size, n_samples, sample_dim, 3
        )

        # intrinsics (model) for each sample
        cxcy = None if cxcy is None else cxcy.view(*batch_size, 1, 2)
        models = cam_sac.fit_minimal(im_coords_, bearings_, cxcy)  # (..., n_models, D)

        if chunk_size <= 0 or chunk_size >= n_samples:
            # NOTE: unsqueeze(-3) below assumes that cam.(un)project will do broadcasting.
            # If in a future cam model this is not allowed, this should be changed to:
            # .expand(*batch_size, n_samples, -1, 3)
            scores: Tensor = err_fun(  # type: ignore
                models, bearings.unsqueeze(-3), im_coords.unsqueeze(-3), cam_sac
            )
            # best model based on MSAC scores
            msac_scores = scores.clamp(max=th).sum(dim=-1)
            best_idx = msac_scores.argmin(dim=-1)[..., None, None]  # (..., 1, 1)
            best_model = models.take_along_dim(best_idx, dim=-2).squeeze(-2)
            inliers = scores.take_along_dim(best_idx, dim=-2).squeeze(-2) < th
            return best_model, inliers

        # Computing the error of all tentative models w.r.t. all N points leads to
        # O(n_models * N) memory complexity, which can lead to OOM or to a memory
        # bound function (more noticeable in CUDA) so, we split the models in chunks
        ch_s = n_samples if chunk_size <= 0 else min(n_samples, chunk_size)
        chunk_idx = torch.arange(0, n_samples + ch_s, ch_s, device=bearings.device)
        msac_scores = torch.cat(
            [
                # NOTE: unsqueeze(-3) below assumes that cam.(un)project will do broadcasting.
                # If in a future cam model this is not allowed, this should be changed to:
                # .expand(*batch_size, n_samples, -1, 3)
                err_fun(
                    models[..., i:j, :],
                    bearings.unsqueeze(-3),
                    im_coords.unsqueeze(-3),
                    cam_sac,
                )
                .clamp(max=th)  # type: ignore
                .sum(dim=-1)
                for i, j in zip(chunk_idx[:-1], chunk_idx[1:])
            ],
            dim=-1,
        )  # (..., n_models)

        best_idx = msac_scores.argmin(dim=-1)[..., None, None]  # (..., 1, 1)
        best_model = models.take_along_dim(best_idx, dim=-2).squeeze(-2)  # (..., D)
        scores, valid = err_fun(best_model, bearings, im_coords, cam_sac, True)
        inliers = scores < th
        return best_model, inliers if valid is None else inliers & valid

    @staticmethod
    def sampler(
        ref_tensor: Tensor, n_samples: int, sample_dim: int, probs: Tensor | None = None
    ) -> Tensor:
        """Random sampling of indices.

        Args:
            ref_tensor: (..., max_index) tensor from which to infer, maximum index to
                sample from (max_index), batch size and device.
            n_samples: Number of samples.
            sample_dim: Dimension of each sample.
            probs: (..., max_index) optional probabilities or weights for sampling a
                correspondence.

        Returns:
            samples: (..., n_samples, sample_dim) random samples.
        """
        torch.manual_seed(0)
        # limit the number of samples to the number of possible combinations
        max_index = ref_tensor.shape[-1]
        n_samples = min(n_samples, comb(max_index, sample_dim))

        # if probabilities/weights are given, sample from categorical distribution
        mi, ns, sd = max_index, n_samples, sample_dim
        if probs is None:
            device = ref_tensor.device
            samples = torch.randint(mi, (ns, sd), device=device)
        else:
            # in contrast to torch.multinomial, Categorical admits batched probabilities
            # samples = torch.multinomial(prob, n, replacement=True).view(ns, sd)
            m = Categorical(probs=probs.detach())
            samples = m.sample((n_samples, sample_dim)).moveaxis((0, 1), (-2, -1))  # type: ignore
            # (..., ns, sd)

        # ensure that in the same sample, there are no repeated indices as this would
        # lead to a singular system of equations
        for i in range(1, sample_dim):
            mask = (samples[..., :i] == samples[..., i, None]).any(dim=-1)
            while mask.any():
                # to ensure the while loop terminates, sample from uniform distribution.
                # This only affects the (potentially) small number of samples that have
                # repeated indices
                samples[..., i][mask] = torch.randint_like(
                    samples[..., i][mask], 0, max_index
                )
                mask = (samples[..., :i] == samples[..., i, None]).any(dim=-1)

        # batchify if needed
        samples = samples.expand(ref_tensor.shape[:-1] + (n_samples, sample_dim))
        return samples

    @staticmethod
    def _sq_reproj_error(
        models: Tensor,
        bearings: Tensor,
        im_coords: Tensor,
        cam_sac: BaseCamera,
        return_valid: bool = False,
    ) -> tuple[Tensor, Tensor | None] | Tensor:
        """Squared norm of reprojection errors (..., n_models, N).

        Args:
            models: (..., D) or (..., n_models, D) camera models.
            bearings: (..., N, 3) or (..., n_models, N, 3) unit bearing vectors.
            im_coords: (..., N, 2) or (..., n_models, N, 2) image coordinates.
            cam_sac: Camera model to use for fitting.

        Returns:
            err_sq_norm: (...., N) or (..., n_models, N) reprojection errors.
        """
        proj, valid = cam_sac.project(models, bearings)
        err_sq_norm = ((im_coords - proj) ** 2).sum(dim=-1)
        if return_valid:
            return err_sq_norm, valid
        return err_sq_norm

    @staticmethod
    def _ang_error(
        models: Tensor,
        bearings: Tensor,
        im_coords: Tensor,
        cam_sac: BaseCamera,
        return_valid: bool = False,
    ) -> tuple[Tensor, Tensor | None] | Tensor:
        """Angular error of unprojected bearings.

        Args:
            models: (..., D) or (..., n_models, D) camera models.
            bearings: (..., N, 3) or (..., n_models, N, 3) unit bearing vectors.
            im_coords: (..., N, 2) or (..., n_models, N, 2) image coordinates.
            cam_sac: Camera model to use for fitting.

        Returns:
            ang_dist: (...., N) or (..., n_models, N) angular errors.
        """
        unproj, valid = cam_sac.unproject(models, im_coords)
        ang_dist = Unit3.distance(unproj, bearings)
        if return_valid:
            return ang_dist, valid
        return ang_dist

