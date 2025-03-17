import torch
from torch import Tensor

from anycalib import utils as ut
from anycalib.cameras.pinhole import Pinhole, check_within_fov, propagate_tangent_covs


class SimplePinhole(Pinhole):
    """Simple pinhole camera model.

    The (ordered) intrinsic parameters are f, cx, cy, where:
        - f [pixels] is the the focal length (fx = fy),
        - (cx, cy) [pixels] is the principal point.

    Args:
        max_fov: Threshold in degrees for masking out bearings/rays whose incidence
            angles correspond to fovs above this admissible field of view.
    """

    NAME = "simple_pinhole"
    NUM_F = 1
    PARAMS_IDX = {"f": 0, "cx": 1, "cy": 2}
    num_k = 0  # no distortion

    def fit_minimal(
        self, im_coords: Tensor, bearings: Tensor, cxcy: Tensor | None = None
    ) -> Tensor:
        """Fit instrinsics with a minimal set of 2D-bearing correspondences.

        Since the focal length is overconstrained by 2 observations, we compute it by
        minimizing the squared errors in the image plane.
        When cxcy is given, the equations are simplified further but care should be
        taken with its input shape: In RANSAC, tipically, im_coords and bearings will
        have a shape of (..., N_samples, 1, 2) and (..., N_samples, 1, 3), respectively.
        Thus, cxcy should have a shape of
            a) (..., 1, 2) to allow broadcasting across all samples, or
            b) should be expanded: (..., N_samples, 2) for the same purpose.

        Args:
            im_coords: (..., MIN_SIZE, 2) image coordinates.
            bearings: (..., MIN_SIZE, 3) unit bearing vectors in the camera frame.
            cxcy: (..., 2) known principal points.

        Returns:
            (..., 3) fitted intrinsic parameters: f, cx, cy.
        """
        eps = torch.finfo(bearings.dtype).eps
        assert (
            self.get_min_sample_size(cxcy is not None)
            == im_coords.shape[-2]
            == bearings.shape[-2]
        ), "Input shapes do not match the minimal sample size."

        if cxcy is None:
            # estimate focal via the Schur complement of the corresponding linear system
            proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)  # (..., 2, 2)
            bsum = proj.sum(dim=-2)  # (..., 2)
            bsum_sq = (bsum * bsum).sum(dim=-1)  # (...)
            bsum_imsum = (bsum * im_coords.sum(dim=-2)).sum(dim=-1)  # (...)
            b_im_sum = (proj * im_coords).sum(dim=(-2, -1))  # (...)
            bsq_sum = (proj * proj).sum(dim=(-2, -1))  # (...)
            den = (bsq_sum - 0.5 * bsum_sq).abs().clamp(eps)
            f = ((b_im_sum - 0.5 * bsum_imsum) / den)[..., None].abs()
            # average principal point
            c = (im_coords - f[..., None] * proj).mean(dim=-2)
            return torch.cat((f, c), dim=-1)

        iproj = bearings[..., 2:] / bearings[..., :2].abs().clamp(eps)  # (Z/XY)
        f = torch.mean(((im_coords - cxcy[..., None, :]) * iproj).abs(), dim=-1)  # (..., 1) # fmt: skip
        return torch.cat((f, cxcy.expand(*f.shape[:-1], 2)), dim=-1)

    def fit(
        self,
        im_coords: Tensor,
        bearings: Tensor,
        cxcy: Tensor | None = None,
        covs: Tensor | None = None,
        *ignored_args,
        **ignored_kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Fit instrinsics with a set of 2D-bearing correspondences.

        Args:
            im_coords: (..., N, 2) image coordinates.
            bearings: (..., N, 3) unit bearing vectors.
            cxcy: (..., 2) known principal points.
            covs: (..., N, 2) diagonal elements of the covariances expressed in the
                tangent space of the input bearings.

        Returns:
            (..., 3) fitted intrinsic parameters: f, cx, cy.
            (...,) integer tensor indicating success. 0 if successful. Otherwise, an
                illegal value was found (<0) or the system is singular (>0).
        """
        eps = torch.finfo(bearings.dtype).eps
        # perspective projection
        proj = bearings[..., :2] / bearings[..., 2:].clamp(eps)  # (..., N, 2)
        valid = check_within_fov(bearings, self.max_fov)  # (..., N)
        bs = proj  # (..., N, 2) observations

        # propagate covariances if present (..., N, 2, 2)
        if covs is not None:
            e_covs = propagate_tangent_covs(bearings, proj, covs.clamp(eps))
            Ws, info = torch.linalg.inv_ex(e_covs)
            valid = valid & (info == 0)
        else:
            Ws = None

        if cxcy is None:
            # form linear system
            As = proj.new_zeros((*proj.shape, 3))
            # normalize first two columns for improved conditioning number
            norm_factor = im_coords.amax((-2, -1), keepdim=True)  # (..., 1, 1)
            As[..., 0] = im_coords / norm_factor
            As[..., 0, 1] = As[..., 1, 2] = -1
            intrinsics, info = ut.solve_2dweighted_lstsq(As, bs, Ws, valid)
            f = norm_factor.squeeze(-2) * intrinsics[..., :1].reciprocal()
            c = intrinsics[..., 1:] * f
            return torch.cat((f, c), dim=-1), info

        As = (im_coords - cxcy[..., None, :])[..., None]  # (..., N, 2, 1)
        inv_f, info = ut.solve_2dweighted_lstsq(As, bs, Ws, valid)  # (..., 1)
        return torch.cat((inv_f.reciprocal(), cxcy), dim=-1), info
