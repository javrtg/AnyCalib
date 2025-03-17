from math import pi, sqrt

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import anycalib.visualization.viz_2d as viz_2d
from anycalib.cameras.base import BaseCamera
from anycalib.manifolds import Unit3

RAD2DEG = 180 / pi


def create_figures(im_size, nrows, ncols, min_fig_size: float = 4.0):
    h, w = im_size
    plot_w = min_fig_size if w < h else min_fig_size * w / h
    plot_h = min_fig_size if h < w else min_fig_size * h / w
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(plot_w * ncols, plot_h * nrows),
        squeeze=False,
    )
    return fig, axs


def make_radial_figure(
    pred: dict,
    data: dict,
    n_pairs: int = 2,
    min_fig_size: float = 4.0,
    show_radial_vecs: bool = False,
) -> dict:
    """Image grid of radial vectors plots."""
    titles = ("Image", "Polar angles GT", "Pred")
    if show_radial_vecs:
        titles = titles + ("Radial vecs GT", "Pred")
    if "tangent_coords" in pred:
        titles = titles + ("Tangent θx GT", "Pred", "Tangent θy GT", "Pred")

    # image grid
    n_pairs = min(n_pairs, len(data["image"]))
    b, _, h, w = data["image"].shape
    fig, axs = create_figures((h, w), n_pairs, len(titles), min_fig_size=min_fig_size)

    images = data["image"].permute(0, 2, 3, 1).clamp(0, 1)
    rays_gt = data["rays"].view(b, h, w, 3)
    rays = pred["rays"].view(b, h, w, 3)
    if "tangent_coords" in pred:
        tcoords_gt = (RAD2DEG * Unit3.logmap_at_z1(rays_gt)).cpu()
        tcoords = (RAD2DEG * pred["tangent_coords"].detach().view(b, h, w, 2)).cpu()
        tc_kwargs = {
            "contours_every": 10,
            "cmap": "Spectral_r",
            "vmin": -90,
            "vmax": 90,
        }

    for i in range(n_pairs):
        axr = axs[i]
        for j, title in enumerate(titles):
            viz_2d.plot_image(axr[j], images[i])
            if i == 0:
                axr[j].set_title(title)
        # polar angles
        viz_2d.plot_polar_angles(axr[1], rays_gt[i])
        viz_2d.plot_polar_angles(axr[2], rays[i])
        j = 3
        if show_radial_vecs:
            # quiver plots with the same scale as the ground-truth
            viz_2d.plot_radial_vectors(axr[j], rays_gt[i], color_vectors="lime")
            fig.canvas.draw()  # ensure the scale attribute of the quiver plot is present
            viz_2d.plot_radial_vectors(
                axr[j + 1],
                rays[i],
                color_vectors="orange",
                scale=axr[1].collections[-1].scale,
            )
            j = 5
        if "tangent_coords" in pred:
            tc_gt = tcoords_gt[i]
            tc = tcoords[i]
            viz_2d.plot_contours(axr[j], tc_gt[..., 0], **tc_kwargs)
            viz_2d.plot_contours(axr[j + 1], tc[..., 0], **tc_kwargs)
            viz_2d.plot_contours(axr[j + 2], tc_gt[..., 1], **tc_kwargs)
            viz_2d.plot_contours(axr[j + 3], tc[..., 1], **tc_kwargs)
    fig.tight_layout()
    return {"radial": fig}


def make_errors_figure(
    pred: dict,
    data: dict,
    n_pairs: int = 2,
    min_fig_size: float = 4.0,
    log_normalize: bool = False,
    agg_fn: str | None = None,  # "prod",
) -> dict:
    """Image grid of error plots."""
    titles = ("Image", "Tangent Error", "Angular Error")
    suptitle = "Errors"
    if "log_covs" in pred:
        suptitle = "Errors and Uncertainties"
        unc_str = "log-uncertainty" if log_normalize else "uncertainty"
        titles += (f"{unc_str}-x", f"{unc_str}-y") if agg_fn is None else (unc_str,)
    if "weights" in pred:
        titles += ("alpha",)

    # image grid
    n_pairs = min(n_pairs, len(data["image"]))
    b, _, h, w = data["image"].shape
    fig, axs = create_figures((h, w), n_pairs, len(titles), min_fig_size=min_fig_size)
    fig.suptitle(suptitle)

    images = data["image"].permute(0, 2, 3, 1).clamp(0, 1)
    rays_gt = data["rays"].view(b, h, w, 3)
    rays = pred["rays"].view(b, h, w, 3)
    log_covs = pred.get("log_covs", None)
    if log_covs is not None:
        covs = log_covs.exp().view(b, h, w, 2)
    if "weights" in pred:
        # compute "alpha(s)" of mixture model
        alpha = torch.softmax(pred["weights"], dim=-1)[..., 0]  # (b, 1 or h*w)
        alpha = (
            alpha.expand(-1, h * w).view(b, h, w).cpu()
            if alpha.shape[-1] == 1
            else alpha.view(b, h, w).cpu()
        )

    for i in range(n_pairs):
        axr = axs[i]
        for j, title in enumerate(titles):
            viz_2d.plot_image(axr[j], images[i])
            if i == 0:
                axr[j].set_title(title)
        # errors
        viz_2d.plot_tangent_errors_as_vectors(axr[1], rays[i], rays_gt[i])
        viz_2d.plot_angular_errors(axr[2], rays[i], rays_gt[i], add_colorbar=True)
        # uncertainties
        if log_covs is not None:
            axr_u = axr[3] if isinstance(agg_fn, str) else (axr[3], axr[4])
            viz_2d.plot_uncertainties_as_heatmap(
                axr_u,
                covs[i],
                log_normalize=log_normalize,
                aggregator=agg_fn,
                add_colorbar=True,
            )
        if "weights" in pred:
            viz_2d.plot_heatmap(axr[-1], alpha[i], add_colorbar=True)

    fig.tight_layout()
    return {"errors": fig}


def make_editmaps_figure(
    pred: dict, data: dict, n_pairs: int = 2, min_fig_size: float = 4.0
):
    """Image grid of editmap plots."""
    titles = ("Image", "Pix AR Error", "Uncertainty", "Radii GT [pix]", "Pred")
    n_pairs = min(n_pairs, len(data["image"]))
    _, _, h, w = data["image"].shape
    rmax = 0.5 * sqrt(h**2 + w**2)
    fig, axs = create_figures((h, w), n_pairs, len(titles), min_fig_size=min_fig_size)

    images = data["image"].permute(0, 2, 3, 1).clamp(0, 1)
    _, _, hp, wp = pred["pix_ar_map"].shape
    assert h / hp == w / wp and h / hp >= 1

    pix_ar_gt = data["pix_ar"]  # (b,)
    radii = (h / hp) * pred["radii"].detach()  # (b, hp, wp)
    # upsample to image resolution
    radii = F.interpolate(radii[:, None], (h, w), mode="bilinear", align_corners=False)[:, 0].cpu()  # fmt:skip
    pix_ar = F.interpolate(
        pred["pix_ar_map"].detach(), (h, w), mode="bilinear", align_corners=False
    )  # (b, 2, h, w)

    pix_ar_err = (pix_ar[:, 0] - pix_ar_gt[:, None, None]).abs().cpu()  # (b, h, w)
    pix_ar_unc = torch.exp(pix_ar[:, 1]).cpu()
    radii_gt = torch.linalg.norm(
        BaseCamera.pixel_grid_coords(h, w, data["cxcy_gt"], 0.5)
        - data["cxcy_gt"][:, None, None],
        dim=-1,
    ).cpu()

    for i in range(n_pairs):
        axr = axs[i]
        for j, title in enumerate(titles):
            viz_2d.plot_image(axr[j], images[i])
            if i == 0:
                axr[j].set_title(title)
        viz_2d.plot_heatmap(axr[1], pix_ar_err[i], alpha=0.5, add_colorbar=True, cmap="error")  # fmt: skip
        viz_2d.plot_text(axr[1], f"GT: {pix_ar_gt[i].item():.1f}")
        viz_2d.plot_heatmap(axr[2], pix_ar_unc[i], alpha=0.5, add_colorbar=True, cmap="turbo_r")  # fmt: skip
        viz_2d.plot_contours(axr[3], radii_gt[i], vmin=0, vmax=rmax, contours_every=25, label_units="")  # fmt: skip
        viz_2d.plot_contours(axr[4], radii[i], vmin=0, vmax=rmax, contours_every=25, label_units="")  # fmt: skip

    fig.tight_layout()
    return {"editmaps": fig}


def make_batch_figures(
    pred: dict, data: dict, n_pairs: int = 3, min_fig_size: float = 3.0
) -> dict:
    """Create figures for debugging"""
    figs = make_radial_figure(pred, data, n_pairs, min_fig_size=min_fig_size)
    figs |= make_errors_figure(pred, data, n_pairs, min_fig_size=min_fig_size)
    if "pix_ar_map" in pred and "radii" in pred:
        figs |= make_editmaps_figure(pred, data, n_pairs, min_fig_size=min_fig_size)
    return figs
