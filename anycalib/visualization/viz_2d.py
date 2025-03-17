import operator
from collections.abc import Callable
from math import pi

import matplotlib.patheffects as pe
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike
from torch import Tensor

from anycalib.manifolds import Unit3

RAD2DEG = 180 / pi


def do_detach_cpu(*args):
    return [arg.detach().cpu() if isinstance(arg, Tensor) else arg for arg in args]


def polar_angles_from_rays(rays: Tensor, as_deg: bool = True) -> Tensor:
    """Compute the polar angles of a set of rays.

    Args:
        rays: (..., N, 3) rays.

    Returns:
        (..., N) polar angles.
    """
    r = torch.linalg.norm(rays[..., :2], dim=-1)
    polar_angles = torch.atan2(r, rays[..., 2])
    if as_deg:
        polar_angles *= RAD2DEG
    return polar_angles


def add_colorbar_to_axes(
    ax: Axes,
    sm: ScalarMappable,
    bbox: tuple[float, float, float, float] | None = None,
    label: str | None = None,
    extreme_ticks_only: bool = True,
    **kwargs,
):
    """Add a colorbar to axis.

    Args:
        ax: The axis to which the colorbar should be added.
        sm: The ScalarMappable object to be used for the colorbar.
        bbox: The bounding box of the colorbar in the axis coordinates,
            [x0, y0, width, height], with (x0, y0) specifying the lower-left corner of
            inset Axes where the colorbar is placed, and its width and height. If None,
            defaults to (1.03, 0, 0.04, 1.0).
        label: The label of the colorbar.
        extreme_ticks: If True, the colorbar will have ticks at the extreme values of
            the ScalarMappable `sm`.

    Returns:
        The colorbar object.
    """
    fig = ax.get_figure()
    assert fig is not None
    if bbox is None:
        bbox = (1.03, 0, 0.04, 1.0)
    cax = ax.inset_axes(bbox)
    ticks = sm.get_clim() if extreme_ticks_only else None
    cbar = fig.colorbar(sm, cax=cax, label=label, ticks=ticks, **kwargs)
    return cbar


def get_scalar_mappable(
    vmin: float, vmax: float, cmap: str | Colormap
) -> ScalarMappable:
    """Create a ScalarMappable object for a given colormap and range."""
    return ScalarMappable(Normalize(vmin, vmax), cmap)


def get_error_cmap(
    colors: list[str] | None = None, n: int = 256, gamma: float = 1.0
) -> Colormap:
    """Create a colormap for visualization of errors."""
    colors = ["lime", "yellow", "orange", "red"] if colors is None else colors
    return LinearSegmentedColormap.from_list("error_cmap", colors, N=n, gamma=gamma)


def plot_image(
    ax: Axes,
    im: ArrayLike,
    cmap: str = "gray",
    axis_off: bool = True,
    text: str | None = None,
):
    """Plot an image on a given axis."""
    im = im.detach().cpu() if isinstance(im, Tensor) else im
    ax.imshow(im, cmap=cmap)
    if axis_off:
        ax.axis("off")
    if text is not None:
        plot_text(ax, text)


def plot_text(
    ax: Axes,
    text: str,
    pos: tuple[float, float] = (0.01, 0.99),
    ha: str = "left",
    va: str = "top",
    fs: float = 15.0,
    color: str = "w",
    lcolor: str = "k",
    lw: float = 4.0,
    zorder: int = 5,
    **kwargs,
):
    """Plot text on a given axis."""
    pe_ = [pe.withStroke(linewidth=lw, foreground=lcolor)] if lw > 0 else None
    ax.text(
        *pos,
        text,
        ha=ha,
        va=va,
        fontsize=fs,
        color=color,
        zorder=zorder,
        transform=ax.transAxes,
        path_effects=pe_,
        **kwargs,
    )


def plot_heatmap(
    ax: Axes,
    heatmap: np.ndarray | Tensor,
    cmap: str | Colormap = "Spectral_r",
    alpha: float | ArrayLike = 0.2,
    vmin: float | None = None,
    vmax: float | None = None,
    add_colorbar: bool = False,
    contours_every: int = -1,
    text: str | None = None,
    **kwargs,
):
    """Plot heatmaps with optional contours.

    Args:
        ax: Axis to plot the heatmap.
        heatmap: 2D heatmap.
        cmap: Colormap of the Heatmap.
        alpha: Opacity value in [0, 1].
        vmin: Mininmum value to clamp the heatmap.
        vmax: Maximum value to clamp the heatmap.
        contours_every: Interval between contours. If < 1, no contours are plotted.
    """
    cmap = get_error_cmap() if cmap == "error" else cmap
    hm = ax.imshow(
        heatmap,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
    )

    if add_colorbar:
        add_colorbar_to_axes(ax, hm)

    if contours_every > 0:
        plot_contours(
            ax,
            heatmap,
            contours_every=contours_every,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            **kwargs,
        )

    if text is not None:
        plot_text(ax, text)


def plot_contours(
    ax: Axes,
    heatmap: np.ndarray | Tensor,
    contours_every: int = 5,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | Colormap = "Spectral_r",
    contours_colors: ArrayLike | None = None,
    lw: float = 2.0,
    label_units: str = "°",
    label_decimals: int = 0,
    label_fs: float = 16,
    label_lw: float = 1.5,
    label_color: str = "w",
    text: str | None = None,
    **kwargs,
):
    """Plot contours of a heatmap. Adapted and inspired from GeoCalib:
    https://github.com/cvg/GeoCalib/blob/main/geocalib/viz2d.py

    Args:
        ax: Axis to plot the heatmap.
        heatmap: 2D heatmap.
        contours_every: Interval between contours.
        vmin: Mininmum value to clamp the heatmap.
        vmax: Maximum value to clamp the heatmap.
        cmap: Colormap of the Heatmap.
        contours_colors: Color of the contour lines. If not given, it will be infered
            from the colormap and heatmap.
        lw: Width of the contour lines.
        label_units: Units of the heatmap values. Will be used to label the contours.
        label_decimals: Number of decimals to use for the contour labels.
    """
    if contours_every <= 0:
        return
    cmap = get_error_cmap() if cmap == "error" else cmap
    vmin = float(heatmap.min()) if vmin is None else vmin
    vmax = float(heatmap.max()) if vmax is None else vmax
    cs = ax.contour(
        heatmap,
        levels=np.arange(vmin, vmax + contours_every, contours_every),
        colors=contours_colors,
        linewidths=lw,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap if contours_colors is None else None,
    )

    if label_fs > 0:
        clabels_ = ax.clabel(
            cs,
            fmt=f"%.{label_decimals}f{label_units}",
            fontsize=label_fs,
            colors=label_color,
        )
        if label_lw > 0:
            for cl in clabels_:
                cl.set_path_effects([pe.withStroke(linewidth=label_lw, foreground="k")])

    if text is not None:
        plot_text(ax, text)


def plot_2d_arrows(
    ax: Axes,
    directions: np.ndarray | Tensor,
    sample_factor: float = 0.03,
    color_vectors: ColorType | None = "lime",
    color_field: np.ndarray | Tensor | None = None,
    alpha: float | Tensor | np.ndarray = 1,
    cmap: str | Colormap = "Spectral_r",
    add_colorbar: bool = False,
    text: str | None = None,
    **kwargs,
):
    """Plot 2D vector field.

    Args:
        ax: Axis to plot the vector field.
        directions: (H, W, 2) 2D vectors.
        sample_factor: Factor to subsample the vectors. if 1, no subsampling is done.
        color_vectors: Color of the vectors. Ignored if color_field is not None.
        color_field: (H, W) field to color the vectors.
    """
    if not 0 < sample_factor <= 1:
        raise ValueError("sample_factor must be in (0, 1].")
    directions, color_field, alpha = do_detach_cpu(directions, color_field, alpha)

    h, w = directions.shape[:2]
    x, y = np.arange(w), np.arange(h)

    # subsample
    if sample_factor < 1:
        step = int(1 / sample_factor)
        start = step // 2
        x, y = x[start::step], y[start::step]
        directions = directions[start::step, start::step]
        if color_field is not None:
            color_field = color_field[start::step, start::step]
        if isinstance(alpha, (Tensor, np.ndarray)):
            alpha = alpha[start::step, start::step]

    args = (x, y, directions[..., 0], directions[..., 1])
    if color_field is not None:
        args = args + (color_field,)

    sm = ax.quiver(
        *args,
        color=color_vectors,
        cmap=get_error_cmap() if cmap == "error" else cmap,
        alpha=alpha,
        **kwargs,
    )

    if add_colorbar and color_field is not None:
        add_colorbar_to_axes(ax, sm)

    if text is not None:
        plot_text(ax, text)


def plot_radial_vectors(
    ax: Axes,
    bearings: Tensor,
    sample_factor: float = 0.03,
    color_vectors: ColorType = "lime",
    color_field: np.ndarray | Tensor | None = None,
    cmap: str | Colormap = "Spectral_r",
    **kwargs,
):
    """Plot radial vectors.

    Args:
        ax: Axis to plot the radial vectors.
        bearings: (H, W, 3) rays.
        sample_factor: Factor to subsample the vectors. if 1, no subsampling is done.
        color_vectors: Color of the vectors. Ignored if color_field is not None.
        color_field: (H, W) field to color the vectors.
    """
    directions = bearings[..., :2].detach().cpu()
    # check if the y-axis is inverted (e.g. by imshow)
    if ax.yaxis_inverted():
        directions = directions.clone()
        directions[..., 1] = -directions[..., 1]

    if isinstance(color_field, Tensor):
        color_field = color_field.detach().cpu()

    plot_2d_arrows(
        ax,
        directions,
        sample_factor=sample_factor,
        color_vectors=color_vectors,
        color_field=color_field,
        cmap=cmap,
        **kwargs,
    )


def plot_tangent_errors_as_vectors(
    ax: Axes,
    bearings: Tensor,
    bearings_gt: Tensor,
    covs: Tensor | None = None,
    sample_factor: float = 0.03,
    color_vectors: ColorType | None = None,
    cmap: str | Colormap = "error",
    vmin: float | None = 0,
    vmax: float | None = None,
    vmin_log_cov: float = 0,
    **kwargs,
):
    # use tangent space of the ground-truth bearings to compute the errors
    errors = Unit3.logmap(bearings_gt, bearings.detach()).cpu()
    color_field = (
        RAD2DEG * torch.linalg.norm(errors, dim=-1) if color_vectors is None else None
    )
    # check if the y-axis is inverted (e.g. by imshow)
    if ax.yaxis_inverted():
        errors[..., 1] = -errors[..., 1]

    if covs is not None:
        covs = covs.detach().cpu()
        # define opacity based on the (normalized) predicted covariance of each error
        weights = covs_to_confidences(covs, vmin_log=vmin_log_cov, as_log=True)
        alpha = weights.mean(-1)
    else:
        alpha = 1.0

    clim = (None if vmin is None else vmin, None if vmax is None else vmax)
    plot_2d_arrows(
        ax,
        errors,
        sample_factor=sample_factor,
        color_vectors=color_vectors,
        color_field=color_field,
        alpha=alpha,
        cmap=cmap,
        clim=clim,
        **kwargs,
    )


def plot_polar_angles(
    ax: Axes,
    bearings: Tensor,
    contours_every: int = 5,
    vmin: float | None = 0.0,
    vmax: float | None = 90.0,
    cmap: str | Colormap = "Spectral_r",
    only_contours: bool = True,
    **kwargs,
):
    """Plot polar angles as a heatmap.

    Args:
        ax: Axis to plot the heatmap.
        bearings: (H, W, 3) rays.
        contours_every: Interval between contours. If < 1, no contours are plotted.
        vmin: Mininmum value to clamp the heatmap.
        vmax: Maximum value to clamp the heatmap.
        cmap: Colormap of the Heatmap.
        only_contours: If True, only the contours are plotted.
    """
    if only_contours:
        plot_contours(
            ax,
            polar_angles_from_rays(bearings.detach(), as_deg=True).cpu(),
            contours_every=contours_every,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label_units="°",
            label_decimals=0,
            **kwargs,
        )
    else:
        plot_heatmap(
            ax,
            polar_angles_from_rays(bearings.detach(), as_deg=True).cpu(),
            contours_every=contours_every,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            label_units="°",
            label_decimals=0,
            **kwargs,
        )


def plot_angular_errors(
    ax: Axes,
    bearings: Tensor,
    bearings_gt: Tensor,
    cmap: str | Colormap = "error",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    covs: Tensor | None = None,
    **kwargs,
):
    """Plot geodesic distance errors as heatmaps.

    Args:
        ax: Axis to plot the heatmap.
        bearings: (H, W, 3) rays.
        bearings_gt: (H, W, 3) ground-truth rays.
        cmap: Colormap of the Heatmap.
        vmin: Mininmum value to clamp the error heatmap.
        vmax: Maximum value to clamp the error heatmap.
        covs: (..., N) covariances of the geodesic distances.
    """
    # geodesic (angular) distance between each predicted bearing
    errors = RAD2DEG * Unit3.distance(bearings.detach(), bearings_gt).cpu()  # (..., N)

    if covs is not None:
        covs = covs.detach().cpu()
        # define opacity based on the (normalized) predicted covariance of each error
        weights = covs_to_confidences(
            covs,
            vmin=0.0 if vmin is None else vmin,
            **kwargs,
        )
        alpha = weights.mean(-1)
    else:
        alpha = 0.5

    plot_heatmap(
        ax,
        errors,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        contours_every=-1,
        **kwargs,
    )


def plot_tangent_errors_as_heatmaps(
    ax_x: Axes,
    ax_y: Axes,
    bearings: Tensor,
    bearings_gt: Tensor,
    cmap: str | Colormap = "error",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    covs: Tensor | None = None,
    **kwargs,
):
    """Plot (absolute) errors defined in the tangent plane of each bearing as heatmaps.

    One heatmap for the x component of the errors and another for the y component.

    Args:
        ax_x: axis for plotting the x component of the errors.
        ax_y: axis for plotting the y component of the errors.
        bearings: (H, W, 3) rays.
        bearings_gt: (H, W, 3) ground-truth rays.
        cmap: Colormap of the Heatmap.
        vmin: In degrees. Mininmum value to clamp the error heatmap.
        vmax: In degrees. Maximum value to clamp the error heatmap.
        covs: (..., N, 2) covariances inthe tangent plane of each bearing.
    """
    # errors in the tangent plane of each ground-truth bearing (H, W, 2)
    errors = RAD2DEG * Unit3.logmap(bearings_gt, bearings.detach()).abs().cpu()

    if covs is not None:
        covs = covs.detach().cpu()
        # define opacity based on the (normalized) predicted covariance of each error
        weights = covs_to_confidences(
            covs,
            vmin=0.0 if vmin is None else vmin,
            **kwargs,
        )
        alpha_x = weights[..., 0]
        alpha_y = weights[..., 1]
    else:
        alpha_x = alpha_y = 0.5

    plot_heatmap(
        ax_x,
        errors[..., 0],
        cmap=cmap,
        alpha=alpha_x,
        vmin=vmin,
        vmax=vmax,
        contours_every=-1,
        **kwargs,
    )

    plot_heatmap(
        ax_y,
        errors[..., 1],
        cmap=cmap,
        alpha=alpha_y,
        vmin=vmin,
        vmax=vmax,
        contours_every=-1,
        **kwargs,
    )


def covs_to_confidences(
    covs: Tensor,
    as_log: bool = True,
    vmin: float | None = 0,
    vmin_log: float | None = 0,
    **kwargs,
):
    """Convert covariances to confidences in [0, 1] for visualization.

    The intuition for selecting the minimum value in the log case (the default) is to
    use vmin_log as the upper-bound angular uncertainty (in degrees) that ideally and
    approximately corresponds to correct predictions. For example, when vmin_log is set
    to 0, uncertainties below 1 degree (1e0) will have full opacity (1.0), while each
    subsequent order of magnitude increase in uncertainty will reduce the opacity
    linearly until the maximum value is reached.
    """
    eps = torch.finfo(covs.dtype).eps
    stds = covs.clamp(eps).sqrt() * RAD2DEG
    if as_log:
        vmin_log = 0 if vmin_log is None else vmin_log
        weights = torch.log10(stds).clamp(vmin_log)
        weights = 1 - (weights - vmin_log) / (weights.max() - vmin_log)
    else:
        vmin = 0 if vmin is None else vmin
        weights = stds.clamp(vmin)
        weights = 1 - (weights - vmin) / (weights.max() - vmin)
    return weights


def plot_uncertainties_as_heatmap(
    ax: Axes | tuple[Axes, Axes],
    covs: Tensor,
    aggregator: str | Callable | None = "mean",
    cmap: str | Colormap = "turbo_r",
    vmin_log: float = 0,
    alpha: float = 0.5,
    log_normalize: bool = True,
    **kwargs,
):
    """Plot uncertainties as heatmaps.

    Args:
        ax: Axis or tuple of axes to plot the uncertainties.
        covs: (H, W, 2) covariances in the tangent plane of each bearing.
        aggregator: Aggregator function to use when plotting the uncertainties. Default
            is "mean".
        cmap: Colormap of the Heatmap.
        vmin_log: Minimum value to clamp the log uncertainties.
        alpha: Opacity value in [0, 1].
    """
    if aggregator is None and isinstance(ax, Axes):
        raise ValueError("aggregator must be given when a single axis is provided.")

    eps = torch.finfo(covs.dtype).eps
    # we visualize the confidences as the inverse of the standard deviations
    std = RAD2DEG * covs.detach().clamp(eps).sqrt()
    if log_normalize:
        std = torch.log10(std).clamp(vmin_log)
        # std = (std - std.min()) / (std.max() - std.min())  # bound values
    std = std.cpu()

    if isinstance(ax, Axes):
        # when just one axis is given, plot the aggregation of the uncertainties
        if isinstance(aggregator, str):
            std = operator.attrgetter(aggregator)(torch)(std, dim=-1)
        elif callable(aggregator):
            std = aggregator(std)
        else:
            raise ValueError("aggregator must be a function or a string.")
        plot_heatmap(ax, std, cmap=cmap, alpha=alpha, **kwargs)
        return

    # when two axes are given, plot the x and y components of the uncertainties
    plot_heatmap(ax[0], std[..., 0], cmap=cmap, alpha=alpha, **kwargs)
    plot_heatmap(ax[1], std[..., 1], cmap=cmap, alpha=alpha, **kwargs)


def plot_uncertainties_as_ellipses():
    raise NotImplementedError
