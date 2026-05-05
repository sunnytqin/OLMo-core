import importlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator, RBFInterpolator
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Polygon


# Reversed gist_heat: bright (cream/yellow) at low loss, dark (red/black) at
# high loss. Truncate the lightest 15% so the basin reads as warm yellow
# instead of washed-out white.
_base = plt.cm.gist_heat_r
cmap = ListedColormap(_base(np.linspace(0.15, 1.0, 256)), name='isoloss_heat')


def load_dataset(module_name):
    """Load ALL_DATASETS from a results data module and flatten into arrays."""
    mod = importlib.import_module(module_name)
    flops, epochs, data_amount, val_loss = [], [], [], []
    for data in mod.ALL_DATASETS:
        for i in range(len(data['flops_multiplier'])):
            if not np.isnan(data['validation_loss'][i]):
                flops.append(data['flops_multiplier'][i])
                epochs.append(data['epochs'][i])
                data_amount.append(data['chinchilla_scale'][i])
                val_loss.append(data['validation_loss'][i])
    return (np.array(flops), np.array(epochs),
            np.array(data_amount), np.array(val_loss))


def overfit_phantoms(x_data, y_data, z_data,
                     max_phantom_per_row=3, log_step=0.3, min_curvature=0.5):
    """Project per-row overtraining trend to add phantom data points.

    For each unique y (data-scale) row with ≥3 points, fit a quadratic in
    log10(x) to the last three samples. If the 2nd-order coefficient exceeds
    ``min_curvature`` (i.e. the row clearly turns upward), append phantoms at
    log10(x_last) + k·log_step using the projected loss. This nudges the RBF
    surface so the iso-loss contours bend up in the over-trained corner
    instead of monotonically decreasing into the extrapolated region.
    """
    px, py, pz = [], [], []
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    z_data = np.asarray(z_data)
    for d in sorted(set(y_data.tolist())):
        m = y_data == d
        if m.sum() < 3:
            continue
        order = np.argsort(x_data[m])
        log_f = np.log10(x_data[m][order])
        z_s = z_data[m][order]
        # Fit on the last 4 points (or all if fewer) so a U-shape with the
        # minimum at the third-from-last sample still registers as concave-up.
        n_fit = min(4, len(log_f))
        coef = np.polyfit(log_f[-n_fit:], z_s[-n_fit:], deg=2)
        if coef[0] < min_curvature:
            continue
        for k in range(1, max_phantom_per_row + 1):
            new_lf = log_f[-1] + log_step * k
            new_z = np.polyval(coef, new_lf)
            if new_z > z_s[-1]:
                px.append(10**new_lf)
                py.append(d)
                pz.append(new_z)
    return np.asarray(px), np.asarray(py), np.asarray(pz)


def auto_contour_levels(z_data):
    """Pick contour levels with denser spacing near the loss minimum.

    Three bands (looser than before — fewer total contours):
      • fine   (step 0.05) within 0.5 of z_min — for the basin
      • medium (step 0.2)  for the next 1.5
      • coarse (step 0.5)  beyond that
    """
    z_min = np.floor(z_data.min() * 20) / 20
    z_max = np.ceil(z_data.max() * 10) / 10
    fine_end = min(z_min + 0.5, z_max)
    med_end = min(z_min + 2.0, z_max)
    fine = np.arange(z_min, fine_end + 1e-9, 0.05)
    med = np.arange(fine_end + 0.2, med_end + 1e-9, 0.2)
    coarse = np.arange(med_end + 0.5, z_max + 0.5, 0.5)
    return np.concatenate([fine, med, coarse])


def create_contour_plot(ax, x_data, y_data, z_data, x_label, x_ticks, x_ticklabels,
                        x_lim, diag_label, show_diagonal=True,
                        font_label=16, font_tick=15, diag_linewidth=2.5,
                        levels=None, mask_view=None, max_epochs=128,
                        fill_style='filled', interp='cubic'):
    """Create a contour plot with the given data and settings on the provided axis.

    Returns (scatter, grid_log_x_1d, grid_log_y_1d, grid_z) so callers can
    use the interpolated surface for gradient tracing.
    """

    ax.set_facecolor('white')

    # Create grid for interpolation (in log space for better results)
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)

    # For RBF extrapolation, expand the grid out to the plot limits so the
    # iso-loss surface fills the empty corners. For cubic (interpolation only),
    # a small margin around the data is enough.
    if interp == 'rbf':
        grid_log_x = np.linspace(np.log10(x_lim[0] * 0.8),
                                 np.log10(x_lim[1] * 1.15), 200)
        grid_log_y = np.linspace(np.log10(0.04), np.log10(20), 200)
    else:
        grid_log_x = np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 200)
        grid_log_y = np.linspace(log_y.min() - 0.1, log_y.max() + 0.1, 200)
    grid_x, grid_y = np.meshgrid(grid_log_x, grid_log_y)

    if interp == 'rbf':
        # Inject phantom overfit points for rows with clear upturn so the RBF
        # surface bends up at high FLOPs instead of monotonically descending.
        px, py, pz = overfit_phantoms(x_data, y_data, z_data)
        if len(px) > 0:
            fit_lx = np.concatenate([log_x, np.log10(px)])
            fit_ly = np.concatenate([log_y, np.log10(py)])
            fit_z = np.concatenate([z_data, pz])
        else:
            fit_lx, fit_ly, fit_z = log_x, log_y, z_data
        rbf = RBFInterpolator(
            np.column_stack([fit_lx, fit_ly]),
            fit_z,
            kernel='thin_plate_spline',
            smoothing=0.5,
        )
        grid_z = rbf(np.column_stack([grid_x.ravel(), grid_y.ravel()]))
        grid_z = grid_z.reshape(grid_x.shape)
    else:
        grid_z = griddata(
            (log_x, log_y),
            z_data,
            (grid_x, grid_y),
            method='cubic',
        )

    if levels is None:
        levels = auto_contour_levels(z_data)

    # Mask out infeasible regions (FLOPs view).
    #   • For the filled style: mask both the <1-epoch (upper-left) and
    #     >max-epochs (lower-right) corners.
    #   • For the lines-only style: only mask the upper-left (<1 epoch is
    #     physically impossible), and let the lower-right contours extrapolate
    #     so the over-trained upturn is visible.
    grid_z_plot = grid_z.copy()
    if mask_view == 'flops':
        if fill_style == 'filled':
            invalid = (grid_y > grid_x) | (grid_y < grid_x - np.log10(max_epochs))
        else:
            invalid = grid_y > grid_x
        grid_z_plot[invalid] = np.nan

    if fill_style == 'filled':
        contourf = ax.contourf(
            10**grid_x, 10**grid_y, grid_z_plot,
            levels=levels,
            cmap=cmap,
            alpha=0.7,
            extend='both'
        )
        contour_lw = 0.8
        clabel_fontsize = 9
    else:
        # Lines-only: thicker contours coloured by loss level for legibility
        contour_lw = 2.0
        clabel_fontsize = 11

    contour = ax.contour(
        10**grid_x, 10**grid_y, grid_z_plot,
        levels=levels,
        cmap=cmap if fill_style != 'filled' else None,
        colors='darkred' if fill_style == 'filled' else None,
        linewidths=contour_lw,
        alpha=0.9
    )

    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=clabel_fontsize, fmt='%.2f',
              colors='darkred' if fill_style == 'filled' else 'black',
              inline_spacing=5)

    # Plot actual data points with color based on loss value
    scatter = ax.scatter(
        x_data, y_data,
        c=z_data,
        cmap=cmap,
        s=80,
        edgecolors='black',
        linewidths=1.2,
        zorder=5,
        vmin=z_data.min(),
        vmax=z_data.max()
    )

    # Boxed loss-value annotation for each data point. Commented out for the
    # cleaner contour-only look — uncomment to put the per-point labels back.
    # for i in range(len(x_data)):
    #     ax.annotate(f'{z_data[i]:.2f}',
    #                 (x_data[i], y_data[i]),
    #                 xytext=(8, 8), textcoords='offset points',
    #                 fontsize=9, fontweight='bold',
    #                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
    #                          edgecolor='gray', alpha=0.8, linewidth=0.5),
    #                 zorder=6)

    # Highlight specific points (best and 1x 1-epoch baseline)
    best_idx = np.argmin(z_data)

    # Mark best point with a white star
    ax.scatter(x_data[best_idx], y_data[best_idx],
               marker='*', s=400, c='white', edgecolors='black',
               linewidths=2, zorder=10, label=r'Model-optimal $L_N$')

    # Mark 1x 1-epoch point with a black filled star
    # Find the point where y_data (chinchilla scale) is 1
    baseline_idx = np.where(y_data == 1)[0]
    if len(baseline_idx) > 0:
        # Find the one with minimum x_data (1 epoch = lowest flops for that scale)
        min_x_idx = baseline_idx[np.argmin(x_data[baseline_idx])]
        ax.scatter(x_data[min_x_idx], y_data[min_x_idx],
                   marker='*', s=400, c='black', edgecolors='black',
                   linewidths=2, zorder=10, label=r'Chinchilla-optimal $L_\mathrm{chin}$')

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels and title
    ax.set_xlabel(x_label, fontsize=font_label, fontweight='bold')
    ax.set_ylabel('Fresh Data D (TTP, Chinchilla x)', fontsize=font_label, fontweight='bold')

    # Customize tick labels for y-axis (TTP = chinchilla_scale * 20)
    y_ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['D=1N\n(0.05x)', 'D=2N\n(0.1x)', 'D=5N\n(0.25x)', 'D=10N\n(0.5x)',
                        'D=20N\n(1x)', 'D=40N\n(2x)', 'D=80N\n(4x)', 'D=160N\n(8x)',
                        'D=320N\n(16x)'],
                       fontsize=font_tick)

    # Customize tick labels for x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=font_tick)

    # Add a diagonal line (iso-compute line) similar to reference
    if show_diagonal:
        diag_x = np.array([x_lim[0], x_lim[1]])
        diag_y = diag_x  # slope of 1 in log-log space
        ax.plot(diag_x, diag_y, 'k--', linewidth=diag_linewidth, alpha=0.7, label=diag_label)

    # Grey out the upper-left infeasible region (above the y=x diagonal).
    if mask_view == 'flops':
        x_lo, x_hi = x_lim[0] * 0.8, x_lim[1] * 1.15
        y_hi = 20  # matches set_ylim below
        if y_hi <= x_hi:
            # Diagonal exits the plot via the top edge → triangular region
            pts = [(x_lo, x_lo), (x_lo, y_hi), (y_hi, y_hi)]
        else:
            # Diagonal exits via the right edge → quadrilateral region
            pts = [(x_lo, x_lo), (x_lo, y_hi), (x_hi, y_hi), (x_hi, x_hi)]
        ax.add_patch(Polygon(pts, closed=True, facecolor='lightgrey',
                             edgecolor='none', alpha=0.55, zorder=4))

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Set axis limits
    ax.set_xlim(x_lim[0] * 0.8, x_lim[1] * 1.15)
    ax.set_ylim(0.04, 20)

    return scatter, grid_log_x, grid_log_y, grid_z


def add_descent_paths(ax, grid_log_x, grid_log_y, grid_z, start_points_xy,
                      color='cyan', lw=2.0, alpha=0.95,
                      label='Empirical optimal (C,D) scaling path'):
    """Trace and overlay steepest-descent paths from start_points_xy.

    Parameters
    ----------
    start_points_xy : list of (x, y) in original (non-log) data space.
    """
    # Fill NaN regions with the max observed loss so the gradient creates a
    # natural repelling wall that keeps paths inside the valid interpolation zone.
    grid_z_filled = grid_z.copy()
    grid_z_filled[np.isnan(grid_z_filled)] = np.nanmax(grid_z_filled)

    # Smooth the surface before differentiating so the descent path doesn't
    # wiggle on local interpolation artifacts.
    grid_z_smooth = gaussian_filter(grid_z_filled, sigma=3.0)

    # Gradient components in log space: grid_z shape is (ny, nx)
    dz_dlogx = np.gradient(grid_z_smooth, grid_log_x, axis=1)
    dz_dlogy = np.gradient(grid_z_smooth, grid_log_y, axis=0)

    # Stack both gradient components into a single (ny, nx, 2) array so we
    # pay for only one interpolator lookup per ODE evaluation.
    grad_stack = np.stack([dz_dlogx, dz_dlogy], axis=-1)
    grad_interp = RegularGridInterpolator(
        (grid_log_y, grid_log_x), grad_stack,
        method='linear', bounds_error=False, fill_value=np.nan
    )

    def ode(t, state):
        lx, ly = state
        g = grad_interp([[ly, lx]])[0]   # shape (2,): [dz/dlogx, dz/dlogy]
        if np.any(np.isnan(g)):
            return [0.0, 0.0]
        norm = np.sqrt(g[0]**2 + g[1]**2) + 1e-10
        return [-g[0] / norm, -g[1] / norm]

    # Stop integration when the path hits the grid boundary
    def hit_boundary(t, state):
        lx, ly = state
        margin = 0.05
        return min(
            lx - (grid_log_x.min() + margin),
            grid_log_x.max() - margin - lx,
            ly - (grid_log_y.min() + margin),
            grid_log_y.max() - margin - ly,
        )
    hit_boundary.terminal = True
    hit_boundary.direction = -1

    for x0, y0 in start_points_xy:
        lx0 = np.log10(x0)
        ly0 = np.log10(y0)

        sol = solve_ivp(
            ode, [0, 20], [lx0, ly0],
            max_step=0.05, events=hit_boundary,
            rtol=1e-3, atol=1e-4
        )

        if not sol.success or sol.y.shape[1] < 4:
            continue

        lx_path, ly_path = sol.y
        x_path = 10**lx_path
        y_path = 10**ly_path

        # Draw the path dashed (only label the first segment)
        ax.plot(x_path, y_path, color=color, lw=lw, alpha=alpha,
                linestyle='--', zorder=8, label=label)
        label = '_nolegend_'  # suppress label for subsequent paths


# ── Predicted iso-loss surface from the multi-epoch Chinchilla fit ────────────
# Anchors come from the one-shot joint fit at k=15 (writeup_final.md §2):
#   L(N, D, D') = E + A/N^α + B/(D + η·D')^β
#   η = R*·(1 - exp(-x/R*)) / x,   x = D'/D
#   log R* = log K + ρ log(D/N) + σ log N
FIT_PARAMS = {
    'E': 0.050,
    'A': 31.5,
    'B': 16539.0,
    'alpha': 0.137,
    'beta': 0.436,
    'log_K': 10.32,
    'rho': -0.270,
    'sigma': -0.388,
}

# Parameter count N for each size key (matches the writeup's headline table).
N_FOR_SIZE = {
    '14m': 1.4e7,
    '30m': 3.0e7,
    '60m': 6.0e7,
    '100m': 1.0e8,
    '190m': 1.9e8,
    '370m': 3.7e8,
    '600m': 6.0e8,
}


def predict_loss(flops_mult, chinchilla_scale, N, params=FIT_PARAMS):
    """Predicted validation loss on the (FLOPs-multiplier, chinchilla-scale) grid.

    flops_multiplier = epochs · chinchilla_scale, so given (x=flops_mult, y=scale)
    we recover D = y · 20N and epochs = x / y. Below the y=x diagonal epochs<1
    is infeasible; the caller masks that region.
    """
    E, A, B = params['E'], params['A'], params['B']
    alpha, beta = params['alpha'], params['beta']
    log_K, rho, sigma = params['log_K'], params['rho'], params['sigma']

    flops_mult = np.asarray(flops_mult, dtype=float)
    chinchilla_scale = np.asarray(chinchilla_scale, dtype=float)

    D = chinchilla_scale * 20.0 * N
    epochs = flops_mult / chinchilla_scale
    D_prime = np.maximum(epochs - 1.0, 0.0) * D
    x_ratio = D_prime / D  # = max(epochs-1, 0)

    log_R_star = log_K + rho * np.log(D / N) + sigma * np.log(N)
    R_star = np.exp(log_R_star)

    # η → 1 as x_ratio → 0 (single-epoch limit).
    safe_x = np.where(x_ratio > 1e-12, x_ratio, 1.0)
    eta = np.where(
        x_ratio > 1e-12,
        R_star * (1.0 - np.exp(-safe_x / R_star)) / safe_x,
        1.0,
    )

    effective_tokens = D + eta * D_prime
    return E + A / np.power(N, alpha) + B / np.power(effective_tokens, beta)


def create_prediction_contour_plot(ax, N, x_lim, x_ticks, x_ticklabels, x_label,
                                    diag_label='Single-epoch scaling path',
                                    show_diagonal=True,
                                    font_label=18, font_tick=16, diag_linewidth=5,
                                    levels=None,
                                    overlay_x=None, overlay_y=None, overlay_z=None):
    """Plot iso-loss contours of the predicted L(N, D, D') surface.

    Mirrors the lines-only / RBF style of `create_contour_plot` so the result
    is directly comparable to the data-fitted figure. Optionally scatters the
    observed (x, y, z) points on top.
    """
    ax.set_facecolor('white')

    grid_log_x = np.linspace(np.log10(x_lim[0] * 0.8),
                             np.log10(x_lim[1] * 1.15), 300)
    grid_log_y = np.linspace(np.log10(0.04), np.log10(20), 300)
    grid_x, grid_y = np.meshgrid(grid_log_x, grid_log_y)

    grid_z = predict_loss(10**grid_x, 10**grid_y, N)

    grid_z_plot = grid_z.copy()
    grid_z_plot[grid_y > grid_x] = np.nan  # epochs < 1 is infeasible

    if levels is None:
        levels = auto_contour_levels(grid_z_plot[~np.isnan(grid_z_plot)])

    contour = ax.contour(
        10**grid_x, 10**grid_y, grid_z_plot,
        levels=levels, cmap=cmap, linewidths=2.0, alpha=0.9,
    )
    ax.clabel(contour, inline=True, fontsize=11, fmt='%.2f',
              colors='black', inline_spacing=5)

    if overlay_x is not None:
        ax.scatter(overlay_x, overlay_y, c=overlay_z, cmap=cmap,
                   s=80, edgecolors='black', linewidths=1.2, zorder=5,
                   vmin=overlay_z.min(), vmax=overlay_z.max())

    if show_diagonal:
        diag_x = np.array([x_lim[0], x_lim[1]])
        ax.plot(diag_x, diag_x, 'k--', linewidth=diag_linewidth, alpha=0.7,
                label=diag_label)

    # Grey out the upper-left infeasible region (above y=x).
    x_lo, x_hi = x_lim[0] * 0.8, x_lim[1] * 1.15
    y_hi = 20
    if y_hi <= x_hi:
        pts = [(x_lo, x_lo), (x_lo, y_hi), (y_hi, y_hi)]
    else:
        pts = [(x_lo, x_lo), (x_lo, y_hi), (x_hi, y_hi), (x_hi, x_hi)]
    ax.add_patch(Polygon(pts, closed=True, facecolor='lightgrey',
                         edgecolor='none', alpha=0.55, zorder=4))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(x_label, fontsize=font_label, fontweight='bold')
    ax.set_ylabel('Fresh Data D (TTP, Chinchilla x)',
                  fontsize=font_label, fontweight='bold')

    y_ticks = [0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['D=1N\n(0.05x)', 'D=2N\n(0.1x)', 'D=5N\n(0.25x)',
                        'D=10N\n(0.5x)', 'D=20N\n(1x)', 'D=40N\n(2x)',
                        'D=80N\n(4x)', 'D=160N\n(8x)', 'D=320N\n(16x)'],
                       fontsize=font_tick)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=font_tick)

    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_xlim(x_lim[0] * 0.8, x_lim[1] * 1.15)
    ax.set_ylim(0.04, 20)

    # Return the masked grid so steepest-descent sees the y>x infeasible region
    # as NaN. add_descent_paths fills NaN with the surface max and Gaussian-
    # smooths it, which builds a soft wall along y=x — without it the path
    # would creep up the left edge because the unmasked prediction surface is
    # monotone in D (no overfitting → minimum at the max-D corner).
    return grid_log_x, grid_log_y, grid_z_plot


# ── Per-model-size plotting configurations ────────────────────────────────────
# x-axis (flops/epochs) ranges and ticks differ across model sizes because the
# sweeps cover different chinchilla multipliers and epoch counts.
PLOT_CONFIGS = {
    '30m': {
        'module': 'dolma_30m',
        'flops_x_ticks': [0.05, 0.1, 0.5, 1, 4, 8, 16, 32, 64, 128, 256],
        'flops_x_ticklabels': ['0.05', '0.1', '0.5', '1', '4', '8', '16', '32', '64', '128', '256'],
        'flops_x_lim': (0.05, 256),
        'epochs_x_ticks': [1, 2, 4, 8, 16, 32, 64, 128],
        'epochs_x_ticklabels': ['1', '2', '4', '8', '16', '32', '64', '128'],
        'epochs_x_lim': (1, 128),
    },
    '370m': {
        'module': 'dolma_370m',
        'flops_x_ticks': [0.05, 0.1, 0.5, 1, 2, 4, 8, 16],
        'flops_x_ticklabels': ['0.05', '0.1', '0.5', '1', '2', '4', '8', '16'],
        'flops_x_lim': (0.05, 16),
        'epochs_x_ticks': [1, 2, 4, 8, 16, 32, 64],
        'epochs_x_ticklabels': ['1', '2', '4', '8', '16', '32', '64'],
        'epochs_x_lim': (1, 64),
    },
}


def make_plots_for_size(size):
    cfg = PLOT_CONFIGS[size]
    flops, epochs, data_amount, val_loss = load_dataset(cfg['module'])
    levels = auto_contour_levels(val_loss)
    # Use the sweep's actual max epochs as the over-trained boundary so the
    # masking polygon hugs the data envelope rather than an arbitrary cutoff.
    max_epochs = int(epochs.max())

    # Start at D=1N (smallest scale, 1-epoch) so the path traces the optimal
    # scale-up trajectory all the way to the model-optimal corner.
    flops_start_points = [(0.05, 0.05)]
    epochs_start_points = [(1.0, 0.05)]

    # ── Combined two-panel figure ─────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor('white')

    # Plot 1: FLOPS on x-axis
    scatter1, glx1, gly1, gz1 = create_contour_plot(
        ax=ax1,
        x_data=flops,
        y_data=data_amount,
        z_data=val_loss,
        x_label='FLOPs (in Chinchilla x)',
        x_ticks=cfg['flops_x_ticks'],
        x_ticklabels=cfg['flops_x_ticklabels'],
        x_lim=cfg['flops_x_lim'],
        diag_label='Single-epoch scaling path',
        levels=levels,
        mask_view='flops',
        max_epochs=max_epochs,
    )
    add_descent_paths(ax1, glx1, gly1, gz1, flops_start_points, color='cyan', lw=2.0)
    ax1.legend(loc='upper left', fontsize=15)

    # Plot 2: Epochs on x-axis
    scatter2, glx2, gly2, gz2 = create_contour_plot(
        ax=ax2,
        x_data=epochs,
        y_data=data_amount,
        z_data=val_loss,
        x_label='Epochs',
        x_ticks=cfg['epochs_x_ticks'],
        x_ticklabels=cfg['epochs_x_ticklabels'],
        x_lim=cfg['epochs_x_lim'],
        diag_label='Single-epoch scaling path',
        show_diagonal=False,
        levels=levels,
    )
    add_descent_paths(ax2, glx2, gly2, gz2, epochs_start_points, color='cyan', lw=2.0)
    ax2.legend(loc='upper left', fontsize=13)

    plt.subplots_adjust(right=0.88, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter2, cax=cbar_ax)
    cbar.set_label('Validation Loss', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)

    out_png = f'/n/home05/sqin/OLMo-core/results/isoloss_contour_{size}.png'
    plt.savefig(out_png, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved to {out_png}")

    # ── Paper-ready single-panel PDF (FLOPs view) ─────────────────────────────
    fig_left, ax_left = plt.subplots(1, 1, figsize=(10, 8))
    fig_left.patch.set_facecolor('white')
    scatter_left, glx_l, gly_l, gz_l = create_contour_plot(
        ax=ax_left,
        x_data=flops,
        y_data=data_amount,
        z_data=val_loss,
        x_label='FLOPs (in Chinchilla x)',
        x_ticks=cfg['flops_x_ticks'],
        x_ticklabels=cfg['flops_x_ticklabels'],
        x_lim=cfg['flops_x_lim'],
        diag_label='Single-epoch scaling path',
        font_label=18, font_tick=16, diag_linewidth=5,
        levels=levels,
        mask_view='flops',
        max_epochs=max_epochs,
        fill_style='lines_only',
        interp='rbf',
    )
    add_descent_paths(ax_left, glx_l, gly_l, gz_l, flops_start_points, color='cyan', lw=4.0)
    ax_left.legend(loc='upper left', fontsize=16)
    cbar_left = fig_left.colorbar(scatter_left, ax=ax_left, fraction=0.046, pad=0.08)
    cbar_left.set_label('Validation Loss', fontsize=18, fontweight='bold')
    cbar_left.ax.tick_params(labelsize=16)
    out_pdf = f'/n/home05/sqin/OLMo-core/results/isoloss_contour_{size}_left.pdf'
    fig_left.savefig(out_pdf, bbox_inches='tight',
                     facecolor='white', edgecolor='none')
    plt.close(fig_left)
    print(f"Left plot PDF saved to {out_pdf}")


def make_prediction_plot_for_size(size):
    """Single-panel PDF showing iso-loss contours predicted by the joint
    multi-epoch Chinchilla fit (writeup_final.md §2). Observed data points are
    overlaid so the user can read off prediction quality by eye."""
    cfg = PLOT_CONFIGS[size]
    N = N_FOR_SIZE[size]
    flops, _epochs, data_amount, val_loss = load_dataset(cfg['module'])
    levels = auto_contour_levels(val_loss)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.patch.set_facecolor('white')
    glx, gly, gz = create_prediction_contour_plot(
        ax=ax, N=N,
        x_lim=cfg['flops_x_lim'],
        x_ticks=cfg['flops_x_ticks'],
        x_ticklabels=cfg['flops_x_ticklabels'],
        x_label='FLOPs (in Chinchilla x)',
        levels=levels,
        overlay_x=flops, overlay_y=data_amount, overlay_z=val_loss,
    )
    add_descent_paths(ax, glx, gly, gz, [(0.05, 0.05)],
                      color='magenta', lw=4.0,
                      label='Predicted optimal (C,D) scaling path')
    ax.legend(loc='upper left', fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=val_loss.min(),
                                                  vmax=val_loss.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label('Predicted Validation Loss', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)

    out_pdf = (f'/n/home05/sqin/OLMo-core/results/'
               f'isoloss_contour_predicted_{size}_left.pdf')
    fig.savefig(out_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Predicted plot PDF saved to {out_pdf}")


def make_combined_plot_for_size(size):
    """Side-by-side empirical vs. predicted iso-loss contours, shared colorbar
    and bottom legend — same layout as the reference 'Empirical | Predicted'
    figure from the data-constrained scaling-laws paper."""
    cfg = PLOT_CONFIGS[size]
    N = N_FOR_SIZE[size]
    flops, epochs_data, data_amount, val_loss = load_dataset(cfg['module'])
    levels = auto_contour_levels(val_loss)
    max_epochs = int(epochs_data.max())

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    fig.patch.set_facecolor('white')

    # Left panel: data-fitted contours (RBF + phantom-projected over-trained corner).
    scatter_left, glx_l, gly_l, gz_l = create_contour_plot(
        ax=ax_left,
        x_data=flops, y_data=data_amount, z_data=val_loss,
        x_label='FLOPs (in Chinchilla x)',
        x_ticks=cfg['flops_x_ticks'],
        x_ticklabels=cfg['flops_x_ticklabels'],
        x_lim=cfg['flops_x_lim'],
        diag_label='Single-epoch scaling path',
        font_label=18, font_tick=16, diag_linewidth=5,
        levels=levels, mask_view='flops', max_epochs=max_epochs,
        fill_style='lines_only', interp='rbf',
    )
    add_descent_paths(ax_left, glx_l, gly_l, gz_l, [(0.05, 0.05)],
                      color='cyan', lw=4.0)
    ax_left.set_title('Empirical IsoLoss Contours',
                      fontsize=20, fontweight='bold', pad=12)

    # Right panel: contours of the joint Chinchilla + η model. No data overlay
    # (left panel already shows the points) and no predicted-optimal star (the
    # empirical star on the left panel is the reference).
    glx_r, gly_r, gz_r = create_prediction_contour_plot(
        ax=ax_right, N=N,
        x_lim=cfg['flops_x_lim'],
        x_ticks=cfg['flops_x_ticks'],
        x_ticklabels=cfg['flops_x_ticklabels'],
        x_label='FLOPs (in Chinchilla x)',
        levels=levels,
    )
    add_descent_paths(ax_right, glx_r, gly_r, gz_r, [(0.05, 0.05)],
                      color='magenta', lw=4.0,
                      label='Predicted optimal (C,D) scaling path')
    ax_right.set_title('Predicted IsoLoss Contours',
                       fontsize=20, fontweight='bold', pad=12)
    ax_right.set_ylabel('')

    # Mirror the empirical reference stars on the right panel for direct
    # comparison with the predicted contours.
    best_idx = int(np.argmin(val_loss))
    ax_right.scatter(flops[best_idx], data_amount[best_idx],
                     marker='*', s=400, c='white', edgecolors='black',
                     linewidths=2, zorder=10, label=r'Model-optimal $L_N$')
    baseline_idx = np.where(data_amount == 1)[0]
    if len(baseline_idx) > 0:
        min_x_idx = baseline_idx[int(np.argmin(flops[baseline_idx]))]
        ax_right.scatter(flops[min_x_idx], data_amount[min_x_idx],
                         marker='*', s=400, c='black', edgecolors='black',
                         linewidths=2, zorder=10,
                         label=r'Chinchilla-optimal $L_\mathrm{chin}$')

    # Shared bottom legend (dedupe by label across both axes).
    handles, labels, seen = [], [], set()
    for ax in (ax_left, ax_right):
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen and not l.startswith('_'):
                handles.append(h)
                labels.append(l)
                seen.add(l)
    fig.legend(handles, labels, loc='lower center', ncol=min(len(labels), 4),
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.subplots_adjust(right=0.90, wspace=0.06, bottom=0.18)
    cbar_ax = fig.add_axes([0.915, 0.18, 0.015, 0.70])
    cbar = fig.colorbar(scatter_left, cax=cbar_ax)
    cbar.set_label('Validation Loss', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=15)

    out_pdf = (f'/n/home05/sqin/OLMo-core/results/'
               f'isoloss_contour_combined_{size}.pdf')
    fig.savefig(out_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Combined plot PDF saved to {out_pdf}")


if __name__ == '__main__':
    for size in ('30m', '370m'):
        make_plots_for_size(size)
        make_prediction_plot_for_size(size)
        make_combined_plot_for_size(size)
