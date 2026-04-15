import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.integrate import solve_ivp
from matplotlib.colors import LinearSegmentedColormap

# Import data from dclm_30m.py
from dclm_30m import ALL_DATASETS

# Combine all data
all_data = ALL_DATASETS

flops = []
epochs = []
data_amount = []
val_loss = []

for data in all_data:
    for i in range(len(data['flops_multiplier'])):
        if not np.isnan(data['validation_loss'][i]):
            flops.append(data['flops_multiplier'][i])
            epochs.append(data['epochs'][i])
            data_amount.append(data['chinchilla_scale'][i])
            val_loss.append(data['validation_loss'][i])

flops = np.array(flops)
epochs = np.array(epochs)
data_amount = np.array(data_amount)
val_loss = np.array(val_loss)

# Create custom colormap (lighter brown/red to orange/yellow - adjusted so lower end is not completely black)
colors = ['#8b3a1e', '#a84828', '#c45a32', '#d97040', '#e88850',
          '#f4a060', '#ffb870', '#ffc880', '#ffd890', '#ffe8a0']
cmap = LinearSegmentedColormap.from_list('isoloss', colors, N=512)


def create_contour_plot(ax, x_data, y_data, z_data, x_label, x_ticks, x_ticklabels,
                        x_lim, diag_label, show_diagonal=True,
                        font_label=16, font_tick=15, diag_linewidth=2.5):
    """Create a contour plot with the given data and settings on the provided axis.

    Returns (scatter, grid_log_x_1d, grid_log_y_1d, grid_z) so callers can
    use the interpolated surface for gradient tracing.
    """

    ax.set_facecolor('white')

    # Create grid for interpolation (in log space for better results)
    log_x = np.log10(x_data)
    log_y = np.log10(y_data)

    # Create fine grid
    grid_log_x = np.linspace(log_x.min() - 0.1, log_x.max() + 0.1, 200)
    grid_log_y = np.linspace(log_y.min() - 0.1, log_y.max() + 0.1, 200)
    grid_x, grid_y = np.meshgrid(grid_log_x, grid_log_y)

    # Interpolate
    grid_z = griddata(
        (log_x, log_y),
        z_data,
        (grid_x, grid_y),
        method='cubic'
    )

    # Define contour levels (finer step for stricter iso-loss bands)
    levels = np.arange(3.85, 5.4, 0.05)

    # Plot filled contours
    contourf = ax.contourf(
        10**grid_x, 10**grid_y, grid_z,
        levels=levels,
        cmap=cmap,
        alpha=0.7,
        extend='both'
    )

    # Plot contour lines
    contour = ax.contour(
        10**grid_x, 10**grid_y, grid_z,
        levels=levels,
        colors='darkred',
        linewidths=0.8,
        alpha=0.9
    )

    # Add contour labels
    ax.clabel(contour, inline=True, fontsize=9, fmt='%.2f',
              colors='darkred', inline_spacing=5)

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

    # Add validation loss labels next to each dot
    for i in range(len(x_data)):
        ax.annotate(f'{z_data[i]:.2f}',
                    (x_data[i], y_data[i]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             edgecolor='gray', alpha=0.8, linewidth=0.5),
                    zorder=6)

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
    # ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Customize tick labels for y-axis (TTP = chinchilla_scale * 20)
    y_ticks = [0.05, 0.1, 0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['D=1N\n(0.05x)', 'D=2N\n(0.1x)', 'D=10N\n(0.5x)', 'D=20N\n(1x)',
                        'D=40N\n(2x)', 'D=80N\n(4x)', 'D=160N\n(8x)', 'D=320N\n(16x)'],
                       fontsize=font_tick)

    # Customize tick labels for x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=font_tick)

    # Add a diagonal line (iso-compute line) similar to reference
    if show_diagonal:
        diag_x = np.array([x_lim[0], x_lim[1]])
        diag_y = diag_x  # slope of 1 in log-log space
        ax.plot(diag_x, diag_y, 'k--', linewidth=diag_linewidth, alpha=0.7, label=diag_label)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Legend is added after descent paths are drawn (see call site)

    # Set axis limits
    ax.set_xlim(x_lim[0] * 0.8, x_lim[1] * 1.15)
    ax.set_ylim(0.04, 20)

    return scatter, grid_log_x, grid_log_y, grid_z


def add_descent_paths(ax, grid_log_x, grid_log_y, grid_z, start_points_xy,
                      color='cyan', lw=2.0, alpha=0.95, label='Optimal scaling path'):
    """Trace and overlay steepest-descent paths from start_points_xy.

    Parameters
    ----------
    start_points_xy : list of (x, y) in original (non-log) data space.
    """
    # Fill NaN regions with the max observed loss so the gradient creates a
    # natural repelling wall that keeps paths inside the valid interpolation zone.
    grid_z_filled = grid_z.copy()
    grid_z_filled[np.isnan(grid_z_filled)] = np.nanmax(grid_z_filled)

    # Gradient components in log space: grid_z shape is (ny, nx)
    dz_dlogx = np.gradient(grid_z_filled, grid_log_x, axis=1)
    dz_dlogy = np.gradient(grid_z_filled, grid_log_y, axis=0)

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



# Single starting point: 1x chinchilla, 1-epoch baseline (standard training setup).
# The path traces the optimal direction to scale compute and data from here.
flops_start_points = [(0.2, 0.1)]
epochs_start_points = [(1.0, 0.1)]

# ── Create figure with two side-by-side subplots ──────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('white')

# Plot 1: FLOPS on x-axis
scatter1, glx1, gly1, gz1 = create_contour_plot(
    ax=ax1,
    x_data=flops,
    y_data=data_amount,
    z_data=val_loss,
    x_label='FLOPs (in Chinchilla x)',
    x_ticks=[0.05, 0.1, 0.5, 1, 4, 8, 16, 32, 64, 128],
    x_ticklabels=['0.05', '0.1', '0.5', '1', '4', '8', '16', '32', '64', '128'],
    # title='Loss Contours (FLOPS)',
    x_lim=(0.05, 128),
    diag_label='Single-epoch scaling path'
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
    x_ticks=[1, 2, 4, 8, 16, 32, 64, 128],
    x_ticklabels=['1', '2', '4', '8', '16', '32', '64', '128'],
x_lim=(1, 128),
    diag_label='Single-epoch scaling path',
    show_diagonal=False
)
add_descent_paths(ax2, glx2, gly2, gz2, epochs_start_points, color='cyan', lw=2.0)
ax2.legend(loc='upper left', fontsize=13)

# Adjust subplot spacing and add colorbar on the right
plt.subplots_adjust(right=0.88, wspace=0.25)
cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cbar = fig.colorbar(scatter2, cax=cbar_ax)
cbar.set_label('Validation Loss', fontsize=14, fontweight='bold')
cbar.ax.tick_params(labelsize=15)

plt.savefig('/n/home05/sqin/OLMo-core/results/isoloss_contour.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Plot saved to isoloss_contour.png")

# ── Save left plot as paper-ready PDF ─────────────────────────────────────────
fig_left, ax_left = plt.subplots(1, 1, figsize=(10, 8))
fig_left.patch.set_facecolor('white')
scatter_left, glx_l, gly_l, gz_l = create_contour_plot(
    ax=ax_left,
    x_data=flops,
    y_data=data_amount,
    z_data=val_loss,
    x_label='FLOPs (in Chinchilla x)',
    x_ticks=[0.05, 0.1, 0.5, 1, 4, 8, 16, 32, 64, 128],
    x_ticklabels=['0.05', '0.1', '0.5', '1', '4', '8', '16', '32', '64', '128'],
    # title='Loss Contours (FLOPS)',
    x_lim=(0.05, 128),
    diag_label='Single-epoch scaling path',
    font_label=18, font_tick=16, diag_linewidth=5,
)
add_descent_paths(ax_left, glx_l, gly_l, gz_l, flops_start_points, color='cyan', lw=4.0)
ax_left.legend(loc='upper left', fontsize=16)
cbar_left = fig_left.colorbar(scatter_left, ax=ax_left, fraction=0.046, pad=0.08)
cbar_left.set_label('Validation Loss', fontsize=18, fontweight='bold')
cbar_left.ax.tick_params(labelsize=16)
fig_left.savefig('/n/home05/sqin/OLMo-core/results/isoloss_contour_left.pdf',
                 bbox_inches='tight', facecolor='white', edgecolor='none')
print("Left plot PDF saved to isoloss_contour_left.pdf")
