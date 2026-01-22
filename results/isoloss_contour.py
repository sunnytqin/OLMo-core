import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap

# Import data from case4_analysis.py
from case4_analysis import data_0_05x, data_0_5x, data_1x, data_2x, data_4x, data_8x, data_16x

# Combine all data
all_data = [data_0_05x, data_0_5x, data_1x, data_2x, data_4x, data_8x, data_16x]

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
cmap = LinearSegmentedColormap.from_list('isoloss', colors, N=256)


def create_contour_plot(ax, x_data, y_data, z_data, x_label, x_ticks, x_ticklabels,
                        title, x_lim, diag_label, show_diagonal=True):
    """Create a contour plot with the given data and settings on the provided axis."""

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

    # Define contour levels
    levels = np.arange(3.85, 5.4, 0.1)

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
               linewidths=2, zorder=10)

    # Mark 1x 1-epoch point with a black filled star
    # Find the point where y_data (chinchilla scale) is 1
    baseline_idx = np.where(y_data == 1)[0]
    if len(baseline_idx) > 0:
        # Find the one with minimum x_data (1 epoch = lowest flops for that scale)
        min_x_idx = baseline_idx[np.argmin(x_data[baseline_idx])]
        ax.scatter(x_data[min_x_idx], y_data[min_x_idx],
                   marker='*', s=400, c='black', edgecolors='black',
                   linewidths=2, zorder=10)

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set axis labels and title
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel('Fresh Data Size (Chinchilla Scale)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Customize tick labels for y-axis
    y_ticks = [0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['0.5x', '1x', '2x', '4x', '8x', '16x'])

    # Customize tick labels for x-axis
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Add a diagonal line (iso-compute line) similar to reference
    if show_diagonal:
        diag_x = np.array([x_lim[0], x_lim[1]])
        diag_y = diag_x  # slope of 1 in log-log space
        ax.plot(diag_x, diag_y, 'k--', linewidth=2.5, alpha=0.7, label=diag_label)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Add legend (only if diagonal line is shown)
    if show_diagonal:
        ax.legend(loc='upper left', fontsize=10)

    # Set axis limits
    ax.set_xlim(x_lim[0] * 0.8, x_lim[1] * 1.15)
    ax.set_ylim(0.04, 20)

    return scatter


# Create figure with two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
fig.patch.set_facecolor('white')

# Plot 1: FLOPS on x-axis
scatter1 = create_contour_plot(
    ax=ax1,
    x_data=flops,
    y_data=data_amount,
    z_data=val_loss,
    x_label='FLOPS Multiplier',
    x_ticks=[0.5, 1, 4, 8, 16, 32, 64, 128],
    x_ticklabels=['0.5', '1', '4', '8', '16', '32', '64', '128'],
    title='Loss Contours (FLOPS)',
    x_lim=(0.5, 128),
    diag_label='Single-epoch baseline'
)

# Plot 2: Epochs on x-axis
scatter2 = create_contour_plot(
    ax=ax2,
    x_data=epochs,
    y_data=data_amount,
    z_data=val_loss,
    x_label='Epochs',
    x_ticks=[1, 2, 4, 8, 16, 32, 64, 128],
    x_ticklabels=['1', '2', '4', '8', '16', '32', '64', '128'],
    title='Loss Contours (Epochs)',
    x_lim=(1, 128),
    diag_label='Single-epoch baseline',
    show_diagonal=False
)

# Adjust subplot spacing and add colorbar on the right
plt.subplots_adjust(right=0.92, wspace=0.25)
cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
cbar = fig.colorbar(scatter2, cax=cbar_ax)
cbar.set_label('Validation Loss', fontsize=12, fontweight='bold')

plt.savefig('/n/home05/sqin/OLMo-core/results/isoloss_contour.png', dpi=150,
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Plot saved to isoloss_contour.png")
