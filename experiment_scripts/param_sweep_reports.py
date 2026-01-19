import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Dict, Tuple, List, Optional


def format_step_number(step: int) -> str:
    """Format step number as 1.5k, 2k, etc."""
    if step >= 1000:
        k_value = step / 1000
        if k_value == int(k_value):
            return f"{int(k_value)}k"
        else:
            return f"{k_value:.1f}k"
    return str(step)


def parse_run_name(column_name: str) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Parse a column name to extract case name, weight decay, and learning rate.

    Args:
        column_name: Column name like "370M_seed42_case3_dclm_repeat_wd1.6_lr0.0006 - eval/lm/dclm-validation/CE loss"

    Returns:
        Tuple of (case_name, weight_decay, learning_rate)
    """
    pattern = r'370M_seed\d+_(case\d+_\w+)_wd([\d.]+)_lr([\d.]+)\s*-\s*eval/lm/dclm-validation/CE loss"?$'
    match = re.search(pattern, column_name)

    if match:
        case_name = match.group(1)
        weight_decay = float(match.group(2))
        learning_rate = float(match.group(3))
        return case_name, weight_decay, learning_rate

    return None, None, None


def extract_last_valid_loss(df: pd.DataFrame, column: str) -> Tuple[float, int]:
    """
    Extract the last valid (non-empty) validation loss and its step number.

    Args:
        df: DataFrame containing the data
        column: Column name to extract from

    Returns:
        Tuple of (last_valid_loss, last_step)
    """
    values = df[column].values
    steps = df['Step'].values

    # Find all non-empty values
    valid_mask = (values != '') & (~pd.isna(values))
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) > 0:
        last_idx = valid_indices[-1]
        last_loss = float(values[last_idx])
        last_step = int(steps[last_idx])
        return last_loss, last_step

    return np.nan, 0


def load_param_sweep_data(csv_path: str, case_filter: str) -> Dict[Tuple[float, float], Tuple[float, int]]:
    """
    Load parameter sweep data from a CSV file.

    Args:
        csv_path: Path to the CSV file
        case_filter: Case name to filter for (e.g., "case3_dclm_repeat")

    Returns:
        Dictionary mapping (weight_decay, learning_rate) to (validation_loss, last_step)
    """
    df = pd.read_csv(csv_path)
    results = {}

    for col in df.columns:
        if col == 'Step':
            continue

        # Skip MIN and MAX columns
        if '__MIN' in col or '__MAX' in col:
            continue

        # Parse the column name
        case_name, weight_decay, learning_rate = parse_run_name(col)

        if case_name is None:
            continue

        if case_name != case_filter:
            continue

        # Extract the last valid loss and step
        last_loss, last_step = extract_last_valid_loss(df, col)

        if not np.isnan(last_loss):
            key = (weight_decay, learning_rate)
            # If we already have data for this hyperparameter combination,
            # keep the one with the higher step count (most progress)
            if key not in results or last_step > results[key][1]:
                results[key] = (last_loss, last_step)

    return results


def plot_param_sweep_heatmap(
    csv_path: str,
    case_name: str,
    output_path: str,
    weight_decays: List[float] = [0.1, 0.4, 1.6],
    learning_rates: List[float] = [0.0003, 0.0006, 0.001, 0.003],
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis_r"  # reversed viridis so lower loss is lighter
):
    """
    Plot a heatmap of validation loss across weight decay and learning rate values.

    Args:
        csv_path: Path to the CSV file with wandb results
        case_name: Case name to filter for (e.g., "case3_dclm_repeat" or "case4_dclm_extended")
        output_path: Path to save the output plot
        weight_decays: List of weight decay values in the sweep
        learning_rates: List of learning rate values in the sweep
        figsize: Figure size (width, height)
        cmap: Colormap to use for the heatmap
    """
    # Load data
    results = load_param_sweep_data(csv_path, case_name)

    # Create grid
    n_wd = len(weight_decays)
    n_lr = len(learning_rates)
    loss_grid = np.full((n_wd, n_lr), np.nan)
    step_grid = np.full((n_wd, n_lr), 0, dtype=int)

    # Fill grid
    for i, wd in enumerate(weight_decays):
        for j, lr in enumerate(learning_rates):
            key = (wd, lr)
            if key in results:
                loss_grid[i, j] = results[key][0]
                step_grid[i, j] = results[key][1]

    # Create plot
    _, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        loss_grid,
        annot=False,  # We'll add custom annotations
        fmt='.3f',
        cmap=cmap,
        cbar_kws={'label': 'Validation Loss (CE)'},
        xticklabels=[f'{lr:.0e}' for lr in learning_rates],
        yticklabels=[f'{wd:.1f}' for wd in weight_decays],
        ax=ax,
        vmin=np.nanmin(loss_grid),
        vmax=np.nanmax(loss_grid),
        square=False
    )

    # Add custom annotations with loss and step
    for i in range(n_wd):
        for j in range(n_lr):
            if not np.isnan(loss_grid[i, j]):
                loss_text = f'{loss_grid[i, j]:.3f}'
                step_text = f'({format_step_number(step_grid[i, j])})'

                # Add loss value
                ax.text(
                    j + 0.5, i + 0.4,
                    loss_text,
                    ha='center', va='center',
                    fontsize=10,
                    fontweight='bold',
                    color='white' if loss_grid[i, j] > np.nanmedian(loss_grid) else 'black'
                )

                # Add step count below
                ax.text(
                    j + 0.5, i + 0.65,
                    step_text,
                    ha='center', va='center',
                    fontsize=8,
                    color='white' if loss_grid[i, j] > np.nanmedian(loss_grid) else 'black'
                )

    # Set labels and title
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Decay', fontsize=12, fontweight='bold')

    # Format case name for title
    case_display = case_name.replace('_', ' ').title()
    ax.set_title(f'Parameter Sweep: {case_display}\nValidation Loss and Last Step',
                 fontsize=14, fontweight='bold', pad=20)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_path}")

    # Close plot
    plt.close()


def generate_all_heatmaps(
    results_dir: str = "results",
    output_dir: str = "results"
):
    """
    Generate heatmaps for all parameter sweep cases.

    Args:
        results_dir: Directory containing the CSV files
        output_dir: Directory to save the output plots
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Case 3: dclm_repeat
    case3_csv = results_path / "case3_dclm_repeat.csv"
    if case3_csv.exists():
        plot_param_sweep_heatmap(
            csv_path=str(case3_csv),
            case_name="case3_dclm_repeat",
            output_path=str(output_path / "case3_dclm_repeat_heatmap.png")
        )
    else:
        print(f"Warning: {case3_csv} not found")

    # Case 4: dclm_extended
    case4_csv = results_path / "case4_dclm_extend.csv"
    if case4_csv.exists():
        plot_param_sweep_heatmap(
            csv_path=str(case4_csv),
            case_name="case4_dclm_extended",
            output_path=str(output_path / "case4_dclm_extended_heatmap.png")
        )
    else:
        print(f"Warning: {case4_csv} not found")


if __name__ == "__main__":
    # Generate all heatmaps
    generate_all_heatmaps()
