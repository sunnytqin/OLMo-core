import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
from typing import Dict, Tuple, List


def parse_chinchilla_run_name(run_name: str) -> Tuple[str, float, float]:
    """
    Parse a run name to extract case name, weight decay, and learning rate.

    Args:
        run_name: Run name like "30M_seed42_case3_dclm_repeat_wd0.1_lr1e-3"

    Returns:
        Tuple of (case_name, weight_decay, learning_rate)
    """
    # Pattern: 30M_seed42_{case_name}_wd{wd}_lr{lr}
    pattern = r'30M_seed\d+_(case\d+_\w+)_wd([\d.]+)_lr([\d.e-]+)'
    match = re.search(pattern, run_name)

    if match:
        case_name = match.group(1)
        weight_decay = float(match.group(2))

        # Parse learning rate (handle scientific notation like 1e-3)
        lr_str = match.group(3)
        learning_rate = float(lr_str)

        return case_name, weight_decay, learning_rate

    return None, None, None


def load_chinchilla_data(json_path: str, case_filter: str) -> Dict[Tuple[float, float], float]:
    """
    Load Chinchilla evaluation results from a JSON file.

    Args:
        json_path: Path to the JSON file
        case_filter: Case name to filter for (e.g., "case3_dclm_repeat")

    Returns:
        Dictionary mapping (weight_decay, learning_rate) to validation_loss
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    results = {}

    for run_name, metrics in data.items():
        case_name, weight_decay, learning_rate = parse_chinchilla_run_name(run_name)

        if case_name is None:
            continue

        if case_name != case_filter:
            continue

        validation_loss = metrics.get('validation_loss')

        if validation_loss is not None:
            key = (weight_decay, learning_rate)
            results[key] = validation_loss

    return results


def plot_chinchilla_heatmap(
    json_path: str,
    case_name: str,
    output_path: str,
    chinchilla_multiplier: str,
    weight_decays: List[float] = [0.1, 0.4, 0.8, 1.6],
    learning_rates: List[float] = [3e-4, 6e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2, 6e-2],
    figsize: Tuple[int, int] = (10, 6),
    cmap: str = "viridis_r"  # reversed viridis so lower loss is lighter
):
    """
    Plot a heatmap of validation loss across weight decay and learning rate values.

    Args:
        json_path: Path to the JSON file with evaluation results
        case_name: Case name to filter for (e.g., "case3_dclm_repeat" or "case4_dclm_extended")
        output_path: Path to save the output plot
        chinchilla_multiplier: String like "4x" or "16x" for the title
        weight_decays: List of weight decay values in the sweep
        learning_rates: List of learning rate values in the sweep
        figsize: Figure size (width, height)
        cmap: Colormap to use for the heatmap
    """
    # Load data
    results = load_chinchilla_data(json_path, case_name)

    # Create grid
    n_wd = len(weight_decays)
    n_lr = len(learning_rates)
    loss_grid = np.full((n_wd, n_lr), np.nan)

    # Fill grid
    for i, wd in enumerate(weight_decays):
        for j, lr in enumerate(learning_rates):
            key = (wd, lr)
            if key in results:
                loss_grid[i, j] = results[key]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        loss_grid,
        annot=True,  # Show values
        fmt='.3f',
        cmap=cmap,
        cbar_kws={'label': 'Validation Loss (CE)'},
        xticklabels=[f'{lr:.0e}' for lr in learning_rates],
        yticklabels=[f'{wd:.1f}' for wd in weight_decays],
        ax=ax,
        vmin=np.nanmin(loss_grid),
        vmax=np.nanmax(loss_grid),
        square=False,
        annot_kws={'fontsize': 11, 'fontweight': 'bold'}
    )

    # Customize cell text colors for better readability
    median_loss = np.nanmedian(loss_grid)
    for text_obj in ax.texts:
        try:
            # Get the text value
            text_val = float(text_obj.get_text())
            # Use white text for darker cells (higher loss), black for lighter cells (lower loss)
            text_color = 'white' if text_val > median_loss else 'black'
            text_obj.set_color(text_color)
        except ValueError:
            # If text is not a number, skip it
            pass

    # Set labels and title
    ax.set_xlabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('Weight Decay', fontsize=12, fontweight='bold')

    # Format case name for title
    case_display = case_name.replace('_', ' ').title()
    ax.set_title(f'Chinchilla {chinchilla_multiplier} - {case_display}\nValidation Loss',
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


def generate_all_chinchilla_heatmaps(
    results_dir: str = "../results",
    output_dir: str = "../results"
):
    """
    Generate heatmaps for all Chinchilla parameter sweep cases.

    Args:
        results_dir: Directory containing the JSON files
        output_dir: Directory to save the output plots
    """
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Chinchilla 4x
    chinchilla4_json = results_path / "eval_results_chinchilla4.json"
    if chinchilla4_json.exists():
        # Case 2: dclm_synthetic
        plot_chinchilla_heatmap(
            json_path=str(chinchilla4_json),
            case_name="case2_dclm_synthetic",
            output_path=str(output_path / "chinchilla4_case2_dclm_synthetic_heatmap.png"),
            chinchilla_multiplier="4x"
        )

        # Case 3: dclm_repeat
        plot_chinchilla_heatmap(
            json_path=str(chinchilla4_json),
            case_name="case3_dclm_repeat",
            output_path=str(output_path / "chinchilla4_case3_dclm_repeat_heatmap.png"),
            chinchilla_multiplier="4x"
        )

        # Case 4: dclm_extended
        plot_chinchilla_heatmap(
            json_path=str(chinchilla4_json),
            case_name="case4_dclm_extended",
            output_path=str(output_path / "chinchilla4_case4_dclm_extended_heatmap.png"),
            chinchilla_multiplier="4x"
        )
    else:
        print(f"Warning: {chinchilla4_json} not found")

    # Chinchilla 8x
    chinchilla8_json = results_path / "eval_results_chinchilla8.json"
    if chinchilla8_json.exists():
        # Case 2: dclm_synthetic
        plot_chinchilla_heatmap(
            json_path=str(chinchilla8_json),
            case_name="case2_dclm_synthetic",
            output_path=str(output_path / "chinchilla8_case2_dclm_synthetic_heatmap.png"),
            chinchilla_multiplier="8x"
        )

        # Case 3: dclm_repeat
        plot_chinchilla_heatmap(
            json_path=str(chinchilla8_json),
            case_name="case3_dclm_repeat",
            output_path=str(output_path / "chinchilla8_case3_dclm_repeat_heatmap.png"),
            chinchilla_multiplier="8x"
        )

        # Case 4: dclm_extended
        plot_chinchilla_heatmap(
            json_path=str(chinchilla8_json),
            case_name="case4_dclm_extended",
            output_path=str(output_path / "chinchilla8_case4_dclm_extended_heatmap.png"),
            chinchilla_multiplier="8x"
        )
    else:
        print(f"Warning: {chinchilla8_json} not found")

    # Chinchilla 16x
    chinchilla16_json = results_path / "eval_results_chinchilla16.json"
    if chinchilla16_json.exists():
        # Case 2: dclm_synthetic
        plot_chinchilla_heatmap(
            json_path=str(chinchilla16_json),
            case_name="case2_dclm_synthetic",
            output_path=str(output_path / "chinchilla16_case2_dclm_synthetic_heatmap.png"),
            chinchilla_multiplier="16x"
        )

        # Case 3: dclm_repeat
        plot_chinchilla_heatmap(
            json_path=str(chinchilla16_json),
            case_name="case3_dclm_repeat",
            output_path=str(output_path / "chinchilla16_case3_dclm_repeat_heatmap.png"),
            chinchilla_multiplier="16x"
        )

        # Case 4: dclm_extended
        plot_chinchilla_heatmap(
            json_path=str(chinchilla16_json),
            case_name="case4_dclm_extended",
            output_path=str(output_path / "chinchilla16_case4_dclm_extended_heatmap.png"),
            chinchilla_multiplier="16x"
        )
    else:
        print(f"Warning: {chinchilla16_json} not found")


if __name__ == "__main__":
    # Generate all heatmaps
    generate_all_chinchilla_heatmaps()
