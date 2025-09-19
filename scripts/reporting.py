import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from typing import Iterable, Mapping, Optional
import argparse
import sys

# Dictionary mapping run names to visualization names
# Edit this dictionary to customize how run names appear in plots
RUN_NAME_TO_VIZ_NAME = {
    "A_no_aug": "No Aug",
    "B_light_aug": "Light Aug (1x)", 
    "C_heavy_aug": "Heavy Aug (1x)",
    "D_light_3": "YOLOv8 Nano",
    "E_light_4": "Light Aug (3x)",
    "F_light_8": "Light Aug (7x)",
    "G_no_geo_aug": "No Geo Aug",
    "H_scale_shear_rotate_geo_aug": "All but Translate Aug",
    "I_all_geo_aug": "All Geo Aug",
    "v8": "YOLOv8",
    "v12_nano": "YOLOv12 Nano 640",
    "v12": "YOLOv12",
    "FasterRCNN": "FasterRCNN",
    "v12_nano_768": "YOLOv12 Nano 768",
    "v12_nano_512": "YOLOv12 Nano 512",
    "v12_nano_440": "YOLOv12 Nano 440",
    "v12_nano_rect": "YOLOv12 Nano 640 Rect",
    "Q1": "10x lower LR",
    "Q2": "10x higher LR",
    "Q3": "long warmup, cosine LR",
    "Q4": "strong decay, linear LR",
    "Q5": "high momentum, light decay",
    "Q6": "low momentum, heavy decay",
    # Add more mappings as needed
}

def load_mission_mapping(splits_root: str = "data/splits_lomo_test_val_only") -> dict:
    """
    Load the mapping from fold indices to mission IDs from LOMO splits directory.
    
    Args:
        splits_root: Directory containing fold_* subdirectories
        
    Returns:
        Dictionary mapping fold_index -> mission_id
    """
    from pathlib import Path
    
    mission_mapping = {}
    splits_path = Path(splits_root)
    
    if not splits_path.exists():
        print(f"Warning: Splits directory {splits_root} not found. Using fold names.")
        return {}
    
    for child in sorted(splits_path.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("fold_"):
            continue
        # Expect pattern fold_<idx>_<mission>
        remainder = name[len("fold_"):]
        parts = remainder.split("_", 1)
        if len(parts) != 2:
            continue
        idx_str, mission_id = parts[0], parts[1]
        try:
            idx = int(idx_str)
            mission_mapping[idx] = mission_id
        except ValueError:
            continue
    
    return mission_mapping

def plot_metric_comparison_by_fold(
    excel_path: str,
    sheet_name: str = "results",
    runs: Optional[Iterable[str]] = None,     # <- order matters; controls legend & bar order
    metric: str = "F1@best",  # <- metric to plot: "F1@best", "mean_IoU@TP", "mAP50-95", "best_epoch", "best_conf"
    run_name_top_label: str = "run_name",
    save_path: Optional[str] = None,
    figsize=(11, 5.5),
    show_values: bool = True,
    value_fmt: str = "{:.3f}",
    ylim=(0.0, 1.02),
    bar_alpha: float = 0.95,
    edge_alpha: float = 0.65,
    # map run -> color (use your brand colors or keep these)
    run_colors: Optional[Mapping[str, str]] = None,  # e.g., {"A_no_aug":"#1f77b4","B_light_aug":"#ff7f0e"}
    splits_root: str = "data/splits_lomo_test_val_only",  # Directory containing LOMO fold directories
):
    """
    Make a grouped bar chart for the specified metric across missions with a final 'mean' group.
    Major groups: mission names (from LOMO folds) and 'mean'.
    Minor bars inside each group: selected runs (in the *order* provided).
    """

    # ---------- Load & parse ----------
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])

    # Resolve run-name column (top-level 'run_name', one subcol created by Excel merges)
    if run_name_top_label not in df.columns.get_level_values(0):
        raise ValueError(f"Top-level '{run_name_top_label}' not found in columns.")
    run_name_sub = df[run_name_top_label].columns[0]
    run_col = (run_name_top_label, run_name_sub)

    # Resolve metric folds
    if metric not in df.columns.get_level_values(0):
        raise ValueError(f"Top-level '{metric}' not found in columns.")
    metric_subcols = list(df[metric].columns)

    # Folds (fold1..foldN) + mean at the end if present
    fold_cols = sorted(
        [c for c in metric_subcols if str(c).lower().startswith("fold")],
        key=lambda x: int("".join(filter(str.isdigit, str(x))) or 0),
    )
    
    # Load mission mapping to convert fold numbers to mission names
    mission_mapping = load_mission_mapping(splits_root)
    
    # Convert fold names to mission names if available
    groups = []
    fold_to_mission = {}  # Map original fold column names to mission names
    
    for fold_col in fold_cols:
        if str(fold_col).lower().startswith("fold"):
            fold_num = int("".join(filter(str.isdigit, str(fold_col))) or 0)
            if fold_num in mission_mapping:
                mission_name = mission_mapping[fold_num]
                groups.append(mission_name)
                fold_to_mission[fold_col] = mission_name
            else:
                groups.append(str(fold_col))  # fallback to original name
                fold_to_mission[fold_col] = str(fold_col)
        else:
            groups.append(str(fold_col))
            fold_to_mission[fold_col] = str(fold_col)
    
    # Add mean at the end if present
    if "mean" in metric_subcols:
        groups.append("mean")
        fold_to_mission["mean"] = "mean"

    # If runs not provided, default to all (in sheet order)
    if runs is None:
        runs = list(df[run_col].astype(str).unique())
    else:
        runs = list(runs)  # ensure indexable

    # Build values matrix [n_groups, n_runs]
    n_groups = len(groups)
    n_runs = len(runs)
    values = np.full((n_groups, n_runs), np.nan, dtype=float)

    # Create a lookup by run name for fast access
    df_idx = df.set_index(run_col)

    for j, run in enumerate(runs):
        if run not in df_idx.index:
            continue
        row = df_idx.loc[run]
        for i, grp in enumerate(groups):
            # Find the original fold column name that maps to this group
            original_fold_col = None
            for fold_col, mission_name in fold_to_mission.items():
                if mission_name == grp:
                    original_fold_col = fold_col
                    break
            
            if original_fold_col is not None:
                col = (metric, original_fold_col)
                if col in df.columns:
                    values[i, j] = row[col]

    # ---------- Styling defaults ----------
    # Nice, readable defaults without global rcParams fuss
    metric_labels = {
        "F1@best": "F1 Score at Best Confidence Threshold",
        "mean_IoU@TP": "Mean IoU at True Positives",
        "mAP50-95": "mAP@0.5:0.95",
        "best_epoch": "Best Epoch",
        "best_conf": "Best Confidence Threshold"
    }
    
    title = metric_labels.get(metric, f"{metric} by Mission and Mean")
    xlabel = "Mission"
    ylabel = metric.replace("@", " ").replace("_", " ").title()

    # Provide stable colors per run (if not given)
    if run_colors is None:
        # Use Matplotlib tab10 in the run order so legend matches bars consistently
        base_colors = plt.cm.tab10.colors
        run_colors = {run: base_colors[i % len(base_colors)] for i, run in enumerate(runs)}

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_groups)
    total_width = 0.82
    bar_width = total_width / max(n_runs, 1)
    offsets = (np.arange(n_runs) - (n_runs - 1) / 2.0) * bar_width

    # Bars per run
    for j, run in enumerate(runs):
        bar_vals = values[:, j]
        # Use visualization name if available, otherwise use original run name
        viz_name = RUN_NAME_TO_VIZ_NAME.get(run, run)
        ax.bar(
            x + offsets[j],
            bar_vals,
            width=bar_width * 0.92,
            label=viz_name,
            alpha=bar_alpha,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
            color=run_colors.get(run, None),
        )

        # Optional value labels
        if show_values:
            for xi, yi in zip(x, bar_vals):
                if np.isnan(yi):
                    continue
                ax.text(
                    xi + offsets[j],
                    yi + 0.012,              # small lift above the bar
                    value_fmt.format(yi),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    rotation=0,
                    zorder=4,
                )

    # X ticks & labels
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, pad=10)

    # Y-axis formatting
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0)

    # Thin separator line before 'mean' group if present
    if "mean" in groups:
        mean_idx = groups.index("mean")
        if mean_idx > 0:
            ax.axvline(mean_idx - 0.5, linestyle=":", linewidth=1.0, color="0.4", alpha=0.7)

    # Legend: match your run order exactly
    leg = ax.legend(title="Run", frameon=True, fontsize=10, title_fontsize=10)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor((0, 0, 0, edge_alpha))

    # Tidy layout
    fig.tight_layout()

    # Save or show
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
    return fig, ax


def parse_arguments():
    """Parse command line arguments for part and selected_metric."""
    parser = argparse.ArgumentParser(
        description="Generate metric comparison plots by fold",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reporting.py                    # Use default values (part=2, metric=F1@best)
  python reporting.py --part 1          # Use part=1 with default metric
  python reporting.py --metric mAP50-95 # Use metric=mAP50-95 with default part
  python reporting.py --part 3 --metric mean_IoU@TP  # Use both arguments
        """
    )
    
    parser.add_argument(
        "--part", 
        type=int, 
        default=2,
        choices=[1, 2, 3, 4, 5, 6],
        help="Part number to determine which runs to plot (default: 2)"
    )
    
    parser.add_argument(
        "--metric", 
        type=str, 
        default="F1@best",
        choices=["F1@best", "mean_IoU@TP", "mAP50-95", "best_epoch", "best_conf"],
        help="Metric to plot (default: F1@best)"
    )
    
    return parser.parse_args()


def get_runs_for_part(part: int) -> list:
    """Get the selected runs for a given part number."""
    if part == 1:
        RUN_NAME_TO_VIZ_NAME["D_light_3"] = "Light Aug"
        return ["A_no_aug", "B_light_aug", "C_heavy_aug"]
    elif part == 2:
        RUN_NAME_TO_VIZ_NAME["D_light_3"] = "Light Aug (2x)"
        return ["B_light_aug", "D_light_3", "E_light_4", "F_light_8"]
    elif part == 3:
        RUN_NAME_TO_VIZ_NAME["D_light_3"] = "Flip Only Aug"
        return ["G_no_geo_aug", "D_light_3", "H_scale_shear_rotate_geo_aug", "I_all_geo_aug"]
    elif part == 4:
        RUN_NAME_TO_VIZ_NAME["D_light_3"] = "YOLOv8 Nano"
        RUN_NAME_TO_VIZ_NAME["v12_nano"] = "YOLOv12 Nano"
        return ["D_light_3", "v8", "v12_nano", "FasterRCNN"]
    elif part == 5:
        RUN_NAME_TO_VIZ_NAME["v12_nano_440"] = "YOLOv12 Nano 440"
        return ["v12_nano", "v12_nano", "v12_nano_768", "v12_nano_512"]
    elif part == 6:
        RUN_NAME_TO_VIZ_NAME["v12_nano"] = "Baseline"
        return ["v12_nano", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]
    else:
        raise ValueError(f"Invalid part number: {part}. Must be 1-5.")


def get_custom_colors() -> dict:
    """Get the custom color mapping for runs."""
    return 
    custom_colors = {
        "A_no_aug": "royalblue",
        "B_light_aug": "darkorange",
        "C_heavy_aug": "mediumseagreen",
        "D_light_3": "indianred",
        "E_light_4": "orchid",
        "F_light_8": "teal",

        "G_no_geo_aug": "royalblue",
        "H_scale_shear_rotate_geo_aug": "darkorange",
        "I_all_geo_aug": "mediumseagreen",

        "v8": "royalblue",
        "v12_nano": "mediumseagreen",
        "FasterRCNN": "slateblue",

        "v12_nano_768": "indianred",
        "v12_nano_512": "orchid",
    }


# ---------- Main execution ----------
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Get runs and metric based on arguments
    part = args.part
    selected_metric = args.metric
    selected_runs = get_runs_for_part(part)
    
    # Get custom colors
    custom_colors = get_custom_colors()

    # Generate the plot
    plot_metric_comparison_by_fold(
        excel_path="results/results_compact.xlsx",
        sheet_name="results",
        runs=selected_runs,
        metric=selected_metric,
        save_path="results/f1_comparison_by_fold_part{}_{}.png".format(part, selected_metric),
        run_colors=custom_colors,
        splits_root="data/splits_lomo_test_val_only",  # Path to LOMO fold directories
    )