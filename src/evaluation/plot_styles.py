"""IEEE Transactions publication styling utility for Matplotlib.

Configures figure typography, dimensions, font sizes, line styles, and color palettes
to conform to IEEE editorial standards for single-column (3.5 in) and double-column
(7.16 in) figures with vector font embedding (Type 42 / TrueType).
"""

import logging
from typing import Dict, Tuple

import matplotlib as mpl

logger = logging.getLogger(__name__)

# Standard IEEE Transactions dimensions in inches
IEEE_SINGLE_COL_WIDTH_INCHES: float = 3.50
IEEE_DOUBLE_COL_WIDTH_INCHES: float = 7.16

# Colorblind-safe, high-contrast palette compliant with print and digital IEEE standards
IEEE_PALETTE: Dict[str, str] = {
    "Ziegler-Nichols": "#D95F02",       # Vermilion / Red-Orange
    "Cohen-Coon": "#7570B3",            # Purple-Slate
    "Skogestad SIMC": "#1B9E77",        # Dark Teal / Green
    "Optimal ITAE (Ms<=1.6)": "#1F78B4",# Classic Navy Blue
    "Optimal ITAE ($M_s \\leq 1.6$)": "#1F78B4",
    "Optimal ITAE ($M_s \\le 1.6$)": "#1F78B4",
    "Optimal ITAE": "#1F78B4",
    "Setpoint": "#222222",              # Near Black
    "Constraint": "#E41A1C",            # Danger Red
    "Pareto": "#252525",                # Charcoal
}

# Distinguishable linestyles for grayscale readability
IEEE_LINESTYLES: Dict[str, str] = {
    "Ziegler-Nichols": "--",
    "Cohen-Coon": "-.",
    "Skogestad SIMC": "-",
    "Optimal ITAE (Ms<=1.6)": "-",
    "Optimal ITAE ($M_s \\leq 1.6$)": "-",
    "Optimal ITAE ($M_s \\le 1.6$)": "-",
    "Optimal ITAE": "-",
    "Setpoint": ":",
    "Constraint": ":",
}

# Markers for scatter / discrete points
IEEE_MARKERS: Dict[str, str] = {
    "Ziegler-Nichols": "x",
    "Cohen-Coon": "^",
    "Skogestad SIMC": "s",
    "Optimal ITAE (Ms<=1.6)": "o",
    "Optimal ITAE ($M_s \\leq 1.6$)": "o",
    "Optimal ITAE ($M_s \\le 1.6$)": "o",
    "Optimal ITAE": "o",
}


def setup_ieee_style(single_column: bool = False) -> None:
    """Configures global Matplotlib rcParams for IEEE Transactions publication quality.

    Ensures fonts are embedded as TrueType (Type 42) for IEEE Xplore compliance,
    serif typography is matched to Times / Computer Modern, and font sizes are legible.

    Args:
        single_column: If True, uses slightly more compact font sizes for 3.5-inch figures.
    """
    logger.info("Applying IEEE publication plot style (single_column=%s)", single_column)

    base_font_size = 8.0 if single_column else 8.5
    label_font_size = 8.5 if single_column else 9.0
    title_font_size = 9.0 if single_column else 9.5

    params = {
        # Backend and TrueType embedding
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Serif typography matching IEEE LaTeX documents
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
            "Computer Modern Roman",
            "serif",
        ],
        "font.size": base_font_size,
        "axes.titlesize": title_font_size,
        "axes.labelsize": label_font_size,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.linewidth": 0.8,
        # Ticks styling
        "xtick.labelsize": base_font_size,
        "ytick.labelsize": base_font_size,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.top": True,
        "ytick.right": True,
        # Legends
        "legend.fontsize": base_font_size - 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#CCCCCC",
        "legend.fancybox": False,
        # Grid lines
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "grid.color": "#AAAAAA",
        # Lines and figures
        "lines.linewidth": 1.2,
        "lines.markersize": 5.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }

    mpl.rcParams.update(params)


def get_figure_dimensions(
    columns: int = 1,
    aspect_ratio: float = 0.75,
    height_override: float = 0.0,
) -> Tuple[float, float]:
    """Computes exact width and height in inches for IEEEtran layouts.

    Args:
        columns: 1 for single column (3.5 in) or 2 for double column (7.16 in).
        aspect_ratio: Height-to-width ratio (default 0.75 for 4:3 ratio).
        height_override: If > 0, explicitly forces this height in inches.

    Returns:
        Tuple of (width_in_inches, height_in_inches).
    """
    width = IEEE_SINGLE_COL_WIDTH_INCHES if columns == 1 else IEEE_DOUBLE_COL_WIDTH_INCHES
    height = height_override if height_override > 0.0 else width * aspect_ratio
    return (float(width), float(height))
