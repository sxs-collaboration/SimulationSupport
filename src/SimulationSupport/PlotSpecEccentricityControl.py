#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence, Union

import click
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.patheffects as path_effects
from adjustText import adjust_text
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib.patches import Ellipse, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

logger = logging.getLogger(__name__)

# Style helpers

# Line styles and markers are cycled over however many run types the user supplies.
# Styles are assigned in the order run types appear in the data.
_LINESTYLE_CYCLE  = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]
_MARKER_CYCLE     = ["o", "D", "s", "^", "v", "P", "X", "*"]
_MARKERSIZE_CYCLE = [5,   4,   5,   5,   5,   5,   5,   7  ]

def build_run_type_styles(run_types: Sequence) -> dict:
    """
    Build per-run-type style dicts from whatever run type labels the user
    has assigned in their RunType column.

    Styles are assigned in the order the run types appear. User controls all naming.

    Args:
        run_types: iterable of user supplied run type label strings.
 
    Returns:
        Dict of dicts keyed by run type, each containing keys
        'linestyle', 'marker', 'markersize', and 'zorder'.
    """
    unique = list(dict.fromkeys(run_types))
    return {
        rt: {
            "linestyle":  _LINESTYLE_CYCLE[i % len(_LINESTYLE_CYCLE)],
            "marker":     _MARKER_CYCLE[i % len(_MARKER_CYCLE)],
            "markersize": _MARKERSIZE_CYCLE[i % len(_MARKERSIZE_CYCLE)],
            "zorder":     i + 1,
        }
        for i, rt in enumerate(unique)
    }

def build_case_color_map(cases: Sequence) -> dict:
    """Assign a distinct colour to each unique case, using the tab10 palette.

    Args:
        cases: iterable of case identifiers.

    Returns:
        Dict mapping each case identifier to a matplotlib colour string.
    """
    unique = list(dict.fromkeys(cases))
    palette = plt.cm.tab10.colors
    return {c: palette[i % len(palette)] for i, c in enumerate(unique)}


# Formatting helpers
def sci_offset_formatter() -> ScalarFormatter:
    """
    Return a ScalarFormatter that always uses scientific offset notation.
    """
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))
    return fmt

def with_alpha(color: str, alpha: float):
    """
    Return the color as an RGBA tuple with the given opacity.

    Args:
        color: any matplotlib color string.
        alpha: opacity in [0, 1].

    Returns:
        RGBA tuple.
    """
    r, g, b, _ = mcolors.to_rgba(color)
    return (r, g, b, alpha)

def save_pdf(name: str, width: float = 3.4, fig=None):
    """
    Save a figure as a PDF with fixed width of 3.4 and auto-increment the filename so old plots do not get
    overwritten. Call after plt.show()

    Args:
        name: base filename.
        width: target width in inches (defaults to 3.4 for PRD column width).
        fig: figure to save (defaults to the current figure).
    """
    i = 1
    filename = f"{name}_{i:02d}.pdf"
    while os.path.exists(filename):
        i += 1
        filename = f"{name}_{i:02d}.pdf"

    if fig is None:
        fig = plt.gcf()
    # Get current figure and resize
    orig_width, orig_height = fig.get_size_inches()
    # Keep aspect ratio by computing proportional height
    new_height = orig_height * (width / orig_width)
    # Resize to PRD width
    fig.set_size_inches(width, new_height)
    # Save
    fig.savefig(filename, format="pdf", bbox_inches="tight")
    print(f"Saved figure as {filename}")

# HDF5 parsing helpers
def extract_params_from_lines(lines: list, include_massratio: bool = True):
    """
    Extract MassRatio, Omega0, adot0, and D0 from their respective Params.input lines.

    Args:
        lines: list of strings from Params.input file.
        include_massratio (bool): condition to determine whether or not to extract MassRatio.

    Returns:
        Tuple (massratio, omega, adot, d0) of floats, or (None, …) on error.
    """
    try:
        # Find relevant lines for each parameter
        mass_ratio_line = next((l for l in lines if "$MassRatio" in l), None)
        omega_line      = next((l for l in lines if "$Omega0" in l), None)
        adot_line       = next((l for l in lines if "$adot0" in l),  None)
        d0_line         = next((l for l in lines if "$D0" in l),     None)

        # Raise error if any are missing
        if None in [mass_ratio_line, omega_line, adot_line, d0_line]:
            raise ValueError("Required parameters not found")

        # Extract numeric values from the strings
        massratio = float(mass_ratio_line.split("=")[1].strip().rstrip(";"))
        omega     = float(omega_line.split("=")[1].strip().rstrip(";"))
        adot      = float(adot_line.split("=")[1].strip().rstrip(";"))
        d0        = float(d0_line.split("=")[1].strip().rstrip(";"))

        return massratio, omega, adot, d0

    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return None, None, None, None

def filter_param_lines(content: str) -> list:
    """
    Return only the lines in the input file that contain the relevant parameters.

    Args:
        content: text of a Params.input file.

    Returns:
        List of matching lines.
    """
    return [
        line for line in content.splitlines()
        if any(key in line for key in ("$MassRatio", "$Omega0", "$adot0", "$D0"))
    ]


def read_dataset(hdf, path: str) -> str:
    """
    Read and decode the string content of the dataset from the HDF5 file.

    Args:
        hdf (h5py.File): open h5py.File object.
        path (str): dataset path inside the file.

    Returns:
        Decoded UTF-8 string.
    """
    return hdf[path][()].decode("utf-8")

# Group by simulation name and eccentricity level - for example, EccRedTest**2025*** and Ecc*
def group_by_sim_and_ecc(paths: list, suffix: str) -> dict:
    """
    Group HDF5 dataset paths by simulation name and eccentricity level.
    Only include paths ending with the specified suffix (eg. "Params.input")

    Args:
        paths: all dataset paths in the HDF5 file.
        suffix (str): file suffix to match, e.g. 'Params.input'.

    Returns:
        Dict mapping simulation name to a list of (ecc_level, path) tuples.
    """
    # Create a dictionary that maps each sim to a list of (ecc, path) pairs
    grouped = defaultdict(list)
    # Loop through each path and keep only the ones that end with the given suffix
    for path in paths:
        if path.endswith(suffix):
            # Split path string into components and only process paths with at least 3 components
            parts = path.split("/")
            # Expect format SimName/EccX/...
            if len(parts) >= 3 and re.match(r"Ecc\d+", parts[1]):
                # Ensure eccentricity directory exists
                sim = parts[0]
                try:
                    ecc = int(parts[1][3:]) # strip "Ecc" and convert to int
                    grouped[sim].append((ecc, path))
                except ValueError:
                    pass
    return grouped


def process_h5_file(file_path: str) -> pd.DataFrame:
    """
    Extract orbital parameter (Omega0, adot0, separation, mass ratio, eccentricity) values
    from all Ecc folders across all simulations in an HDF5 file.

    Args:
        file_path: path to the HDF5 file.

    Returns:
        DataFrame with columns [Sim, MassRatio, EccLevel, Omega0, Adot0,
        Initial Separation, Eccentricity] and one row per eccentricity level.
    """

    data_all_iterations = []

    with h5py.File(file_path, "r") as hdf:
        all_paths = []

        # Recursively collect all dataset paths in the file
        def collect_paths(name, obj):
            if isinstance(obj, h5py.Dataset):
                all_paths.append(name)

        hdf.visititems(collect_paths)

        # Group dataset by simulation and eccentricity level
        grouped_params = group_by_sim_and_ecc(all_paths, "Params.input")

        for sim, entries in grouped_params.items():
            # Sort entries in order of ascending eccentricity level (Ecc0, Ecc1, ...)
            entries.sort()

            # Iterate through all eccentricity levels and extract data
            for ecc, path in entries:
                try:
                    # Read Params.input content and extract relevaznt parameters
                    content = read_dataset(hdf, path)
                    lines = filter_param_lines(content)
                    massratio, omega, adot, d0 = extract_params_from_lines(lines)

                    ecc_file = (
                        f"{sim}/Ecc{ecc}/Ev/JoinedForEcc/Fit_F2cos2_SS.dat"
                    )
                    if ecc_file in all_paths:
                        ecc_lines = read_dataset(hdf, ecc_file).splitlines()
                        ecc_value = float(
                            ecc_lines[-1].split()[-1].strip().rstrip(";")
                        )
                    else:
                        ecc_value = None

                    # Store data in dictionary with one row per iteration
                    data_all_iterations.append({
                        "Sim":               sim,
                        "MassRatio":         massratio,
                        "EccLevel":          ecc,
                        "Omega0":            omega,
                        "Adot0":             adot,
                        "Initial Separation": d0,
                        "Eccentricity":      ecc_value,
                    })

                except Exception as e:
                    print(
                        f"Error in Params.input for {sim} in {file_path}: {e}"
                    )

    return pd.DataFrame(data_all_iterations)

# JSON parsing helpers
def find_metadata(
    paths: list,
    chosen_lev: str,
    suffix: str = "metadata.json",
    root: Path = None,
) -> dict:
    """
    Find metadata.json files at a chosen Lev directory for each sim folder.

    Args:
        paths: all relative file paths.
        chosen_lev: which Lev directory to select, e.g. 'Lev3' or 'Lev4'.
                    Must be provided - no default assumed.
        suffix: filename to match.
        root: absolute root path to prepend.

    Returns:
        Dict mapping sim name to a list of absolute Paths.
    """
    grouped = defaultdict(list)
    for path in paths:
        if path.endswith(suffix):
            parts = path.split("/")
            if len(parts) >= 3 and parts[1] == chosen_lev:
                sim = parts[0]
                grouped[sim].append(root / Path(path))
    return grouped

def process_json_files(metadata_paths: dict) -> pd.DataFrame:
    """
    Read each metadata.json file and extract parameters into a DataFrame.

    Args:
        metadata_paths: dict mapping sim name to a list of absolute Paths.

    Returns:
        DataFrame with orbital parameters extracted from each JSON file.
    """

    data_all = []
    for sim in sorted(metadata_paths.keys(), key=lambda s: int(s)):
        for path in metadata_paths[sim]:
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                spin1 = data.get("initial_dimensionless_spin1", [None, None, None])
                spin2 = data.get("initial_dimensionless_spin2", [None, None, None])

                mass1 = data.get("initial_mass1", None)
                mass2 = data.get("initial_mass2", 1)

                data_all.append({
                    "Sim":               data.get("simulation_name", sim),
                    "MassRatio":         mass1 / mass2 if mass1 is not None else None,
                    "Initial Separation": data.get("initial_separation", None),
                    "Omega0":            data.get("initial_orbital_frequency", None),
                    "Adot0":             data.get("initial_adot", None),
                    "S1x": spin1[0], "S1y": spin1[1], "S1z": spin1[2],
                    "S2x": spin2[0], "S2y": spin2[1], "S2z": spin2[2],
                    "Eccentricity":      data.get("reference_eccentricity", None),
                })
            except Exception as e:
                print(f"Error reading {path}: {e}")

    return pd.DataFrame(data_all)

# Tolerance ellipse helpers
def choose_inset_anchor(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    w: float = 0.35,
    h: float = 0.5,
    pad_x: float = 0.02,
    pad_y: float = 0.02,
):
    """
    Choose an inset anchor (x0, y0, w, h) in Axes coords that overlaps the
    fewest data points.

    Args:
        ax: parent Axes object.
        x: data x-coordinates of existing points.
        y: data y-coordinates of existing points.
        w: inset width in Axes fraction.
        h: inset height in Axes fraction.
        pad_x: horizontal padding from Axes edge.
        pad_y: vertical padding from Axes edge.

    Returns:
        Tuple (x0, y0, w, h) in Axes fraction coordinates.
    """
    pts = ax.transAxes.inverted().transform(
        ax.transData.transform(np.c_[x, y])
    )
    px, py = pts[:, 0], pts[:, 1]

    candidates = [
        (pad_x, pad_y),
        (1 - w - pad_x, pad_y),
        (pad_x, 1 - h - pad_y),
        (1 - w - pad_x, 1 - h - pad_y),
    ]
    best = candidates[0]
    best_score = None
    for x0, y0 in candidates:
        in_box = (
            (px >= x0) & (px <= x0 + w)
            & (py >= y0) & (py <= y0 + h)
        )
        score = in_box.sum()
        if best_score is None or score < best_score:
            best_score = score
            best = (x0, y0)
    return best[0], best[1], w, h

def add_tolerance_ellipse_fixed(
    ax,
    df_case: pd.DataFrame,
    case_id: str,
    run_type: str,
    dOmega: float,
    dAdot: float,
    case_col: str = "CaseLetter",
    facecolor: str = "hotpink",
    edgecolor: str = "hotpink",
    alpha: float = 0.10,
    lw: float = 0.8,
    zorder: Optional[int] = None,
    label: Optional[str] = None,
):
    """
    Draw a single tolerance ellipse centered on the final iteration point.

    Args:
        ax: Axes to draw on.
        df_case: per-iteration DataFrame for the case.
        case_id: case letter, e.g. 'A'.
        run_type: run type label set by the user in the RunType column.
        dOmega: half-width of the ellipse in Omega0 units.
        dAdot: half-height of the ellipse in Adot0 units.
        case_col: column name that holds the case letter.
        facecolor: fill colour.
        edgecolor: edge colour.
        alpha: fill opacity.
        lw: edge line width.
        zorder: drawing order.
        label: legend label.

    Returns:
        The Ellipse patch, or None if no data is found.
    """
    d = df_case.copy()
    d[case_col] = d[case_col].astype(str)
    d = d[(d[case_col] == str(case_id)) & (d["RunType"] == run_type)]
    if d.empty:
        return None

    row_final = d.sort_values("EccLevel").iloc[-1]
    x_final = float(row_final["Omega0"])
    y_final = float(row_final["Adot0"])

    fc = mcolors.to_rgba(facecolor, 0.12)
    ec = mcolors.to_rgba(edgecolor, 0.62)

    ell = Ellipse(
        (x_final, y_final),
        width=2 * dOmega,
        height=2 * dAdot,
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
        label=label,
    )
    ax.add_patch(ell)
    return ell


def max_unapplied_deltas(
    df_case: pd.DataFrame,
    df_next: pd.DataFrame,
    case_id: str,
    run_types: list,
    case_col: str = "CaseLetter",
):
    """
    Compute the maximum unapplied correction size across run types.

    Args:
        df_case: per-iteration DataFrame.
        df_next: DataFrame of next-step parameters with columns
                 [Case, RunType, Omega0_new, Adot0_new].
        case_id: case letter.
        run_types: list of run type strings to consider.
        case_col: column name that holds the case letter.

    Returns:
        Tuple (dOmega_max, dAdot_max), or None if no data is found.
    """
    d = df_case.copy()
    d[case_col] = d[case_col].astype(str)
    out = []

    for rt in run_types:
        dd = d[(d[case_col] == str(case_id)) & (d["RunType"] == rt)]
        if dd.empty:
            continue
        row_final = dd.sort_values("EccLevel").iloc[-1]
        x_final = float(row_final["Omega0"])
        y_final = float(row_final["Adot0"])

        hit = df_next[
            (df_next["Case"].astype(str) == str(case_id))
            & (df_next["RunType"].astype(str) == rt)
        ]
        if hit.empty:
            continue

        x_new = float(hit.iloc[0]["Omega0_new"])
        y_new = float(hit.iloc[0]["Adot0_new"])
        out.append((abs(x_new - x_final), abs(y_new - y_final)))

    if not out:
        return None
    
    dOmega_max = max(v[0] for v in out)
    dAdot_max = max(v[1] for v in out)
    return dOmega_max, dAdot_max


def add_concentric_tolerance_from_next_step(
    ax,
    df_case: pd.DataFrame,
    df_next: pd.DataFrame,
    case_id: str,
    run_type: str,
    case_col: str = "CaseLetter",
    base_color: str = "hotpink",
    multipliers: tuple = (3, 2, 1),
    fill_alphas: tuple = (0.05, 0.08, 0.12),
    edge_alphas: tuple = (0.20, 0.35, 0.55),
    lw: tuple = (1.0, 1.1, 1.2),
    zorder: int = 1,
    min_frac: Optional[float] = None,
) -> list:
    """
    Draw concentric tolerance ellipses centered at the final executed point.
    The ellipse sizes are set by the unapplied correction |next - final| in (Omega, Adot).

    Args:
        ax: Axes to draw on.
        df_case: per-iteration DataFrame.
        df_next: DataFrame of next-step parameters.
        case_id: case letter.
        run_type: run type string.
        case_col: column name that holds the case letter.
        base_color: colour for all ellipses.
        multipliers: scale factors for the three ellipses.
        fill_alphas: fill opacities for each ellipse.
        edge_alphas: edge opacities for each ellipse.
        lw: line widths for each ellipse.
        zorder: drawing order (placed behind lines and markers).
        min_frac: if set, enforce a minimum ellipse size as a fraction of the current Axes range.

    Returns:
        List of Ellipse patches that were added.
    """
    d = df_case.copy()
    d[case_col] = d[case_col].astype(str)
    d = d[(d[case_col] == str(case_id)) & (d["RunType"] == run_type)]
    if d.empty:
        return []

    row_final = d.sort_values("EccLevel").iloc[-1]
    x_final = float(row_final["Omega0"])
    y_final = float(row_final["Adot0"])

    df_next2 = df_next.copy()
    df_next2["Case"] = df_next2["Case"].astype(str)
    df_next2["RunType"] = df_next2["RunType"].astype(str)
    hit = df_next2[
        (df_next2["Case"] == str(case_id))
        & (df_next2["RunType"] == str(run_type))
    ]
    if hit.empty:
        return []

    x_new = float(hit.iloc[0]["Omega0_new"])
    y_new = float(hit.iloc[0]["Adot0_new"])
    dOmega = abs(x_new - x_final)
    dAdot  = abs(y_new - y_final)

    if min_frac is not None:
        xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        if xspan > 0:
            dOmega = max(dOmega, min_frac * xspan)
        if yspan > 0:
            dAdot = max(dAdot, min_frac * yspan)

    ells = []
    for m, fa, ea, lw_i in zip(multipliers, fill_alphas, edge_alphas, lw):
        ell = Ellipse(
            (x_final, y_final),
            width=2 * (m * dOmega),
            height=2 * (m * dAdot),
            facecolor=with_alpha(base_color, fa),
            edgecolor=with_alpha(base_color, ea),
            linewidth=lw_i,
            zorder=zorder,
        )
        ax.add_patch(ell)
        ells.append(ell)

    return ells

# Plotting functions
def plot_spec_eccentricity_control(
    df_all: pd.DataFrame,
    sim_name: str,
    ax=None,
    color: str = "black",
    label: Optional[str] = None,
    linestyle: str = "-",
    linewidth: Optional[float] = None,
    run_type: Optional[str] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    zorder: int = 10,
    adot_mode: str = "raw",
):
    """
    Plot the trajectory through (Omega0, Adot0) space for a single
    simulation across eccentricity reduction iterations. Optionally overlay
    eccentricity contours. Can plot Adot0 in its true format, absolute value,
    or log absolute value.

    Key differences from PlotEccentricityControl.py:
      1. Handles cases with fewer than 3 iterations (skips contour map,
        still plots points and path) because many simulations need only 2 iterations.
      2. Draws directional arrows along the trajectory.
      3. Annotates each point with its iteration number.

    \f
    Arguments:
        df_all: DataFrame containing per-iteration simulation data.
                Must include columns [Sim, Omega0, Adot0, EccLevel, Eccentricity].
        sim_name: simulation name to filter and plot.
        ax: Axis to plot on; If None, creates a new figure.
        color: color of line/points for the simulation.
        label: legend label.
        linestyle: matplotlib line style string.
        linewidth: line width.
        run_type: run type label for this simulation that was set by the user in the RunType column.
        marker: matplotlib marker string.
        markersize: marker size in points.
        zorder: base drawing order; arrows and annotations are drawn above.
        adot_mode: how to display Adot0 — 'raw' (default), 'abs' (absolute
                   value), or 'logabs' (log10 of absolute value).

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot
        ax (matplotlib.axes.Axes): Axis
    """
    # Filter simulation data to only include the specified simulation
    sim_data = df_all[df_all["Sim"] == sim_name].copy()
    # Ensure iterations are plotted in order
    sim_data.sort_values("EccLevel", inplace=True)

    if sim_data.empty:
        print(f"No data found for simulation {sim_name}")
        return ax

    # Transform Adot0 according to requested mode
    if adot_mode == "abs":
        sim_data["Adot0_plot"] = np.abs(sim_data["Adot0"].values.astype(float))
    elif adot_mode == "logabs":
        safe_abs = np.abs(sim_data["Adot0"].values.astype(float))
        sim_data["Adot0_plot"] = np.where(safe_abs > 0, np.log10(safe_abs), np.nan)
    else:
        sim_data["Adot0_plot"] = sim_data["Adot0"]

    xvals = sim_data["Omega0"].to_numpy()
    yvals = sim_data["Adot0_plot"].to_numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    marker_alpha = 0.5

    # Plot trajectory
    ax.plot(
        xvals, yvals,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markerfacecolor=with_alpha(color, marker_alpha),
        markeredgecolor=color,
        markeredgewidth=0.4,
        label=label,
        zorder=zorder + 1,
    )

    # Directional arrows along the trajectory
    for (x0, y0), (x1, y1) in zip(
        zip(xvals[:-1], yvals[:-1]), zip(xvals[1:], yvals[1:])
    ):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0,
            edgecolor=color,
            facecolor=color,
            zorder=zorder + 2,
        ))

    # Iteration number annotations
    for it, x, y in zip(sim_data["EccLevel"].astype(int), xvals, yvals):
        ax.annotate(
            f"{it}",
            (x, y),
            xycoords="data",
            ha="center",
            va="center",
            fontsize=4,
            color="black",
            zorder=zorder + 200,
            clip_on=True,
        )

    ax.relim()
    ax.autoscale_view()
    return ax

# Option 1: Eccentricity vs iteration number
def plot_eccentricity_vs_iteration(
    df_all,
    case_col: str,
    output_path: str = "eccentricity_vs_iteration.pdf",
    run_type_legend_labels=None,
    case_to_color=None,
    case_display_labels=None,
    y_min: float = 1e-4,
):
    """
    Plot eccentricity versus iteration number for all simulations.
 
    Each line represents one simulation. Line style and marker encode the
    run type; colour encodes the case. No run type naming convention is
    assumed — the user controls all labeling in the RunType column.
 
    \f
    Arguments:
        df_all: per-iteration DataFrame. Must contain columns
                [Sim, EccLevel, Eccentricity, RunType] plus the column
                named by case_col. The RunType column must be populated
                by the user before calling this function.
        case_col: column that identifies the physical case for each row; used to assign colours.
        output_path: path for the saved PDF.
        run_type_legend_labels: optional dict mapping RunType values to custom display labels in the legend, e.g.
                  {'baseline': 'PN (2025)', 'GPR': 'GPR (2025)'}. If None, raw RunType values are used.
        case_to_color: optional dict mapping case identifiers to matplotlib
                  colour strings. If None, colours are assigned automatically from tab10.
        case_display_labels: optional dict mapping case identifiers to display labels for the case legend, e.g.
                  {'111': 'A', '271': 'B'}. If None, raw identifiers are used.
        y_min: lower y-axis limit (default 1e-4).
 
    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot
        ax (matplotlib.axes.Axes): Axis
    """
    run_types_present = list(dict.fromkeys(df_all["RunType"].dropna().values))
    rt_styles = build_run_type_styles(run_types_present)
 
    if case_to_color is None:
        all_cases = list(dict.fromkeys(df_all[case_col].dropna().values))
        case_to_color = build_case_color_map(all_cases)
 
    legend_label   = run_type_legend_labels or {}
    display_labels = case_display_labels    or {}
 
    fig, ax = plt.subplots(figsize=(3.4, 3))
 
    for sim_name, df_sim in df_all.sort_values("EccLevel").groupby("Sim"):
        case  = df_sim[case_col].iloc[0]
        rt    = df_sim["RunType"].iloc[0]
        color = case_to_color.get(case, "black")
        s     = rt_styles.get(rt, rt_styles[next(iter(rt_styles))])
 
        ax.plot(
            df_sim["EccLevel"],
            df_sim["Eccentricity"],
            marker=s["marker"],
            markersize=s["markersize"],
            zorder=s["zorder"],
            linestyle=s["linestyle"],
            linewidth=0.65,
            color=color,
            label="_nolegend_",
            markeredgecolor="white",
            markeredgewidth=0.05,
        )
 
    ax.set_yscale("log")
    ax.set_ylim(y_min, None)
    ax.set_xlabel("Iteration number", fontsize=8.5)
    ax.set_ylabel("Eccentricity",     fontsize=8.5)
    ax.set_title("Eccentricity reduction across iterations", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=7)
 
    # Run type legend
    run_type_handles = [
        mlines.Line2D(
            [], [], color="black",
            linestyle=rt_styles[rt]["linestyle"],
            marker=rt_styles[rt]["marker"],
            markersize=rt_styles[rt]["markersize"],
            linewidth=0.65,
            label=legend_label.get(rt, rt),
        )
        for rt in run_types_present
    ]
    type_legend = ax.legend(
        handles=run_type_handles,
        title="Initial Guess Type",
        loc="upper right",
        bbox_to_anchor=(0.85, 1),
        title_fontsize=7, fontsize=6,
        handlelength=4, handletextpad=0.6,
        frameon=False,
    )
    ax.add_artist(type_legend)
 
    # Case color legend
    cases_in_data = list(dict.fromkeys(df_all[case_col].dropna().values))
    cases_sorted  = sorted(cases_in_data, key=lambda c: display_labels.get(c, str(c)))
    case_handles  = [
        mlines.Line2D(
            [], [], color=case_to_color.get(c, "black"),
            marker="s", markersize=3, linestyle="None",
            label=display_labels.get(c, str(c)),
        )
        for c in cases_sorted
    ]
    ax.legend(
        handles=case_handles, title="Case",
        loc="upper right", bbox_to_anchor=(1, 1),
        title_fontsize=7, fontsize=6, handletextpad=0, frameon=False,
    )
 
    fig.tight_layout()
    fig.savefig(output_path)
    return fig


# Option 2: trajectory plot — one case per panel
def plot_trajectories(
    df_all: pd.DataFrame,
    case_col: str,
    cases_to_plot: Sequence,
    drop_by_panel: dict,
    new_params: pd.DataFrame,
    output_path: str = "trajectories.pdf",
    inset_tail_counts: Optional[dict] = None,
    run_type_legend_labels: Optional[dict] = None,
    case_to_color: Optional[dict] = None,
):
    """
    Plot (Omega0, Adot0) trajectories with zoom insets, making one panel per case.

    Arguments:
        df_all: full per-iteration DataFrame. Must contain columns
                [Sim, Omega0, Adot0, EccLevel, Eccentricity, RunType]
                plus the column named. The RunType column must be populated
                before calling the function.
        case_col: column in df_all that identifies the physical case,
                  e.g. 'Case' or 'SimID'. Each unique value becomes one panel.
        cases_to_plot: ordered sequence of values from case_col to display; one panel each.
        drop_by_panel: dict mapping a case value to a set of other case values
                       to exclude from that panel. Pass an empty dict to show
                       all data in every panel.
        new_params: DataFrame of next-step parameters for tolerance ellipses,
                    with columns [<case_col>, RunType, Omega0_new, Adot0_new].
                    Pass an empty DataFrame to skip the ellipses.
        output_path: path for the saved PDF.
        inset_tail_counts: optional dict controlling how many tail iterations
                    each run type contributes to the inset zoom region, e.g.
                    {'PN': 1, 'GPR': None} where None means all
                    iterations. Defaults to all iterations for every run type.
        run_type_legend_labels: optional dict mapping RunType values to custom
                    display labels in the legend, e.g.
                    {'PN': 'PN (2025)', 'GPR run': 'GPR (2025)'}.
                    If None, the raw RunType values are used as labels.
        case_to_color: optional dict mapping case identifiers to matplotlib
                    colour strings, e.g. {'caseA': 'red', 'caseB': 'blue'}.
                    If None, colours are assigned automatically from tab10.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot
        ax (matplotlib.axes.Axes): Axis
    """
    n = len(cases_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(n * 8 / 3, 3), dpi=300, sharey=False)
    fig.suptitle("Eccentricity Trajectory through Parameter Space", fontsize=10)
    if n == 1:
        axes = [axes]

    # Build style maps
    run_types_present = list(dict.fromkeys(df_all["RunType"].dropna().values))
    rt_styles = build_run_type_styles(run_types_present)

    # Color map — use user supplied map or build one automatically
    if case_to_color is None:
        all_cases = list(dict.fromkeys(df_all[case_col].dropna().values))
        case_to_color = build_case_color_map(all_cases)

        # Track which cases actually appear across all panels for the legend
        cases_used: set = set()
    
    for ax, case_id in zip(axes, cases_to_plot):
        drops = drop_by_panel.get(case_id, set())
        df_case = df_all[(df_all[case_col] == case_id) & (~df_all[case_col].isin(drops))].copy()
        cases_used |= set(df_case[case_col].unique())
    
        for sim_name in sorted(df_case["Sim"].unique()):
            rt = df_case.loc[df_case["Sim"] == sim_name, "RunType"].iloc[0]
            color = case_to_color.get(case_id, "black")
            s = rt_styles.get(rt, rt_styles[next(iter(rt_styles))])

            plot_spec_eccentricity_control(
                df_case, sim_name,
                ax=ax,
                color=color,
                adot_mode="raw",
                linestyle=s["linestyle"],
                run_type=rt,
                marker=s["marker"],
                linewidth=0.6,
                markersize=s["markersize"],
                zorder=s["zorder"],
                label="_nolegend_",
            )

        # Main axis limits
        xmin, xmax = df_case["Omega0"].min(), df_case["Omega0"].max()
        ymin, ymax = df_case["Adot0"].min(),  df_case["Adot0"].max()

        dx = 0.05 * (xmax - xmin) or 1e-6
        dy = 0.05 * (ymax - ymin) or 1e-6

        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)

        # Build inset zoom region: per-run-type tail counts let the user
        # control how many iterations each run type contributes to the zoom.
        # None means all iterations; a positive int means that many tail rows.
        zoom_parts = []
        for rt in run_types_present:
            df_rt = df_case[df_case["RunType"] == rt]
            if df_rt.empty:
                continue
            tail_n = (inset_tail_counts or {}).get(rt, None)
            if tail_n is not None:
                df_rt = df_rt.sort_values("EccLevel").groupby("Sim").tail(tail_n)
            zoom_parts.append(df_rt)
        df_zoom = pd.concat(zoom_parts, ignore_index=True) if zoom_parts else pd.DataFrame()

        if not df_zoom.empty:
            x_all = df_case["Omega0"].to_numpy()
            y_all = df_case["Adot0"].to_numpy()
            x0, y0, w, h = choose_inset_anchor(ax, x_all, y_all, w=0.55, h=0.5, pad_x=0.02, pad_y=0.02)
            x_offset= 0.08
            y_offset = 0.09   # (axes fraction; can tweak)
            x0 = min(x0 + x_offset, 1 - w - 0.01)
            y0 = min(y0 + y_offset, 1 - h - 0.01)

            axins = inset_axes(
                ax, width="100%", height="100%",
                bbox_to_anchor=(x0, y0, w, h),
                bbox_transform=ax.transAxes,
                loc="lower left",
                borderpad=0.0,
            )
            axins.set_facecolor("white")
            axins.patch.set_alpha(0.92)
            for spine in axins.spines.values():
                spine.set_linewidth(0.5)

            # Plot onto inset
            for sim_name in sorted(df_case["Sim"].unique()):
                rt = df_case.loc[df_case["Sim"] == sim_name, "RunType"].iloc[0]
                color = case_to_color.get(case_id, "black")
                s = rt_styles.get(rt, rt_styles[next(iter(rt_styles))])
                plot_spec_eccentricity_control(
                    df_case, sim_name,
                    ax=axins,
                    color=color,
                    adot_mode="raw",
                    linestyle=s["linestyle"],
                    linewidth=0.6,
                    marker=s["marker"],
                    markersize=s["markersize"],
                    zorder=s["zorder"],
                    run_type=rt,
                    label="_nolegend_",
                )

            # Inset limits
            xin_min, xin_max = df_zoom["Omega0"].min(), df_zoom["Omega0"].max()
            yin_min, yin_max = df_zoom["Adot0"].min(),  df_zoom["Adot0"].max()

            xrange = (xin_max - xin_min) or 1e-6
            yrange = (yin_max - yin_min) or 1e-6

            pad_top = 1.8 * yrange     # space on top
            pad_bottom = 0.2 * yrange # space on bottom
            pad_left  = 0.4 * xrange  #  space on left
            pad_right = 0.3 * xrange  # space on right

            axins.set_xlim(xin_min - pad_left, xin_max + pad_right)
            axins.set_ylim(yin_min - pad_bottom, yin_max + pad_top)

            axins.xaxis.set_major_locator(MaxNLocator(3))
            axins.yaxis.set_major_locator(MaxNLocator(3))
            axins.xaxis.set_major_formatter(sci_offset_formatter())
            axins.yaxis.set_major_formatter(sci_offset_formatter())
            axins.tick_params(which="major", width=0.4, length=2,
                              labelsize=5, direction="in")
            axins.grid(False)
            for offset_text in (
                axins.xaxis.get_offset_text(), axins.yaxis.get_offset_text()
            ):
                offset_text.set_fontsize(5)
            axins.xaxis.get_offset_text().set_x(1)
            axins.xaxis.get_offset_text().set_y(1.15)

            # Bold annotation and highlighted marker on the final iteration
            # of each run type in the inset
            for sim_name in sorted(df_case["Sim"].unique()):
                for rt in run_types_present:
                    df_rt = df_case[
                        (df_case["Sim"] == sim_name) &
                        (df_case["RunType"] == rt)
                    ]
                    if df_rt.empty:
                        continue

                    last = df_rt.sort_values("EccLevel").iloc[-1]
                    s = rt_styles.get(rt, rt_styles[next(iter(rt_styles))])
                    pt_color = case_to_color.get(
                        last.get(case_col, case_id), "black"
                    )
                    axins.annotate(
                        f"{int(last['EccLevel'])}",
                        (last["Omega0"], last["Adot0"]),
                        xycoords="data",
                        textcoords="offset points",
                        xytext=(0, 0),
                        ha="center", va="center",
                        fontsize=4, fontweight="bold",
                        color="black", zorder=300, clip_on=True,
                    )
                    axins.plot(
                        last["Omega0"], last["Adot0"],
                        marker=s["marker"],
                        linestyle="None",
                        markersize=s["markersize"],
                        markerfacecolor=pt_color,
                        markeredgecolor=pt_color,
                        markeredgewidth=0.5,
                        zorder=0,
                    )

            # Tolerance ellipses (only get drawn when the next-step params are provided)
            if not new_params.empty:
                for rt in df_case["RunType"].dropna().unique():
                    add_concentric_tolerance_from_next_step(
                        axins,
                        df_case=df_case,
                        df_next=new_params,
                        case_id=case_id,
                        run_type=rt,
                        case_col=case_col,
                        base_color="hotpink",
                        multipliers=(3, 2, 1),
                        fill_alphas=(0.03, 0.06, 0.10),
                        edge_alphas=(0.18, 0.28, 0.45),
                        lw=(0.8, 0.9, 1.0),
                        zorder=1,
                        min_frac=None,
                    )

        # Main axis formatting
        ax.grid(True, alpha=0.35)
        ax.set_xlabel(r"$\Omega_0$", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel(r"$\dot{a}_0$", fontsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.xaxis.set_major_formatter(sci_offset_formatter())
        ax.yaxis.set_major_formatter(sci_offset_formatter())
        ax.tick_params(axis="both", which="major", labelsize=7)
        for offset_text in (
            ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()
        ):
            offset_text.set_fontsize(7)
        ax.xaxis.get_offset_text().set_x(1)
        ax.xaxis.get_offset_text().set_y(0)

    fig.canvas.draw()

    for ax in axes:
        for i, lab in enumerate(ax.get_xticklabels()):
            if i % 2 == 1:
                lab.set_visible(False)

    # Legends on the middle (or only) axis
    ax_legend = axes[len(axes) // 2]

    # Run type legend — use custom display labels if provided, otherwise
    # fall back to the raw RunType values from the data
    legend_label = run_type_legend_labels or {}
    run_type_handles = [
        mlines.Line2D(
            [], [], color="black",
            linestyle=rt_styles[rt]["linestyle"],
            marker=rt_styles[rt]["marker"],
            markersize=rt_styles[rt]["markersize"],
            linewidth=0.7,
            label=legend_label.get(rt, rt),
        )
        for rt in run_types_present
    ]
    type_legend = ax_legend.legend(
        handles=run_type_handles,
        title="Initial Guess Type",
        loc="upper left",
        bbox_to_anchor=(0.15, 1),
        fontsize=7, title_fontsize=7,
        handlelength=4, handletextpad=0.6,
        frameon=False,
    )
    ax_legend.add_artist(type_legend)

    # Case color legend — only show cases that actually appeared in the panels
    cases_in_plot = [c for c in cases_to_plot if c in cases_used]

    case_handles = [
        mlines.Line2D([], [], color=case_to_color[c], marker="s",
                      linestyle="None", markersize=4, label=str(c))
        for c in cases_in_plot
    ]
    if case_handles:
        ax_legend.legend(
            handles=case_handles, title="Case",
            loc="upper left", bbox_to_anchor=(0, 1),
            fontsize=7, title_fontsize=7, handletextpad=0, frameon=False,
        )

    fig.tight_layout()
    fig.savefig(output_path)
    return fig

# Option 3: Iteration counts scatter plot
def plot_iteration_counts(
    df: pd.DataFrame,
    annotate: str = "iterations",
    title: Optional[str] = None,
):
    """
    Scatter plot showing the number of eccentricity reduction iterations
    per simulation in (MassRatio, Initial Separation) space.

    Each point represents one simulation. Marker size encodes the iteration
    count; colour encodes the simulation name.

    \f
    Arguments:
        df: per-iteration simulation data with columns
            [Sim, EccLevel, MassRatio, Initial Separation].
        annotate: 'iterations' to label each point with its count, or 'sim'
                  to label with the simulation name.
        title: plot title; if None, default is used.

    Returns:
        fig (matplotlib.figure.Figure): Figure containing the plot
        ax (matplotlib.axes.Axes): Axis
    """
    required_cols = {"Sim", "EccLevel", "MassRatio", "Initial Separation"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    counts_df = (
        df.groupby("Sim").agg(
            Iterations=("EccLevel", "count"),
            MassRatio=("MassRatio", "first"),
            InitialSeparation=("Initial Separation", "first"),
        )
        .reset_index()
    )
    counts_df["Iterations"] -= 1  # EccLevel 0 is the initial guess

    sizes = 80 * counts_df["Iterations"]
    unique_sims = counts_df["Sim"].unique()
    color_map = dict(zip(unique_sims, plt.cm.tab20.colors[: len(unique_sims)]))
    colors = counts_df["Sim"].map(color_map)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        counts_df["MassRatio"],
        counts_df["InitialSeparation"],
        c=colors,
        s=sizes,
        alpha=0.5,
        zorder=10,
    )

    texts = []
    for _, row in counts_df.iterrows():
        label = (
            str(row["Iterations"]) if annotate == "iterations" else row["Sim"]
        )
        txt = ax.text(
            row["MassRatio"], row["InitialSeparation"], label,
            fontsize=8, color=color_map[row["Sim"]], zorder=20,
            path_effects=[
                path_effects.withStroke(linewidth=1.2, foreground="black")
            ],
        )
        texts.append(txt)

    adjust_text(
        texts,
        expand_points=(1.2, 1.4),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        ax=ax,
    )

    ax.set_xlabel("Mass Ratio")
    ax.set_ylabel("Initial Separation")
    ax.set_title(
        title if title is not None
        else "Number of Eccentricity Reduction Iterations"
    )
    ax.grid(True, zorder=0)

    handles = [
        ax.scatter([], [], color=[c], s=60, alpha=0.7)
        for c in color_map.values()
    ]
    ax.legend(
        handles, list(color_map.keys()),
        bbox_to_anchor=(1, 1), loc="upper left",
        fontsize=10, title="Simulations",
    )

    return fig

# CLIs
@click.command(
    name="spec-eccentricity-control",
    help=plot_spec_eccentricity_control.__doc__,
)
@click.argument(
    "h5_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
)
@click.option(
    "--output", "-o",
    default="trajectories.pdf",
    show_default=True,
    help="Path for the output PDF.",
)
@click.option(
    "--lev",
    required=True,
    help=(
        "Lev directory to read metadata from, e.g. 'Lev3' or 'Lev4'. "
        "Must match a directory present in the HDF5 file structure. "
    ),
)
@click.option(
    "--run-type-col",
    default="RunType",
    show_default=True,
    help=(
        "Column in the loaded DataFrame that identifies the run type for "
        "each simulation (e.g. 'GPR', 'PN'). This column must be "
        "present in the data before calling the plotting functions. "
        "Defaults to 'RunType'."
    ),
)
@click.option(
    "--case-col",
    default="Sim",
    show_default=True,
    help=(
        "Column to use as the case identifier for panel grouping. "
        "Defaults to 'Sim' (one panel per simulation). Change to any "
        "other column present in the data, e.g. 'MassRatio'."
    ),
)
def plot_spec_eccentricity_control_command(
    h5_files, output, lev, run_type_col, case_col
):
    """Load HDF5 files and produce the trajectory panel plot.
 
    Before running, ensure that your DataFrame has a column identifying the run
    type for each simulation (e.g. 'GPR', 'PN').
    """
    if not h5_files:
        raise click.UsageError("Provide at least one HDF5 file.")

    frames = [process_h5_file(f) for f in h5_files]
    df_iter = pd.concat(frames, ignore_index=True)

    for col in (run_type_col, case_col):
        if col not in df_iter.columns:
            raise click.UsageError(
                f"Column '{col}' not found in the loaded data. "
                f"Available columns: {list(df_iter.columns)}"
            )

    if df_iter[run_type_col].isna().all():
        raise click.UsageError(
            f"Column '{run_type_col}' is empty. "
            "Please populate it with run type labels (e.g. 'GPR', "
            "'PN') before running."
        )

    cases_to_plot = sorted(df_iter[case_col].dropna().unique())

    plot_trajectories(
        df_iter,
        case_col=case_col,
        cases_to_plot=cases_to_plot,
        drop_by_panel={},
        new_params=pd.DataFrame(),
        output_path=output,
    )
    logger.info("Saved %s", output)


if __name__ == "__main__":
    plot_spec_eccentricity_control_command(help_option_names=["-h", "--help"])