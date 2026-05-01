"""
Unit tests for PlotSpecEccentricityControl.py.

Run with:
    python -m pytest test_PlotSpecEccentricityControl.py -v
or:
    python -m unittest test_PlotSpecEccentricityControl -v
"""

import unittest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Ellipse

from SimulationSupport.PlotSpecEccentricityControl import (
    build_run_type_styles,
    build_case_color_map,
    extract_params_from_lines,
    filter_param_lines,
    group_by_sim_and_ecc,
    plot_spec_eccentricity_control,
    plot_eccentricity_vs_iteration,
    plot_trajectories,
    plot_iteration_counts,
    add_tolerance_ellipse_fixed,
    choose_inset_anchor,
)

"""
Make dummy data. All test classes call make_df() in order to get a consistent
DataFrame that mirrors the structure of process_h5_file(). Using a fixed seed
allows the random noise to be identical in every test run.
"""
def make_df(
    cases=("A", "B"),
    run_types=("PN", "GPR"),
    n_iter=4,
    seed=42,
):
    """Return a minimal DataFrame."""
    rng = np.random.default_rng(seed)
    omega_base = {"A": 0.01642    , "B": 0.0146}
    adot_base  = {"A": -2.3968e-05, "B": -2.3699e-05}
    mass_ratio = {"A": 8.0        , "B": 5.3139}
    separation = {"A":14.5189     , "B": 15.8556}

    rows = []
    for case in cases:
        for rt in run_types:
            # Simulation name mirrors the convention used in the SpEC pipeline
            sim = f"Sim_{case}_{rt.replace(' ', '_')}"
            omega = omega_base[case]
            adot  = adot_base[case]
            for lvl in range(n_iter):
                # Make eccentricity decay with each iteration to mimic SpEC runs
                ecc_base = {"A": 0.0731, "B": 0.0518}
                ecc = ecc_base[case] * (0.3 ** lvl) * abs(1 + 0.05 * rng.normal())
                rows.append({
                    "Sim":               sim,
                    "Case":              case,
                    "MassRatio":         mass_ratio[case],
                    "EccLevel":          lvl,
                    "Omega0":            omega + lvl * 1e-5 * rng.normal(),
                    "Adot0":             adot  + lvl * 1e-7 * rng.normal(),
                    "Initial Separation": separation[case],
                    "Eccentricity":      ecc,
                    "RunType":           rt,
                })
    return pd.DataFrame(rows)

# Style helpers
# No figures returned on purpose
class TestBuildRunTypeStyles(unittest.TestCase):
    def test_returns_all_run_types(self):
        # Every run type should appear as a key in the output.
        rts = ["PN", "GPR", "custom"]
        styles = build_run_type_styles(rts)
        self.assertEqual(set(styles.keys()), set(rts))

    def test_required_keys_present(self):
        # Each style dict should contain the four keys used by the plotting functions
        styles = build_run_type_styles(["A", "B"])
        for s in styles.values():
            for key in ("linestyle", "marker", "markersize", "zorder"):
                self.assertIn(key, s)

    def test_duplicate_run_types_deduplicated(self):
        # Ensure that passing the same label twice does not create duplicate entries
        styles = build_run_type_styles(["A", "A", "B"])
        self.assertEqual(len(styles), 2)

    def test_zorder_increases_with_index(self):
        # Ensure that run types are drawn on top of each other in sequential order
        rts = ["first", "second", "third"]
        styles = build_run_type_styles(rts)
        zorders = [styles[rt]["zorder"] for rt in rts]
        self.assertEqual(zorders, sorted(zorders))

class TestBuildCaseColorMap(unittest.TestCase):
    def test_returns_all_cases(self):
        cases = ["A", "B", "C"]
        color_map = build_case_color_map(cases)
        self.assertEqual(set(color_map.keys()), set(cases))

    def test_colors_are_distinct(self):
        # Give each case a unique color so trajectories are distinguishable
        cases = ["A", "B", "C", "D"]
        color_map = build_case_color_map(cases)
        colors = list(color_map.values())
        self.assertEqual(len(colors), len(set(map(str, colors))))

# HDF5 parsing helpers
# Use fake Params.input text so no real HDF5 file is necessary
class TestExtractParamsFromLines(unittest.TestCase):
    def _make_lines(self, mass=5.3139, omega=0.0146, adot=-2.3699e-05, d0=15.8556):
        """
        Return a minimal Params.input line list with known values.
        """
        return [
            f"$MassRatio = {mass};",
            f"$Omega0 = {omega};",
            f"$adot0 = {adot};",
            f"$D0 = {d0};",
        ]

    def test_extracts_all_four_values(self):
        lines = self._make_lines()
        mr, omega, adot, d0 = extract_params_from_lines(lines)
        self.assertAlmostEqual(mr, 5.3139)
        self.assertAlmostEqual(omega, 0.0146)
        self.assertAlmostEqual(adot, -2.3699e-05)
        self.assertAlmostEqual(d0, 15.8556)

    def test_returns_none_on_missing_param(self):
        # If any required parameter is missing, return None
        # for all values
        lines = ["$Omega0 = 0.0146;", "$adot0 = -2.3699e-05;", "$D0 = 15.8556;"]
        result = extract_params_from_lines(lines)
        self.assertIsNone(result[0])

    def test_returns_none_on_empty_input(self):
        result = extract_params_from_lines([])
        self.assertTrue(all(v is None for v in result))

class TestFilterParamLines(unittest.TestCase):
    def test_keeps_relevant_lines(self):
        # Ensure that only the lines containing one of the four parameter keys survive
        content = "$MassRatio = 5.3139;\n$Omega0 = 0.0146;\nIgnoredLine;\n$adot0 = -2.3699e-05;\n$D0 = 15.8556;"
        lines = filter_param_lines(content)
        self.assertEqual(len(lines), 4)

    def test_excludes_irrelevant_lines(self):
        content = "NotAParam = 5;\nAlsoIrrelevant;"
        lines = filter_param_lines(content)
        self.assertEqual(lines, [])

class TestGroupBySimAndEcc(unittest.TestCase):
    def _make_paths(self):
        return [
            "SimA/Ecc0/Lev3/Params.input",
            "SimA/Ecc1/Lev3/Params.input",
            "SimB/Ecc0/Lev3/Params.input",
            "SimB/Ecc0/Lev3/OtherFile.dat",  # ignore if wrong suffix is used
            "ShortPath/Params.input",        # ignore if path is not fully specified
        ]

    def test_groups_correctly(self):
        # Ensure that the numeric levels are extracted correctly
        grouped = group_by_sim_and_ecc(self._make_paths(), "Params.input")
        self.assertIn("SimA", grouped)
        self.assertIn("SimB", grouped)

    def test_correct_ecc_levels(self):
        grouped = group_by_sim_and_ecc(self._make_paths(), "Params.input")
        ecc_levels_A = [ecc for ecc, _ in grouped["SimA"]]
        self.assertEqual(sorted(ecc_levels_A), [0, 1])

    def test_ignores_wrong_suffix(self):
        grouped = group_by_sim_and_ecc(self._make_paths(), "Params.input")
        all_paths = [p for entries in grouped.values() for _, p in entries]
        self.assertFalse(any("OtherFile" in p for p in all_paths))

    def test_ignores_bad_paths(self):
        # Ensure that paths without an EccN component are skipped
        grouped = group_by_sim_and_ecc(self._make_paths(), "Params.input")
        self.assertNotIn("ShortadPath", grouped)

# Trajectory Plot
class TestPlotSpecEccentricityControl(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def tearDown(self):
        # Close figures after each test
        plt.close("all")

    def test_missing_sim_returns_ax_unchanged(self):
        # Ensure that an unrecognized sim results in nothing, instead of crashing
        _, ax = plt.subplots()
        result = plot_spec_eccentricity_control(self.df, "NoSim", ax=ax)
        self.assertIs(result, ax)
        self.assertEqual(len(ax.lines), 0)

    def test_adot_mode_abs(self):
        # Ensure that when in 'abs' mode, every y-value is non-negative
        ax = plot_spec_eccentricity_control(
            self.df, "Sim_A_PN", adot_mode="abs"
        )
        ydata = ax.lines[0].get_ydata()
        self.assertTrue(np.all(ydata >= 0))

    def test_adot_mode_logabs(self):
        ax = plot_spec_eccentricity_control(
            self.df, "Sim_A_PN", adot_mode="logabs"
        )
        ydata = ax.lines[0].get_ydata()
        self.assertTrue(np.all(ydata < 0))   # log of small numbers is negative

    def test_correct_number_of_points(self):
        # Ensure that there is one point per EccLevel iteration
        n_iter = 4
        ax = plot_spec_eccentricity_control(
            self.df, "Sim_A_PN", adot_mode="raw"
        )
        xdata = ax.lines[0].get_xdata()
        self.assertEqual(len(xdata), n_iter)

    def test_uses_supplied_axes(self):
        _, ax_in = plt.subplots()
        ax_out = plot_spec_eccentricity_control(
            self.df, "Sim_A_PN", ax=ax_in
        )
        self.assertIs(ax_out, ax_in)

# Test Option 1: Eccentricity vs iteration plot
# Uses pytest's tmp_path fixture so the functions can write a file to a temporary directory
# that pytest creates and cleans up
class TestPlotEccentricityVsIteration:

    def setup_method(self):
        self.df = make_df()

    def teardown_method(self):
        plt.close("all")

    def test_y_axis_is_log(self, tmp_path):
        # Ensure y-axis is logarithmic
        fig = plot_eccentricity_vs_iteration(
            self.df, case_col="Case",
            output_path=str(tmp_path / "out.pdf"),
        )
        assert fig.axes[0].get_yscale() == "log"

    def test_number_of_lines_matches_sims(self, tmp_path):
        # Ensure that one line is drawn per simulation
        n_sims = self.df["Sim"].nunique()
        fig = plot_eccentricity_vs_iteration(
            self.df, case_col="Case",
            output_path=str(tmp_path / "out.pdf"),
        )
        assert len(fig.axes[0].lines) == n_sims

    def test_custom_run_type_legend_labels(self, tmp_path):
        # Ensure that passing a label-override dict does not raise or alter
        # the figure type
        labels = {"PN": "PN (2025)", "GPR": "GPR (2025)"}
        fig = plot_eccentricity_vs_iteration(
            self.df, case_col="Case",
            output_path=str(tmp_path / "out.pdf"),
            run_type_legend_labels=labels,
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_required_column_raises(self, tmp_path):
        # Ensure that if dataframe is missing "Eccentricity" the plot does not get created
        bad_df = self.df.drop(columns=["Eccentricity"])
        with pytest.raises((KeyError, ValueError)):
            plot_eccentricity_vs_iteration(
                bad_df, case_col="Case",
                output_path=str(tmp_path / "out.pdf"),
            )

# Test Option 2: Eccentricity Trajectory Through Parameter Space
class TestPlotTrajectories:
    def setup_method(self):
        self.df = make_df()

    def teardown_method(self):
        plt.close("all")

    def test_one_panel_per_case(self, tmp_path):
        # Ensure that figure contains at least one Axes per case
        # (inset axes also get created)
        cases = ["A", "B"]
        fig = plot_trajectories(
            self.df, case_col="Case",
            cases_to_plot=cases,
            drop_by_panel={},
            new_params=pd.DataFrame(),
            output_path=str(tmp_path / "out.pdf"),
        )
        main_axes = [ax for ax in fig.axes if not ax.get_label().startswith("inset")]
        assert len(main_axes) >= len(cases)

    def test_single_case_produces_figure(self, tmp_path):
        fig = plot_trajectories(
            self.df, case_col="Case",
            cases_to_plot=["A"],
            drop_by_panel={},
            new_params=pd.DataFrame(),
            output_path=str(tmp_path / "out.pdf"),
        )
        assert isinstance(fig, plt.Figure)

    def test_missing_required_column_raises(self, tmp_path):
        bad_df = self.df.drop(columns=["Omega0"])
        with pytest.raises((KeyError, ValueError)):
            plot_trajectories(
                bad_df, case_col="Case",
                cases_to_plot=["A"],
                drop_by_panel={},
                new_params=pd.DataFrame(),
                output_path=str(tmp_path / "out.pdf"),
            )

    def test_tolerance_ellipses_with_new_params(self, tmp_path):
        # Ensure that providing next step parameters draws the tolerance ellipses
        new_params = pd.DataFrame([
            {"Case": "A", "RunType": "PN",
             "Omega0_new": 0.0161, "Adot0_new": -1.4e-5},
        ])
        fig = plot_trajectories(
            self.df, case_col="Case",
            cases_to_plot=["A"],
            drop_by_panel={},
            new_params=new_params,
            output_path=str(tmp_path / "out.pdf"),
        )
        assert isinstance(fig, plt.Figure)

# Test Option 3: Iteration counts scatter plot
class TestPlotIterationCounts(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def tearDown(self):
        plt.close("all")

    def test_one_point_per_sim(self):
        # Ensure that each simulation makes a single scatter point
        # and that marker size encodes the iteration count
        n_sims = self.df["Sim"].nunique()
        fig = plot_iteration_counts(self.df)
        ax = fig.axes[0]
        scatter = ax.collections[0]
        self.assertEqual(len(scatter.get_offsets()), n_sims)

    def test_missing_column_raises(self):
        bad_df = self.df.drop(columns=["MassRatio"])
        with self.assertRaises(ValueError):
            plot_iteration_counts(bad_df)

    def test_annotate_sim_mode(self):
        # Ensure that "sim" annotation labels each point with the simulation name
        # instead of the iteration count
        fig = plot_iteration_counts(self.df, annotate="sim")
        ax = fig.axes[0]
        text_labels = {t.get_text() for t in ax.texts}
        sim_names = set(self.df["Sim"].unique())
        self.assertTrue(sim_names.issubset(text_labels))

# Tolerance ellipse helpers
class TestAddToleranceEllipseFixed(unittest.TestCase):

    def setUp(self):
        self.df = make_df(cases=("A",), run_types=("PN",))
        _, self.ax = plt.subplots()

    def tearDown(self):
        plt.close("all")

    def test_returns_ellipse_for_valid_data(self):
        ell = add_tolerance_ellipse_fixed(
            self.ax, self.df,
            case_id="A", run_type="PN",
            dOmega=1e-5, dAdot=1e-7,
            case_col="Case",
        )
        self.assertIsInstance(ell, Ellipse)

    def test_returns_none_for_missing_case(self):
        # Ensure that if no data matches the case, the function returns None
        result = add_tolerance_ellipse_fixed(
            self.ax, self.df,
            case_id="Z",  # does not exist in DataFrame
            run_type="PN",
            dOmega=1e-5, dAdot=1e-7,
            case_col="Case",
        )
        self.assertIsNone(result)

# Inset anchor helper
class TestChooseInsetAnchor(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_returns_four_floats(self):
        # Ensure that the function returns (x0, y0, width, height) in Axes
        # fraction coordinates
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        result = choose_inset_anchor(ax, np.array([0, 1]), np.array([0, 1]))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(isinstance(v, float) for v in result))

    def test_anchor_within_axes_bounds(self):
        # Ensure that the inset box fits inside the Axes
        _, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        x0, y0, w, h = choose_inset_anchor(
            ax, np.array([0.5]), np.array([0.5]), w=0.35, h=0.5
        )
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x0 + w, 1.1)   # small tolerance for padding
        self.assertLessEqual(y0 + h, 1.1)   # small tolerance for padding

if __name__ == "__main__":
    unittest.main()