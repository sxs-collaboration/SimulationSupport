# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Unit tests for gpr.py
    Generate a simple, synthetic regression problem and run the pipeline.
    Assert that the shapes are reasonable, uncertainties are positive, and
    that the mean predictions are within a plausible range of the true values.

Run with:
    python -m unittest test_gpr.py -v
"""

import shutil
import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import matplotlib
# Use a non-interactive backend so the tests can run in CLI without plot outputs
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SimulationSupport.gpr import (
    normalize_data,
    denormalize_predictions,
    plot_loo_residuals,
    load_gpr_checkpoint,
    predict_with_gpr_model,
    run_gpr_pipeline,
    train_gpr_model,
    train_model_and_eigenvalue_analysis,
    loo_predictions,
    parse_test_runs,
    apply_gpr_corrections,
    save_gpr_corrected,
    loo_crossval,
)

# build dummy data using the first 25 rows of the q87d subset of the SXS catalog
# so the GPR sees realistic inputs

data = {
    "name":                      ["0005/Lev3","0024/Lev3","0029/Lev3","0033/Lev3","0040/Lev3",
                                  "0053/Lev3","0061/Lev3","0074/Lev3","0075/Lev3","0090/Lev3",
                                  "0098/Lev3","0103/Lev3","0116/Lev3","0119/Lev3","0129/Lev3",
                                  "0136/Lev3","0137/Lev3","0150/Lev3","0167/Lev3","0172/Lev3",
                                  "0196/Lev3","0202/Lev3","0208/Lev3","0227/Lev3","0233/Lev3"],
    "initial_separation":        [15.4389038086,15.4389038086,15.5040283203,14.6396484375,14.5452880859,
                                  14.1087646484,14.6162719727,13.7968139648,13.7147827148,14.5942382812,
                                  13.7731323242,15.5687255859,15.5036010742,14.6194458008,15.2434082031,
                                  15.2301025391,14.6903686523,13.893371582, 15.3317260742,14.6107177734,
                                  14.7182617188,13.9721069336,14.0308837891,14.0565795898,15.1602783203],
    "initial_orbital_frequency": [0.014654531914,0.014860142586,0.015084361469,0.01620808222,0.015785070996,
                                  0.016362387988,0.016242686536,0.016949502583,0.01706179036,0.016079896462,
                                  0.017149966717,0.014894346572,0.015112114431,0.015844335773,0.015124364223,
                                  0.015266118864,0.015667744442,0.016664057381,0.014942674686,0.01631465861,
                                  0.015710440024,0.017116384531,0.016794580234,0.016911672833,0.015638646724],
    "initial_adot":              [0.0002543673632399, 0.0004166173600393,-0.0002060797755785, 1.77357245289e-05,
                                  0.0004084045122486, 0.0004373360466015,-0.0001221160761242,-0.0002222554346198,
                                 -7.46014821058e-05,  0.0004880880687731,-0.0002824217451136,-0.0001103804210338,
                                 -5.67145259009e-05,  0.0003876877144537, 0.0003522984264178, 0.0003360603167823,
                                 -0.0002297042844619,  0.000477850411973,-0.000273810769737,  3.90284734822e-05,
                                 -0.0002504804492629,  0.0004482434609702, 0.0001333317109452, 7.25875286598e-05,
                                  2.09845214299e-05],
    "initial_mass1":             [0.8888887661,0.8888888831,0.8888888923,0.8888889298,0.8888889684,
                                  0.8888889301,0.8888889164,0.8888888850,0.8888888763,0.8888888843,
                                  0.8888888550,0.8888888540,0.8888888952,0.8888887652,0.8888888915,
                                  0.8888888929,0.8888888599,0.8888888538,0.8888888983,0.8888888887,
                                  0.8888889493,0.8888887748,0.8883642105,0.8851189822,0.8847983839],
    "initial_mass2":             [0.1111111071,0.1111111041,0.1111111057,0.1111111031,0.1111111034,
                                  0.1111110952,0.1111111047,0.1111111038,0.1111111121,0.1111111121,
                                  0.1111111171,0.1111111039,0.1111111063,0.1111110983,0.1111111122,
                                  0.1111111126,0.1111111011,0.1111111016,0.1111111090,0.1111111120,
                                  0.1111110994,0.1111111047,0.1116357477,0.1148809394,0.1152016166],
    "mass_ratio":                [7.999999181,8.000000454,8.000000417,8.000000947,8.000001272,
                                  8.000001515,8.000000707,8.000000493,7.999999815,7.999999888,
                                  7.999999266,8.000000205,8.000000406,7.999999810,7.999999942,
                                  7.999999930,8.000000460,8.000000367,8.000000238,7.999999937,
                                  8.000001384,7.999999434,7.957703771,7.704663516,7.680433748],
    "S1x":                       [0.0029858096,-0.0085569936, 0.0076431454, 4.584092e-10,-0.1648777061,
                                  4.93e-14,    -0.6903272443,-2.075e-13,    0.0048135673, 0.6882731085,
                                  0.0089373354,-0.0083914523,-0.0018924338,-0.5121848050,-0.5560521742,
                                 -0.2859498352,-0.7949598565,-0.2826884009,-0.0634056275, 0.5825820193,
                                  0.2207038127,-0.5696952911,-0.1918726915, 0.3101865861,-0.1082468279],
    "S1y":                       [-0.0084486166, 0.0030999687, 0.0081180109,-2.121064e-09,-0.3644327332,
                                 -1.451e-13,    0.4041708160, 3.106e-13,    0.0147820718, 0.4076044929,
                                  0.0072301099,-5.53496770114e-05,-0.0053780953,-0.6145119322,-0.1220366044,
                                  0.5073077057,-0.0895995945,-0.4903055869,-0.5680175544,-0.5480483778,
                                  0.3335956697,-0.0509227797,-0.5331555565,-0.4531958787,-0.5537179453],
    "S1z":                       [-0.7999498301,-0.7999482259,-0.7999222761, 3.16194906e-08, 0.0009580564,
                                  0.4000000094, 0.0089571880, 0.7999999912, 0.7998489471,-0.0115447217,
                                  0.7999174209,-0.7999560067,-0.7999796919,-0.0052652532,-0.5620582843,
                                 -0.5485161020,-0.0005410857, 0.5654062129,-0.5597565670, 0.0150459354,
                                 -1.45776072113e-05, 0.5593302960, 0.5054170288, 0.5776470753,-0.3879338734],
    "S2x":                      [-0.5645005528, 0.3328568564, 0.3677119674, 0.3917565944,-0.0444657894,
                                 -9.249e-13,    0.2680696007, 3.071e-12,   -0.7453978438, 0.5674248406,
                                 -0.3610413280, 0.1416101602,-0.2976859903, 0.7471041464, 0.6775265354,
                                  0.6175278943,-0.0338204706,-0.0668072084, 0.0686693855, 0.0431199097,
                                 -0.0364000622,-0.3636767042,-0.5453375753,-0.2493440242,-0.0096849315],
    "S2y":                      [ 0.0390313909, 0.4578603629,-0.7104735099,-0.0807751389, 0.0400815429,
                                  6.915e-13,   -0.7530121235, 8.711e-13,    0.2904554361, 0.5601653574,
                                  0.4378772819, 0.5473793025, 0.2671696013, 0.2851266592, 0.4225247241,
                                  0.5041956446,-0.1070785015, 0.0652872564,-0.0350587122, 0.3964557110,
                                  0.0446577749,-0.7066452498, 0.1662539400, 0.6440094114,-0.4143960212],
    "S2z":                      [ 0.5655165723, 0.5652995012,-2.37727910172e-05, 0.0003757402, 0.7977569301,
                                  0.8000000541, 0.0331604673,-0.8000000174, 0.0041122862, 0.0651097883,
                                 -0.5638344849,-0.5659641337, 5.9622237494e-06, 0.0229361990,-0.0492440706,
                                 -0.0666594438,-0.7920795750, 0.7945276682,-0.7962758525, 0.0309907677,
                                 -0.7979226469, 0.0915710403, 0.3331595869, 0.3315026153,-0.0276898088],
    "eccentricity":             [2.77e-05,8.42e-05,5.84e-05,9.57e-05,4.59e-05,
                                 1.75e-05,6.86e-05,9.47e-05,8.53e-05,8.98e-05,
                                 9.26e-05,8.19e-05,4.67e-05,5.53e-05,6.39e-05,
                                 9.25e-05,6.94e-05,5.29e-05,3.41e-05,8.61e-05,
                                 5.59e-05,9.28e-05,7.44e-05,5.85e-05,7.23e-05],
    "spec_pn_guess_omega":      [0.014654531914,0.014860142586,0.015084361469,0.01620808222,0.015785070996,
                                 0.016362387988,0.016242686536,0.016949502583,0.01706179036,0.016079896462,
                                 0.017149966717,0.014894346572,0.015112114431,0.015844335773,0.015124364223,
                                 0.015266118864,0.015667744442,0.016664057381,0.014942674686,0.01631465861,
                                 0.015710440024,0.017116384531,0.016794580234,0.016911672833,0.015638646724],
    "spec_pn_guess_adot":       [0.0002543673632399, 0.0004166173600393,-0.0002060797755785, 1.77357245289e-05,
                                 0.0004084045122486, 0.0004373360466015,-0.0001221160761242,-0.0002222554346198,
                                 -7.46014821058e-05,  0.0004880880687731,-0.0002824217451136,-0.0001103804210338,
                                 -5.67145259009e-05,  0.0003876877144537, 0.0003522984264178, 0.0003360603167823,
                                 -0.0002297042844619,  0.000477850411973,-0.000273810769737,  3.90284734822e-05,
                                 -0.0002504804492629,  0.0004482434609702, 0.0001333317109452, 7.25875286598e-05,
                                 2.09845214299e-05],
}


def make_df():
    """Return a DataFrame built from 25 rows of the q87d subset of the SXS catalog."""
    return pd.DataFrame(data)

# Test Data normalization
class TestNormalizeData(unittest.TestCase):

    def test_output_shapes_match_input(self):
        # Test that the normalized arrays have the same number of rows as the input
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        X_norm, Y_norm, _, _ = normalize_data(X, Y)
        self.assertEqual(X_norm.shape[0], len(X))
        self.assertEqual(Y_norm.shape[0], len(Y))

    def test_normalized_mean_is_zero(self):
        # Test that standardization shifts the mean to ~0
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        X_norm, Y_norm, _, _ = normalize_data(X, Y)
        self.assertAlmostEqual(float(np.mean(X_norm)), 0.0, places=10)
        self.assertAlmostEqual(float(np.mean(Y_norm)), 0.0, places=10)

    def test_normalized_std_is_one(self):
        # Test that standardization scales the standard deviation to ~1
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        X_norm, Y_norm, _, _ = normalize_data(X, Y)
        self.assertAlmostEqual(float(np.std(X_norm)), 1.0, places=10)
        self.assertAlmostEqual(float(np.std(Y_norm)), 1.0, places=10)

    def test_returns_four_objects(self):
        # Test that the function returns (X_norm, Y_norm, scaler_X, scaler_Y)
        X = np.array([1.0, 2.0, 3.0])
        Y = np.array([4.0, 5.0, 6.0])
        result = normalize_data(X, Y)
        self.assertEqual(len(result), 4)

    def test_scalers_can_inverse_transform(self):
        # Verify that scaler_Y is able to recover the original Y values
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        _, Y_norm, _, scaler_Y = normalize_data(X, Y)
        Y_recovered = scaler_Y.inverse_transform(Y_norm).squeeze()
        np.testing.assert_allclose(Y_recovered, Y, rtol=1e-10)

# Denormalize predictions
class TestDenormalizePredictions(unittest.TestCase):

    def setUp(self):
        # Normalize a simple array so we have a fitted scaler_Y to test against
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        _, Y_norm, _, self.scaler_Y = normalize_data(X, Y)
        self.Y_norm = Y_norm.squeeze()
        self.Y_orig = Y

    def test_mean_recovers_original_scale(self):
        # Test that passing the normalized values returns the original Y values
        stddev = np.zeros_like(self.Y_norm)
        mean_un, _ = denormalize_predictions(self.Y_norm, stddev, self.scaler_Y)
        np.testing.assert_allclose(mean_un, self.Y_orig, rtol=1e-6)

    def test_stddev_is_non_negative(self):
        # Test that uncertainties are always non-negative after denormalization
        stddev_norm = np.abs(np.random.randn(len(self.Y_norm)))
        _, stddev_un = denormalize_predictions(self.Y_norm, stddev_norm, self.scaler_Y)
        self.assertTrue(np.all(stddev_un >= 0))

    def test_zero_stddev_stays_zero(self):
        # Test that zero uncertainty in normalized space remaisn zero after rescaling
        stddev_norm = np.zeros(len(self.Y_norm))
        _, stddev_un = denormalize_predictions(self.Y_norm, stddev_norm, self.scaler_Y)
        np.testing.assert_allclose(stddev_un, 0.0, atol=1e-10)


class TestGPRLibrary(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Train the GPR model once for the entire test class. Reuse the trained
        model across tests to avoid running the training loop every time for each test.
        """
        cls.test_dir = Path("SimulationSupport") / "gpr_tests"
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        cls.X, cls.Y = cls.make_data(n=20)

        # Train once - gets reused by all tests except test_run_gpr_pipeline,
        # which calls the full pipeline internally
        cls.model, cls.likelihood = train_gpr_model(cls.X, cls.Y)

        # Predictions also get computed once and reused
        cls.Y_pred, cls.Y_std = predict_with_gpr_model(
            cls.X, cls.model, cls.likelihood
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up and remove test directory after all the tests in the class are done."""
        shutil.rmtree(cls.test_dir)

    @staticmethod
    def make_data(n=20, noise_lev=0.05):
        """
        Generate a small synthetic regression dataset -
        points are sampled from a noisy sine curve:
        y = sin(5 * x) + Gaussian noise.

        The default is intentionally small (n=20) to keep the tests fast.
        It is large enough to verify shapes, normalization, and basic
        prediction quality, but not intended to test the accuracy of the GPR model itself.
        """
        # Random number generator with a fixed seed 0;  gets the same random data every time
        rng = np.random.default_rng(0)

        # Inputs with shape (n, 1)
        X = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)

        # Used flattened X with shape (n,) to evaluate the function
        y = np.sin(5 * X.reshape(-1))

        # Add n Gaussian noise samples with mean 0 and variance 1
        noise = noise_lev * rng.standard_normal(n)

        Y = y + noise  # Observed data is y = sin(5x) + Gaussian noise

        return X, Y

    def test_train_predict_and_uncertainty(self):
        """
        Check that:
        - train_gpr_model and predict_with_gpr_model run and
            return predictions of the correct shape
        - uncertainties are positive and non-trivial
        """
        # Check shapes
        self.assertEqual(
            self.Y_pred.shape,
            self.Y.shape,
            f"Expected prediction shape {self.Y.shape}, got"
            f" {self.Y_pred.shape}",
        )

        self.assertEqual(
            self.Y_std.shape,
            self.Y.shape,
            f"Expected standard deviation shape {self.Y.shape}, got"
            f" {self.Y_std.shape}",
        )

        # Check uncertainties are non-negative
        self.assertTrue(
            np.all(self.Y_std >= 0),
            f"Some of the predicted standard deviations are negative",
        )

        # Check that uncertainties are nontrivial
        self.assertTrue(
            np.any(self.Y_std > 1e-10), "All uncertainties are zero"
        )

    def test_run_gpr_pipeline(self):
        """
        Test that run_gpr_pipeline correctly trains, predicts, and returns
        outputs of the right shape. The underlying functions train_gpr_model
        and predict_with_gpr_model are tested separately with real computation in
        test_train_predict_and_uncertainty.
        """
        mock_pred = np.ones_like(self.Y)
        mock_std = np.ones_like(self.Y) * 0.1

        with (
            patch(
                "SimulationSupport.gpr.train_gpr_model",
                return_value=(self.model, self.likelihood),
            ),
            patch(
                "SimulationSupport.gpr.predict_with_gpr_model",
                return_value=(mock_pred, mock_std),
            ),
        ):

            model, likelihood, Y_pred = run_gpr_pipeline(
                self.X, self.Y, target_name="test", plot=False, silent=True
            )
            self.assertEqual(Y_pred.shape, self.Y.shape)
            self.assertIsNotNone(model)
            self.assertIsNotNone(likelihood)

    def test_normalization_stored_and_applied(self):
        """
        Test that the normalization parameters are stored and applied correctly:
        - input_mean and input_std have correct shapes
        - normalized inputs, X, have mean of ~0 and std of ~1  per feature
        """
        # Check shapes of stored statistics
        exp_shape = (self.X.shape[1],)

        self.assertEqual(
            self.model.input_mean.shape,
            exp_shape,
            f"Expected input_mean shape {exp_shape}, got"
            f" {self.model.input_mean.shape}",
        )

        self.assertEqual(
            self.model.input_std.shape,
            exp_shape,
            f"Expected input_std shape {exp_shape}, got"
            f" {self.model.input_std.shape}",
        )

        # Recreate normalized X using the stored statistics
        normalized_X = (self.X - self.model.input_mean) / self.model.input_std

        # Column-wise mean and std
        col_means = normalized_X.mean(axis=0)
        col_stds = normalized_X.std(axis=0)

        # Check that each mean is close to 0 with an allowed tolerance of 0.1
        self.assertTrue(
            np.allclose(col_means, 0, atol=1e-1),
            f"Expected means ~0, got {col_means}",
        )

        # Check that each std is close to 1 with an allowed tolerance of 0.1
        self.assertTrue(
            np.allclose(col_stds, 1.0, atol=1e-1),
            f"Expected stddevs ~1, got {col_stds}",
        )

    def test_output_denorm(self):
        """
        Test that the stored output_mean and output_std are consistent with
        denormalize_output, i.e. that it is correctly undoing the normalization.
        """
        # Sanity check on stored output statistics
        self.assertIsInstance(
            self.model.output_mean,
            (float, np.floating),
            "Expected output mean to be a float, got"
            f" {type(self.model.output_mean)}",
        )

        self.assertIsInstance(
            self.model.output_std,
            (float, np.floating),
            "Expected output standard deviation to be a float, got"
            f" {type(self.model.output_std)}",
        )

        self.assertGreater(
            self.model.output_std,
            0.0,
            "output standard deviation should be positive",
        )

        # Test normalization on a small subset of Y
        Y_subset = self.Y[:5]
        Y_normalized = (
            Y_subset - self.model.output_mean
        ) / self.model.output_std
        Y_denormalized = self.model.denormalize_output(Y_normalized)

        # Check that the original Y is recovered
        self.assertTrue(
            np.allclose(Y_denormalized, Y_subset, atol=1e-8),
            f"Expected normalization {Y_subset}, got {Y_denormalized}."
            f" Normalization failed.\nOriginal: {Y_subset}, Denormalized:"
            f" {Y_denormalized}",
        )



# Test parsing
class TestParseTestRuns(unittest.TestCase):

    def _make_run_strings(self):
        """Return two minimal run strings in the format expected by parse_test_runs."""
        return [
            "RunID=0111 ZwickyDays=10 q=8.0 chiA=0.0,0.0,-0.8 chiB=0.0,0.0,0.8 "
            "D0=14.5189208984 Omega0=0.0164280601103 adot0=-2.39687233838e-05",
            "RunID=0449 ZwickyDays=10 q=5.31 chiA=0.0,0.0,0.5 chiB=0.0,0.0,-0.5 "
            "D0=15.855651855 Omega0=0.0146017859056 adot0=-2.369980335e-05",
        ]

    def test_returns_dataframe(self):
        df = parse_test_runs(self._make_run_strings())
        self.assertIsInstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self):
        # One row per run string
        df = parse_test_runs(self._make_run_strings())
        self.assertEqual(len(df), 2)

    def test_required_columns_present(self):
        # Test that all columns needed by apply_gpr_corrections are present
        df = parse_test_runs(self._make_run_strings())
        for col in ["name", "initial_separation", "mass_ratio",
                    "S1x", "S1y", "S1z", "S2x", "S2y", "S2z",
                    "spec_pn_guess_omega", "spec_pn_guess_adot"]:
            self.assertIn(col, df.columns)

    def test_name_has_test_prefix(self):
        # Test that parse_test_runs prepends "test_" to the RunID for traceability
        df = parse_test_runs(self._make_run_strings())
        self.assertTrue(df["name"].iloc[0].startswith("test_"))

    def test_correct_mass_ratio(self):
        df = parse_test_runs(self._make_run_strings())
        self.assertAlmostEqual(df["mass_ratio"].iloc[0], 8.0)

    def test_correct_omega0(self):
        df = parse_test_runs(self._make_run_strings())
        self.assertAlmostEqual(df["spec_pn_guess_omega"].iloc[0], 0.0164280601103)

    def test_unknown_fields_ignored(self):
        df = parse_test_runs(self._make_run_strings())
        self.assertNotIn("ZwickyDays", df.columns)

    def test_spin_components_parsed_correctly(self):
        df = parse_test_runs(self._make_run_strings())
        self.assertAlmostEqual(df["S1z"].iloc[0], -0.8)
        self.assertAlmostEqual(df["S2z"].iloc[0],  0.8)


# Test application of GPR corrections

class TestApplyGprCorrections(unittest.TestCase):

    def setUp(self):
        self.df = make_df()
        # Train simple GPR models for omega and adot using separation as input
        input_columns = [
            "initial_separation", "mass_ratio",
            "S1x", "S1y", "S1z",
            "S2x", "S2y", "S2z",
        ]
        X = self.df[input_columns].values
        Y_omega = self.df["initial_orbital_frequency"].values
        Y_adot  = self.df["initial_adot"].values
        # Train one model per target variable
        self.model_omega, self.likelihood_omega = train_gpr_model(X, Y_omega)
        self.model_adot,  self.likelihood_adot  = train_gpr_model(X, Y_adot)

    def tearDown(self):
        plt.close("all")

    def test_returns_dataframe(self):
        result = apply_gpr_corrections(
            self.df.copy(),
            self.model_omega, self.likelihood_omega,
            self.model_adot,  self.likelihood_adot,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_correction_columns_added(self):
        # Test that all four output columns are present after corrections are applied
        result = apply_gpr_corrections(
            self.df.copy(),
            self.model_omega, self.likelihood_omega,
            self.model_adot,  self.likelihood_adot,
        )
        for col in ["delta_pred_omega", "delta_pred_adot",
                    "gpr_corrected_omega", "gpr_corrected_adot"]:
            self.assertIn(col, result.columns)

    def test_corrected_omega_is_pn_plus_delta(self):
        # Test that gpr_corrected_omega equals spec_pn_guess_omega + delta_pred_omega
        df = self.df.copy()
        result = apply_gpr_corrections(
            df,
            self.model_omega, self.likelihood_omega,
            self.model_adot,  self.likelihood_adot,
        )
        expected = result["spec_pn_guess_omega"] + result["delta_pred_omega"]
        np.testing.assert_allclose(
            result["gpr_corrected_omega"].values, expected.values, rtol=1e-10
        )

    def test_corrected_adot_is_pn_plus_delta(self):
        # Test that gpr_corrected_adot equals spec_pn_guess_adot + delta_pred_adot
        df = self.df.copy()
        result = apply_gpr_corrections(
            df,
            self.model_omega, self.likelihood_omega,
            self.model_adot,  self.likelihood_adot,
        )
        expected = result["spec_pn_guess_adot"] + result["delta_pred_adot"]
        np.testing.assert_allclose(
            result["gpr_corrected_adot"].values, expected.values, rtol=1e-10
        )


# Test save_gpr_corrected
class TestSaveGprCorrected(unittest.TestCase):

    def setUp(self):
        self.df = make_df().copy()
        # Reformat names and add the corrected columns save_gpr_corrected expects
        self.df["name"]              = [f"test_{n.replace('/Lev3','')}" for n in self.df["name"]]
        self.df["gpr_corrected_omega"] = self.df["initial_orbital_frequency"]
        self.df["gpr_corrected_adot"]  = self.df["initial_adot"]
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _output(self):
        """Return a writable output path inside the temporary directory."""
        return str(Path(self.tmp_dir.name) / "corrected_values.txt")

    def test_file_is_created(self):
        save_gpr_corrected(self.df, self._output())
        self.assertTrue(os.path.exists(self._output()))

    def test_correct_number_of_lines(self):
        save_gpr_corrected(self.df, self._output())
        with open(self._output()) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), len(self.df))

    def test_lines_contain_required_fields(self):
        # Test that every line contains all 8 of the required fields
        save_gpr_corrected(self.df, self._output())
        with open(self._output()) as f:
            first_line = f.readline()
        for field in ["RunID=", "ZwickyDays=", "q=", "chiA=", "chiB=",
                      "D0=", "Omega0=", "adot0="]:
            self.assertIn(field, first_line)

    def test_runid_has_no_test_prefix(self):
        # Test that save_gpr_corrected strips the "test_" prefix added by parse_test_runs
        save_gpr_corrected(self.df, self._output())
        with open(self._output()) as f:
            first_line = f.readline()
        self.assertNotIn("test_", first_line)

    def test_custom_omega_col(self):
        # Test that passing a non-default omega column name works
        self.df["my_omega"] = self.df["initial_orbital_frequency"] * 1.01
        save_gpr_corrected(self.df, self._output(), omega_col="my_omega")
        self.assertTrue(os.path.exists(self._output()))


# Test Leave one out cross validation

class TestLooCrossval(unittest.TestCase):

    def setUp(self):
        self.df = make_df()
        input_columns = [
            "initial_separation", "mass_ratio",
            "S1x", "S1y", "S1z",
            "S2x", "S2y", "S2z",
        ]
        self.X = self.df[input_columns].values
        self.Y = self.df["initial_orbital_frequency"].values

    def tearDown(self):
        plt.close("all")

    def test_returns_five_outputs(self):
        # Test that function returns (predictions, uncertainties, rmse, mase, r2)
        result = loo_crossval(
            self.X, self.Y,
            train_gpr_model, predict_with_gpr_model,
            target_name="omega"
        )
        self.assertEqual(len(result), 5)

    def test_predictions_shape(self):
        # One prediction per data point
        preds, _, _, _, _ = loo_crossval(
            self.X, self.Y,
            train_gpr_model, predict_with_gpr_model,
            target_name="omega"
        )
        self.assertEqual(preds.shape, self.Y.shape)

    def test_uncertainties_are_non_negative(self):
        # Test that GP predictive uncertainties are non-negative
        _, uncertainties, _, _, _ = loo_crossval(
            self.X, self.Y,
            train_gpr_model, predict_with_gpr_model,
            target_name="omega"
        )
        self.assertTrue(np.all(uncertainties >= 0))

    def test_r_squared_between_zero_and_one(self):
        # Test that R^2 is a squared correlation and lies in [0, 1]
        _, _, _, _, r2 = loo_crossval(
            self.X, self.Y,
            train_gpr_model, predict_with_gpr_model,
            target_name="omega"
        )
        self.assertGreaterEqual(r2, 0.0)
        self.assertLessEqual(r2, 1.0)


# Test plot_loo_residuals
# No GPR calls

class TestPlotLooResiduals(unittest.TestCase):

    def tearDown(self):
        plt.close("all")

    def test_residuals_computed_correctly(self):
        # Test that residual = true - predicted for each point
        Y     = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = np.array([1.1, 1.9, 3.1, 3.8, 5.2])
        residuals = plot_loo_residuals(Y, preds, show=False)
        np.testing.assert_allclose(residuals, Y - preds, rtol=1e-10)

    def test_returns_array_of_correct_shape(self):
        # Test that output shape matches the input arrays
        Y     = np.array([1.0, 2.0, 3.0])
        preds = np.array([1.0, 2.0, 3.0])
        residuals = plot_loo_residuals(Y, preds, show=False)
        self.assertEqual(residuals.shape, Y.shape)

    def test_perfect_predictions_give_zero_residuals(self):
        # Test that if the predictions match the truth exactly, all the residuals are 0
        Y     = np.array([1.0, 2.0, 3.0, 4.0])
        preds = Y.copy()
        residuals = plot_loo_residuals(Y, preds, show=False)
        np.testing.assert_allclose(residuals, 0.0, atol=1e-10)


# Test load_gpr_checkpoint

class TestLoadGprCheckpoint(unittest.TestCase):

    def setUp(self):
        """Train a small model and save a checkpoint for load_gpr_checkpoint to read."""
        import gpytorch
        from SimulationSupport.gpr import GPRegressionModel

        self.tmp_dir   = tempfile.TemporaryDirectory()
        self.ckpt_path = str(Path(self.tmp_dir.name) / "test_model.pt")

        df             = make_df()
        input_columns  = ["initial_separation", "mass_ratio",
                         "S1x", "S1y", "S1z", "S2x", "S2y", "S2z"]
        X              = df[input_columns].values
        Y              = df["initial_orbital_frequency"].values

        model, likelihood = train_gpr_model(X, Y)

        # Save checkpoint in the format expected by load_gpr_checkpoint
        torch.save({
            "model_state_dict":      model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "metadata": {
                "input_features": input_columns,
            },
            "normalization": {
                "input_mean":  model.input_mean.tolist(),
                "input_std":   model.input_std.tolist(),
                "output_mean": float(model.output_mean),
                "output_std":  float(model.output_std),
            },
        }, self.ckpt_path)

    def tearDown(self):
        self.tmp_dir.cleanup()
        plt.close("all")

    def test_returns_three_objects(self):
        # Test that function returns (model, likelihood, meta)
        result = load_gpr_checkpoint(self.ckpt_path)
        self.assertEqual(len(result), 3)

    def test_model_is_in_eval_mode(self):
        # Test that model.training is False
        model, _, _ = load_gpr_checkpoint(self.ckpt_path)
        self.assertFalse(model.training)

    def test_likelihood_is_in_eval_mode(self):
        _, likelihood, _ = load_gpr_checkpoint(self.ckpt_path)
        self.assertFalse(likelihood.training)

    def test_metadata_contains_input_features(self):
        # Test that metadata stores the feature list so users know what columns to pass
        _, _, meta = load_gpr_checkpoint(self.ckpt_path)
        self.assertIn("input_features", meta)

    def test_normalization_stats_restored(self):
        # Test that all four normalization stats are not None after loading
        model, _, _ = load_gpr_checkpoint(self.ckpt_path)
        self.assertIsNotNone(model.input_mean)
        self.assertIsNotNone(model.input_std)
        self.assertIsNotNone(model.output_mean)
        self.assertIsNotNone(model.output_std)

    def test_model_can_predict_after_loading(self):
        # Test that the loaded model produces valid predictions
        model, likelihood, meta = load_gpr_checkpoint(self.ckpt_path)
        df = make_df()
        X_test = df[meta["input_features"]].values[:3]
        preds, stds = predict_with_gpr_model(X_test, model, likelihood)
        self.assertEqual(len(preds), 3)
        self.assertTrue(np.all(stds >= 0))

if __name__ == "__main__":
    unittest.main(verbosity=2)
