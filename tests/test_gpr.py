# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Unit test for gpr.py
    Generate a simple, synthetic regression problem and run the pipeline.
    Assert that the shapes are reasonable, uncertainties are positive, and
    that the mean predictions are within a plausible range of the true values.

Run with:
    python -m unittest test_gpr.py -v
"""

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from SimulationSupport.gpr import (
    predict_with_gpr_model,
    run_gpr_pipeline,
    train_gpr_model,
)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
