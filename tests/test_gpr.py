# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Unit test for gpr.py
    Generate a simple, synthetic regression problem and run the pipeline
    Assert that the shapes are reasonable, uncertainties are positive, and
    that the mean predictions are within a reasonable range of the true values
"""

import shutil
import unittest
from pathlib import Path

import numpy as np
from gpr import (
    predict_with_gpr_model,
    run_gpr_pipeline,
    train_gpr_model,
)


class TestGPRLibrary(unittest.TestCase):
    # Set up and prepare test directory and file paths
    def setUp(self):
        self.test_dir = Path("SimulationSupport") / "gpr_library_tests"

        # Clean up any existing test directory and create a new one
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.X, self.Y = self.make_data()

    # Clean up and remove test directory after tests are done
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @staticmethod
    def make_data(n=50, noise_lev=0.05):
        """
        Generate synthetic regression dataset -
        points sampled from a noisy sine curve:
        y = sin(5 * x) + Gaussian noise.
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
        X, Y = self.X, self.Y  # fake dataset

        # Train the GPR model
        model, likelihood = train_gpr_model(X, Y)

        # Make predictions
        Y_pred, Y_std = predict_with_gpr_model(X, model, likelihood)

        # Check shapes
        self.assertEqual(
            Y_pred.shape,
            Y.shape,
            f"Expected prediction shape {Y.shape}, got {Y_pred.shape}",
        )

        self.assertEqual(
            Y_std.shape,
            Y.shape,
            f"Expected standard deviation shape {Y.shape}, got {Y_std.shape}",
        )

        # Check uncertainties are positive
        self.assertTrue(
            np.all(Y_std >= 0),
            f"Some of the predicted standard deviations are negative",
        )

        # Check that uncertainties are nontrivial
        self.assertTrue(np.any(Y_std > 1e-10), "All uncertainties are zero")

    def test_run_gpr_pipeline(self):
        """
        Test the full GPR pipeline function.
        Check that outputs are well correlated with true values.
        """
        X, Y = self.X, self.Y

        # Run the full GPR pipeline
        model, likelihood, Y_pred = run_gpr_pipeline(
            X, Y, target_name="test", plot=False, silent=True
        )

        corr = np.corrcoef(Y, Y_pred)[0, 1]

        self.assertGreater(
            corr, 0.9, f"Expected correlation > 0.9, got {corr:.3f}"
        )

    def test_normalization_stored_and_applied(self):
        """
        Test that the normalization parameters are stored and applied correctly:
        - input_mean and input_std have correct shapes
        - normalized inputs, X, have mean of ~0 and std of ~1  per feature
        """
        X, Y = self.X, self.Y
        model, likelihood = train_gpr_model(X, Y)

        # Check shapes of stored statistics
        exp_shape = (X.shape[1],)

        self.assertEqual(
            model.input_mean.shape,
            exp_shape,
            f"Expected input_mean shape {exp_shape}, got"
            f" {model.input_mean.shape}",
        )

        self.assertEqual(
            model.input_std.shape,
            exp_shape,
            f"Expected input_std shape {exp_shape}, got"
            f" {model.input_std.shape}",
        )

        # Recreate normalized X using the stored statistics
        normalized_X = (X - model.input_mean) / model.input_std

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
        denormalize_output, ie that it is correctly undoing the normalization.
        """
        X, Y = self.X, self.Y
        model, likelihood = train_gpr_model(X, Y)

        # Sanity check on stored output statistics
        self.assertIsInstance(
            model.output_mean,
            (float, np.floating),
            "Expected output mean to be a float, got"
            f" {type(model.output_mean)}",
        )

        self.assertIsInstance(
            model.output_std,
            (float, np.floating),
            "Expected output standard deviation to be a float, got"
            f" {type(model.output_std)}",
        )

        self.assertGreater(
            model.output_std,
            0.0,
            "output standard deviation should be positive",
        )

        # Test normalization on a small subset of Y
        Y_subset = Y[:5]
        Y_normalized = (Y_subset - model.output_mean) / model.output_std
        Y_denormalized = model.denormalize_output(Y_normalized)

        # Check that the original Y is recovered
        self.assertTrue(
            np.allclose(Y_denormalized, Y_subset, atol=1e-8),
            f"Expected normalization {Y_subset}, got {Y_denormalized}."
            f" Normalization failed.\nOriginal: {Y_subset}, Denormalized:"
            f" {Y_denormalized}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
