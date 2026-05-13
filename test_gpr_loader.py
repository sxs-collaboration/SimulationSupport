"""
test_gpr_loader.py

Unit tests for gpr_loader.py using Python built-in unittest.

Run with:
    python -m unittest test_gpr_loader.py -v

No real model files or Julia/SXS dependencies are required - all heavy
dependencies are mocked.
"""

import unittest
import numpy as np
import pandas as pd
import sys
from unittest.mock import patch, MagicMock

# Mock heavy dependencies before importing the module being tested
# This prevents Julia, sxs, qgrid, gpytorch, torch, etc. from being imported
# during the test

sys.modules["qgrid"]                   = MagicMock()
sys.modules["qgridnext"]               = MagicMock()
sys.modules["sxs"]                     = MagicMock()
sys.modules["sxs.julia"]               = MagicMock()
sys.modules["sxs.julia.PostNewtonian"] = MagicMock()
sys.modules["GPR_library"]             = MagicMock()
sys.modules["plotly"]                  = MagicMock()
sys.modules["plotly.graph_objects"]    = MagicMock()
sys.modules["torch"]                   = MagicMock()
sys.modules["gpytorch"]                   = MagicMock()

import gpr_loader

# Test load_gpr_checkpoint
class TestLoadGPRCheckpoint(unittest.TestCase):
    def setUp(self):
        """Build a fake checkpoint dictionary that imitates what torch.load would return. """
        self.mock_checkpoint = {
            "metadata":{
                "input_features": [
                    "mass_ratio",
                    "initial_separation",
                    "S1x", "S1y", "S1z",
                    "S2x", "S2y", "S2z",
                ],
                "base_column": "spec_pn_guess_omega",
            },
            "model_state_dict":{},
            "likelihood_state_dict": {},
            "normalization":{
                "input_mean":  [0.0],
                "input_std":   [1.0],
                "output_mean": 0.0,
                "output_std":  1.0,
            },
        }
    
    def test_returns_model_likelihood_meta(self):
        """Test that load_gpr_checkpoint returns the model, likelihood, and metadata."""
        mock_model      = MagicMock()
        mock_likelihood = MagicMock()

        with patch("torch.load", return_value = self.mock_checkpoint), \
             patch("gpr_loader.gpr.GPRegressionModel", return_value=mock_model), \
             patch("gpytorch.likelihoods.GaussianLikelihood", return_value=mock_likelihood):
             model, likelihood, meta = gpr_loader.load_gpr_checkpoint("fake_path.pth")
             self.assertEqual(meta, self.mock_checkpoint["metadata"])
             mock_model.eval.assert_called_once()
             mock_likelihood.eval.assert_called_once()

    def test_normalization_is_set(self):
        """Test that load_gpr_checkpoint calls set_normalization correctly."""
        mock_model      = MagicMock()
        mock_likelihood = MagicMock()

        with patch("torch.load", return_value = self.mock_checkpoint), \
             patch("gpr_loader.gpr.GPRegressionModel", return_value=mock_model), \
             patch("gpr_loader.gpytorch.likelihoods.GaussianLikelihood", return_value=mock_likelihood):

             gpr_loader.load_gpr_checkpoint("fake_path.pth")

             mock_model.set_normalization.assert_called_once()
             call_kwargs = mock_model.set_normalization.call_args.kwargs
             self.assertIn("input_mean",  call_kwargs)
             self.assertIn("input_std",   call_kwargs)
             self.assertIn("output_mean", call_kwargs)
             self.assertIn("output_std",  call_kwargs)

    def test_model_loaded_in_eval_mode(self):
        """Both model and likelihood should be put into eval mode."""
        mock_model = MagicMock()
        mock_likelihood = MagicMock()

        with patch("torch.load", return_value=self.mock_checkpoint),\
             patch("gpr_loader.gpr.GPRegressionModel", return_value=mock_model),\
             patch("gpr_loader.gpytorch.likelihoods.GaussianLikelihood",
                return_value=mock_likelihood):
            
            gpr_loader.load_gpr_checkpoint("fake_path.pth")

            mock_model.eval.assert_called_once()
            mock_likelihood.eval.assert_called_once()

# Test run_inference
class TestRunInference(unittest.TestCase):
    def setUp(self):
        """ Set up test data for run_inference tests."""
        self.sample_metadata_strings = [
            "RunID=0111 ZwickyDays=176 q=8.0 chiA=-0.555366763183,-0.575801728447,"
            "   -0.00249659459858 chiB=-0.0516668737278,0.102310708412,0.791746644364"
            "   D0=14.5189208984 Omega0=0.0164280601103 adot0=-2.39687233838e-05",
            "RunID=0271 ZwickyDays=138 q=7.16339643655 chiA=-0.247507893273,"
            "   0.666875889694,0.327541836441 chiB=-0.143315584105,0.537208136535,"
            "   -0.545488940029 D0=14.6237792969 Omega0=0.0162422365835 adot0=-2.43238483539e-05",
            "RunID=0449 ZwickyDays=92 q=5.31391992299 chiA=-0.676142027445,0.39956096948"
            "   ,-0.0775314966678 chiB=-0.339102219344,0.441481250187,-0.535621006618"
            "   D0=15.8556518555 Omega0=0.0146017859056 adot0=-2.369980335e-05",
        ]
        self.mock_meta = {
            "input_features": [
                "mass_ratio",
                "initial_separation",
                "S1x", "S1y", "S1z",
                "S2x", "S2y", "S2z"
            ],
            "base_column": "spec_pn_guess_omega",
        }
        self.mock_dataframe = pd.DataFrame({
            "name":                ["test_0111", "test_0271", "test_0449"],
            "initial_separation":  [14.518921, 14.623779, 15.855652],
            "spec_pn_guess_omega": [0.016428,  0.016242,  0.014602],
            "spec_pn_guess_adot":  [-0.000024, -0.000024, -0.000024],
            "mass_ratio":          [8.0, 7.163396, 5.313920],
            "S1x":                 [-0.555367, -0.247508, -0.676142],
            "S1y":                 [-0.575802,  0.666876,  0.399561],
            "S1z":                 [-0.002497,  0.327542, -0.077531],
            "S2x":                 [-0.051667, -0.143316, -0.339102],
            "S2y":                 [ 0.102311,  0.537208,  0.441481],
            "S2z":                 [ 0.791747, -0.545489, -0.535621],
            "initial_mass1":       [None, None, None],
            "initial_mass2":       [None, None, None],
            "eccentricity":        [None, None, None],
        })
        self.mock_model      = MagicMock()
        self.mock_likelihood = MagicMock()

    def test_output_columns_present(self):
        """Test that run_inference adds the GPR corrections and uncertainty columns correctly."""
        with patch("gpr_loader.load_gpr_checkpoint",
                   return_value = (self.mock_model, self.mock_likelihood, self.mock_meta)), \
             patch("gpr_loader.gpr.parse_test_runs",
                   return_value = self.mock_dataframe), \
             patch("gpr_loader.gpr.predict_with_gpr_model",
                   return_value = (np.zeros(3), np.zeros(3))):
             df = gpr_loader.run_inference(
                 self.sample_metadata_strings, "omega.pth", "adot.pth"
             )
             expected_columns = [
                 "delta_pred_omega", "uncertainty_omega",
                 "delta_pred_adot" , "uncertainty_adot" ,
                 "gpr_corrected_omega" , "gpr_corrected_adot",
             ]
             for col in expected_columns:
                 self.assertIn(col, df.columns)

    def test_correction_applied(self):
        """Test whether GPR-corrected values are in fact the sum of the PN baseline +
            the predicted deltas.
        """
        delta_omega = np.array([-0.000560, -0.000539, -0.000439])
        delta_adot  = np.array([ 0.000714, -0.000123, -0.000261]) # shape (3, )

        with patch("gpr_loader.load_gpr_checkpoint",
                   return_value = (self.mock_model, self.mock_likelihood, self.mock_meta)), \
             patch("gpr_loader.gpr.parse_test_runs",
                   return_value = self.mock_dataframe.copy()), \
             patch("gpr_loader.gpr.predict_with_gpr_model",
                   side_effect = [
                       (delta_omega, np.zeros(3)),
                       (delta_adot,  np.zeros(3)),
                   ]):
            df = gpr_loader.run_inference(
                self.sample_metadata_strings, "omega.pth", "adot.pth"
            )
            np.testing.assert_allclose(
                df["gpr_corrected_omega"].values,
                self.mock_dataframe["spec_pn_guess_omega"].values + delta_omega,
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                df["gpr_corrected_adot"].values,
                self.mock_dataframe["spec_pn_guess_adot"].values + delta_adot,
                rtol=1e-6,
            )
    
    def test_correct_number_rows(self):
        """Test that the output DataFrame has the same number of rows as the input list."""
        with patch("gpr_loader.load_gpr_checkpoint",
                   return_value=(self.mock_model, self.mock_likelihood, self.mock_meta)), \
             patch("gpr_loader.gpr.parse_test_runs",
                   return_value=self.mock_dataframe), \
             patch("gpr_loader.gpr.predict_with_gpr_model",
                   return_value=(np.zeros(3), np.zeros(3))):
 
            df = gpr_loader.run_inference(
                self.sample_metadata_strings, "omega.pth", "adot.pth"
            )
 
            self.assertEqual(len(df), len(self.sample_metadata_strings))

    def test_uncertainties_are_nonneg(self):
        """Test that the ncertainty values are always non-negative."""
        with patch("gpr_loader.load_gpr_checkpoint",
                   return_value=(self.mock_model, self.mock_likelihood, self.mock_meta)), \
             patch("gpr_loader.gpr.parse_test_runs",
                   return_value=self.mock_dataframe), \
             patch("gpr_loader.gpr.predict_with_gpr_model",
                   return_value=(np.zeros(3), np.array([0.0001, 0.0002, 0.0003]))):
 
            df = gpr_loader.run_inference(
                self.sample_metadata_strings, "omega.pth", "adot.pth"
            )
 
            self.assertTrue((df["uncertainty_omega"] >= 0).all())
            self.assertTrue((df["uncertainty_adot"]  >= 0).all())
    
    def test_features_used_correctly(self):
        """Test that predict_with_gpr_model receives a matrix with one column per feature."""
        with patch("gpr_loader.load_gpr_checkpoint",
                return_value=(self.mock_model, self.mock_likelihood, self.mock_meta)), \
            patch("gpr_loader.gpr.parse_test_runs",
                return_value = self.mock_dataframe.copy()), \
            patch("gpr_loader.gpr.predict_with_gpr_model", 
                return_value=(np.zeros(3), np.zeros(3))) as mock_predict:
                
            gpr_loader.run_inference(
                self.sample_metadata_strings, "omega.pth", "adot.pth"
            )
            args, _ = mock_predict.call_args
            X_passed = args[0]
            self.assertEqual(X_passed.shape[1], len(self.mock_meta["input_features"]))

# Test CLI argument parsing
class TestArgParsing(unittest.TestCase):

    def test_metadata_arg(self):
        """Test that --metadata flag is stored and --file defaults to None."""
        with patch("sys.argv", ["gpr_loader.py", "--metadata", "RunID=0111 q=8.0"]):
            args = gpr_loader.parse_args()
            self.assertEqual(args.metadata, "RunID=0111 q=8.0")
            self.assertIsNone(args.file)

    def test_file_arg(self):
        """Test that --file flag is stored and --metadata defaults to None."""
        with patch("sys.argv", ["gpr_loader.py", "--file", "my_runs.txt"]):
            args = gpr_loader.parse_args()
            self.assertEqual(args.file, "my_runs.txt")
            self.assertIsNone(args.metadata)
 
    def test_metadata_and_file_are_mutually_exclusive(self):
        """Test that if user passes both --metadata and --file, it raises SystemExit."""
        with patch("sys.argv", ["gpr_loader.py",
                                 "--metadata", "q=8.0",
                                 "--file", "runs.txt"]):
            with self.assertRaises(SystemExit):
                gpr_loader.parse_args()
 
    def test_no_input_raises_system_exit(self):
        """Test that if user doesn't pass --metadata or --file, it raises SystemExit."""
        with patch("sys.argv", ["gpr_loader.py"]):
            with self.assertRaises(SystemExit):
                gpr_loader.parse_args()
 
    def test_default_model_paths(self):
        """Test that the default model paths are gpr_model_omega.pth and gpr_model_adot.pth."""
        with patch("sys.argv", ["gpr_loader.py", "--metadata", "q=8.0"]):
            args = gpr_loader.parse_args()
            self.assertEqual(args.model_omega, "gpr_model_omega.pth")
            self.assertEqual(args.model_adot,  "gpr_model_adot.pth")
 
    def test_custom_model_paths(self):
        """Test that the custom model paths override the defaults."""
        with patch("sys.argv", [
                    "gpr_loader.py",
                    "--metadata",
                    "q=8.0",
                    "--model_omega",
                    "custom_omega.pth",
                    "--model_adot",
                    "custom_adot.pth",
        ]):
            args = gpr_loader.parse_args()
            self.assertEqual(args.model_omega, "custom_omega.pth")
            self.assertEqual(args.model_adot,  "custom_adot.pth")
 
    def test_output_arg(self):
        """Test that --output flag is stored correctly."""
        with patch("sys.argv", ["gpr_loader.py",
                                 "--metadata", "q=8.0",
                                 "--output", "results.csv"]):
            args = gpr_loader.parse_args()
            self.assertEqual(args.output, "results.csv")
 
    def test_output_defaults_to_none(self):
        """Test that --output defaults to None if not provided."""
        with patch("sys.argv", ["gpr_loader.py", "--metadata", "q=8.0"]):
            args = gpr_loader.parse_args()
            self.assertIsNone(args.output)

if __name__ == "__main__":
    unittest.main(verbosity=2)