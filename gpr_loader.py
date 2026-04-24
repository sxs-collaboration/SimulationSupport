# Distributed under the MIT License.
# See LICENSE.txt for details.

# Loads saved GPR models and runs inference on new binary black hole simulations. No training is done here. 
# How to use:
# 1.for a single simulation (metadata string):
#   python gpr_loader.py --metadata "RunID=0111 ZwickyDays=176 q=8.0 chiA=-0.555366763183,-0.575801728447,-0.00249659459858
#                                    chiB=-0.0516668737278,0.102310708412,0.791746644364 D0=14.5189208984 Omega0=0.0164280601103
#                                    adot0=-2.39687233838e-05",
# 2. multiple simulations (one metadata string per line from a text file)
#   python gpr_loader.py --file my_runs.txt
# 3. specify custom model paths (optional)
#   python gpr_loader.py --file my_runs.txt --model_omega path/to/gpr_model_omega.pth --model_adot path/to/gpr_model.adot.pth

import os
import argparse
import numpy as np

# Import torch and related libraries
import torch
import gpytorch # GP library built on PyTorch

# Import GPR library
import importlib
import GPR_library as gpr
from GPR_library import GPRegressionModel

# Model Loading
def load_gpr_checkpoint(ckpt_path):
    """
    Loads a saved GPR checkpoint file from disk.
    Restores model weights, likelihood, normalization parameters,
    and (optionally) the raw training data.

    Args:
        ckpt_path (str): path to the saved .pth checkpoint file

    Returns:
        model (GPRegressionModel): GPR model in evaluation mode with
            trained weights loaded
        likelihood (gpytorch.likelihoods.GaussianLikelihood): likelihood
            in evaluation mode
        meta (dict): metadata dictionary containing the input_features,
            base_column, etc.
    """

    # Open, read, and load the saved checkpoint file into memory as a
    # Python dictionary
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Unpack metadata - describes what the model was trained on
    # (features, base_column, etc.)
    meta = ckpt["metadata"]
    features = meta["input_features"]
    D = len(features)  # Count number of input features

    # Create dummy input and output tensors as placeholders to construct the
    # model object correctly. These get replaced with the trained values
    # saved in the checkpoint later in model.load_state_dict and
    # likelihood.load_state_dict
    dummy_x = torch.zeros(1, D) # ensures the model knows there are D input features
    dummy_y = torch.zeros(1)    # ensures scalar output

    # Initialize likelihood and model with the right structure
    # likelihood represents the assumed noise model of the data
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # construct the GP model object
    model = gpr.GPRegressionModel(dummy_x, dummy_y, likelihood)

    # Load the trained parameters back into the model and likelihood
    model.load_state_dict(ckpt["model_state_dict"])
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    # Restore the normalization statistics used during training
    # so predictions are returned in the correct physical scale instead of
    # in raw standardized numbers
    norm = ckpt["normalization"]
    model.set_normalization(
        input_mean=np.array(norm["input_mean"]), # mean of the training features
        input_std=np.array(norm["input_std"]),   # SD of the training features
        output_mean=norm["output_mean"],         # mean of the training targets (deltas)
        output_std=norm["output_std"],           # SD of the training targets (deltas)
    )

    # Put model and likelihood into evaluation mode (for inference only)
    model.eval()
    likelihood.eval()

    return model, likelihood, meta

# Inference
def run_inference(metadata_strings, model_omega_pth, model_adot_pth):
    """
    Runs GPR inference on a list of metadata strings.

    Args:
        - metadata_strings (list of str): each string is one simulation's metadata
            in SpEC format, e.g.: "RunID =0111 q=8.0 chiA=... chiB=... D0=...
            Omega0=... adot0=..."
        - model_omega_pth (str): path to the saved GPR model checkpoint for Omega0
        - model_adot_pth (str): path to the saved GPR model checkpoint for adot0
    
    Returns:
        - df_test (pd.DataFrame): DataFrame with input parameters, predicted corrections,
            uncertainties, and GPR-corrected Omega0 and adot0 values.
    """
    # Parse metadata strings into a DataFrame
    df_test=gpr.parse_test_runs(metadata_strings)
    # Load models
    model_omega, likelihood_omega, meta_omega = load_gpr_checkpoint(model_omega_pth)
    model_adot, likelihood_adot, meta_adot = load_gpr_checkpoint(model_adot_pth)

    features = meta_omega["input_features"]

    # Extract input features
    X_test = df_test[features].values

    # Predict corrections (deltas) and uncertainties
    delta_omega_pred, omega_unc = gpr.predict_with_gpr_model(
        X_test, model_omega, likelihood_omega)
    delta_adot_pred,  adot_unc = gpr.predict_with_gpr_model(
        X_test, model_adot, likelihood_adot)

    # Store predictions
    df_test["delta_pred_omega"] = delta_omega_pred
    df_test["uncertainty_omega"] = omega_unc

    # Store uncertainties
    df_test["delta_pred_adot"]  = delta_adot_pred
    df_test["uncertainty_adot"] = adot_unc

    # Apply corrections to the PN baseline guesses
    df_test["gpr_corrected_omega"] = df_test["spec_pn_guess_omega"] + delta_omega_pred
    df_test["gpr_corrected_adot"]  = df_test["spec_pn_guess_adot"]  + delta_adot_pred

    return df_test

# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description ="Run GPR inference on new BBH simulations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input: either a single metadata string or a file of multiple strings
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--metadata", type=str,
        help="Single simulation metadata string in SpEC format.",
    )
    input_group.add_argument(
        "--file", type=str,
        help="Path to a text file with one metadata string per line.",
    )

    # Model paths (optional)
    parser.add_argument(
        "--model_omega", type=str, default="gpr_model_omega.pth",
        help = "Path to saved GPR model for Omega0 (default is gpr_model_omega.pth).",
    )
    parser.add_argument(
        "--model_adot", type=str, default="gpr_model_adot.pth",
        help = "Path to saved GPR model for adot0 (default: gpr_model_adot.pth).",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None,
        help = "Optional path to save results as a CSV file.",
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Load metadata strings
    if args.metadata:
        metadata_strings = [args.metadata]
    else:
        with open(args.file, "r") as f:
            metadata_strings = [line.strip() for line in f if line.strip()]

    # Run inference
    df_results = run_inference(metadata_strings, args.model_omega, args.model_adot)

    # Print results
    print("\nGPR Inference Results:")
    print("=" * 60)
    output_cols = [
        "name",
        "spec_pn_guess_omega", "gpr_corrected_omega", "uncertainty_omega",
        "spec_pn_guess_adot", "gpr_corrected_adot", "uncertainty_adot",
    ]
    print(df_results[output_cols].to_string(index=False))

    # Optional: save to CSV
    if args.output:
        df_results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
