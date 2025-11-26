# Distributed under the MIT License.
# See LICENSE.txt for details.

"""
Gaussian Process Regression Machine Learning Function Library.
Contains all functions necessary to run the GPR Model used to predict better
low-eccentricity orbital parameter initial guesses.
"""

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch


class GPRegressionModel(gpytorch.models.ExactGP):
    """
    This class derives from gpytorch.models.ExactGP an infinite number of basis functions;
    GP is non-parametric and models functions globally; is limited only by training points

    Exact GP with a mixture of RBF and Matern kernels, a linear mean function,
    and normalization capabilities for inputs and outputs.

    Args:
        train_x (torch.Tensor): Training input data
        train_y (torch.Tensor): Training targets
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Gaussian noise likelihood
    """

    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        # Supports all dimensions (ie GPR can be run from 1-8 dimensions)
        input_dim = train_x.shape[1] if train_x.dim() > 1 else 1

        # Define base kernels
        # We use a mixture of the RBF (smooth global trends) + Matern-5/2 (local
        # variations) kernels. The smoothness parameter of the Matern Kernel is ν=5/2 (nu=2.5).
        # The number 2.5 was chosen because it is exactly twice mean-square differentiable.
        # ν=1/2 or ν=3/2 would be too rough; RBF alone (ν->infinity) would be overconfident
        # near merger where the function has sharper local changes.

        # ard_num_dims enables Automatic Relevance Determination: each input dimension gets
        # its own length scale, allowing the GP to learn which parameters most strongly influence
        # the waveform
        self.rbf_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
        self.matern_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=input_dim
        )

        # Wrap each kernel with a scale kernel - introduces learnable scaling factor
        self.scaled_rbf = gpytorch.kernels.ScaleKernel(self.rbf_kernel)
        self.scaled_matern = gpytorch.kernels.ScaleKernel(self.matern_kernel)

        # Combine kernels - the sum of the kernels allows the model to capture more complex
        # behavior than either kernel alone would
        self.covar_module = self.scaled_rbf + self.scaled_matern

        # Mean function - use linear mean instead of default 0 mean
        # Remove hardcoding of model to expect 1D inputs and therefore, matrix mismatch if you pass 2D inputs
        self.mean_module = gpytorch.means.LinearMean(input_size=input_dim)

        # Normalization parameters - store the mean and std of the inputs and outputs
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None

    def set_normalization(self, input_mean, input_std, output_mean, output_std):
        """
        Store normalization parameters in the model.
        """
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std

    def normalize_input(self, X):
        """
        Normalize input using stored parameters. Scale input to zero mean and unit variance.
        """
        return (X - self.input_mean) / self.input_std

    def denormalize_output(self, Y_normalized):
        """
        Denormalize output using stored parameters (converts normalized output back to original scale).
        """
        return (Y_normalized * self.output_std) + self.output_mean

    def forward(self, x, normalize_input=False):
        """
        Forward pass with optional input normalization.

        Args:
            x (torch.Tensor): Input data
            normalize_input (bool): If True, normalize x using stored parameters.

        Returns:
            gpytorch.distributions.MultivariateNormal: Distribution for the input.
        """
        if normalize_input:
            x = self.normalize_input(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# GPR training function
def train_gpr_model(raw_X, raw_Y):
    """
    Train a GPR model with normalization parameters stored in the model.

    Args:
        raw_X (numpy.ndarray): Raw input data
        raw_Y (numpy.ndarray): Raw output data

    Returns:
        GPRegressionModel: Trained model with the normalization parameters stored
        gpytorch.likelihoods.GaussianLikelihood: Likelihood for the model
    """
    # Compute normalization parameters
    input_mean = raw_X.mean(axis=0)
    input_std = raw_X.std(axis=0)
    output_mean = raw_Y.mean()
    output_std = raw_Y.std()

    # Normalize data column-wise; needed for all dimension > 1
    normalized_X = (raw_X - input_mean) / input_std
    normalized_Y = (raw_Y - output_mean) / output_std

    # Ensure X is proper dimension
    if normalized_X.ndim == 1:
        normalized_X = normalized_X.reshape(-1, 1)

    # Convert to PyTorch tensors
    train_X = torch.from_numpy(normalized_X).float()
    train_Y = torch.from_numpy(normalized_Y).float()

    # Define the likelihood and the model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_X, train_Y, likelihood)

    # Store normalization parameters in the model
    model.set_normalization(input_mean, input_std, output_mean, output_std)

    # Set model to training mode
    model.train()
    likelihood.train()

    # Use Adam optimizer with initial learning rate of 0.05; this value is commonly chosen
    # as a starting point for Adam with GPyTorch models because it is large enough
    # for fast convergence, and small enough to avoid overshooting the loss minimum

    # The Adam optimizer is chosen over Standard Gradient Descent (SGD) because it adapts
    # per-parameter learning rates; important when input dimensions have different scales
    # (e.g. mass ratio vs spin components); works well with GPyTorch's marginal log-likelihood.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Add learning rate scheduler:
    # Reduce LR by a factor of 0.5 when validation loss stops improving;
    # this value is chosen because it is aggressive enough to escape plateaus
    # when progress stalls, but large enough to avoid stopping training completely;
    # Helps fine-tune hyperparameters (kernel length scales, noise) in the later
    # stages of training without manually scheduling the learning rate.

    # Wait 5 epochs before reducing the LR if there is no improvement; this number
    # is chosen because it is short enough to react to plateaus, but large enough to
    # avoid reacting to noise
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Track the best model to prevent overfitting
    best_loss = float("inf")  # Initialize best loss as infinity
    best_state = None  # Placeholder for best model state

    # Training loop for 200 iterations; 200 was chosen empirically for this dataset.
    # In practice, early stopping terminates training once the loss
    # plateaus, so the exact value isn't critical. If utilizing a larger
    # dataset where more iterations may be needed, increase this value if the loss
    # is still decreasing at iteration 200.
    # If early stopping instead triggers very early, decrease this value.
    for i in range(200):
        # Ensure model is in training mode at the start of each iteration
        model.train()
        likelihood.train()

        # Reset gradients from the previous iteration
        optimizer.zero_grad()

        # Compute model output
        output = model(train_X)

        # Compute negative log-likelihood loss
        loss = -mll(output, train_Y)

        # Backpropagation: compute gradients of loss wrt model parameters
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Update and adjust the learning rate
        scheduler.step(loss)

        # Save the best model parameters (if current loss is the lowest)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = model.state_dict().copy()

    # Load the best model state after training
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, likelihood


# GPR prediction function
def predict_with_gpr_model(raw_X, model, likelihood):
    """
    Predict using the GPR model with stored normalization parameters.

    Args:
        raw_X (numpy.ndarray): Raw input data.
        model (GPRegressionModel): Trained model.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Likelihood
        for the model.

    Returns:
        numpy.ndarray: Predicted mean (denormalized).
        numpy.ndarray: Predicted standard deviation (denormalized).
    """
    # Normalize the input using the model's stored parameters
    normalized_X = (raw_X - model.input_mean) / model.input_std
    X_tensor = torch.from_numpy(normalized_X).float()

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions
    with torch.no_grad():
        observed_pred = likelihood(model(X_tensor))

    # Denormalize the predictions
    mean_normalized = observed_pred.mean.numpy()
    stddev_normalized = observed_pred.variance.sqrt().numpy()

    mean_denormalized = model.denormalize_output(mean_normalized)
    stddev_denormalized = stddev_normalized * model.output_std

    return mean_denormalized, stddev_denormalized


# GPR pipeline function - runs the entire process - including training, predicting, plotting - and outputs performance metrics
# This function encompasses previous functions defined above: train_gpr_model and predict_with_gpr_model and runs them together
def run_gpr_pipeline(X, Y, target_name="target", plot=True, silent=False):
    """
    Train a GPR model on (X, Y), predict on X, plot, and report metrics for a given target.

    Args:
        X (np.ndarray): Input data.
        Y (np.ndarray): Target output deltas.
        target_name (str): For labeling plots & output.
        plot (bool): whether to produce correlation plots.
        silent (bool): whether to suppress print statements entirely.

    Returns:
        model
        likelihood
        Y_pred
        uncertainties
    """

    # Train GPR model
    model, likelihood = train_gpr_model(X, Y)

    # Make predictions
    Y_pred, uncertainties = predict_with_gpr_model(X, model, likelihood)

    # If specified, create correlation plot and compute metrics
    if plot:
        plt.figure(figsize=(8, 6))
        plt.scatter(Y, Y_pred, alpha=0.6, s=20)

        # Make perfect correlation line (y = x)
        min_val = min(Y.min(), Y_pred.min())
        max_val = max(Y.max(), Y_pred.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Correlation",
        )

        # Labels and formatting
        plt.xlabel(f"ΔTrue {target_name}", fontsize=12)
        plt.ylabel(f"GPR Predicted Δ{target_name}", fontsize=12)
        plt.title(
            f"GPR Predictions vs True Values ({target_name})", fontsize=14
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Calculate and display metrics
        corr = np.corrcoef(Y, Y_pred)[0, 1]
        r2 = corr**2
        rmse = np.sqrt(np.mean((Y - Y_pred) ** 2))
        mae = np.mean(np.abs(Y - Y_pred))
        metrics_text = f"R² = {r2:.4f}\nRMSE = {rmse:.8f}\nMAE = {mae:.8f}"
        plt.text(
            0.95,
            0.05,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            verticalalignment="bottom",
            horizontalalignment="right",
        )

        plt.tight_layout()
        plt.show()

    if not silent:
        # Print performance metrics regardless of plotting
        print(f"R²: goal: > 0.95 excellent, > 0.90 good, < 0.70 poor")
        print(f"RMSE goal: < 1 % of target range, lower is better")
        print(f"MAE goal: < 1 % of target range, lower is better")

    return model, likelihood, Y_pred
