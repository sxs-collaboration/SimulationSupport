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
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# import juliacall-related modules
from sxs.julia import PostNewtonian

# Normalize data using the standardization formula
def normalize_data(X, Y):
    """
    Standardizes features and targets. Treats X and Y
    as 1D arrays and reshapes them to 2D arrays. For multidimensional X
    (with shape (N, D)), it passes X directly without rehsaping and fits
    per column instead, which is the default behavior of StandardScaler.

    Args:
        X: input features (N, ) or (N, D)
        Y: targets (N, ) or (N, D)

    Returns:
        X_normalized: normalized input features
        Y_normalized: normalized target values
        scaler_X: fitted StandardScaler for X
        scaler_Y: fitted StandardScaler for Y
    """
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()

    # Reshape X and Y into 2D arrays before fitting
    # as StandardScaler expects shape (N, 1) for 1D inputs
    X_normalized = scaler_X.fit_transform(X.reshape(-1, 1))
    Y_normalized = scaler_Y.fit_transform(Y.reshape(-1, 1))

    return X_normalized, Y_normalized, scaler_X, scaler_Y


# Denormalize predictions by converting them back to their original scale.
# The model makes predictions in standardized space, but we need the real values
# in order to correctly interpret the results
def denormalize_predictions(pred_mean, pred_stddev, scaler_Y):
    """
    Map the standardized predictions back to the original scale.

    Args:
        pred_mean: mean of the predictions in standardized space
        pred_stddev: standard deviation of the predictions in standardized space
        scaler_Y: fitted StandardScaler for Y

    Returns:
        mean_unnormalized: mean of the predictions in the original scale
        stddev_unnormalized: standard deviation of the predictions in the original scale
    """
    # Extract the scalar standard deviation used during fitting.
    # This is needed in order to rescale the uncertainty separately from the mean
    scale = scaler_Y.scale_[0]

    # inverse_transform expects a 2D input; squeeze converts it back to 1D
    mean_unnormalized = scaler_Y.inverse_transform(
        pred_mean.reshape(-1, 1)
    ).squeeze()

    # Standard deviations scale linearly, so multiply directly
    # instead of using inverse_transform
    stddev_unnormalized = scale * pred_stddev

    return mean_unnormalized, stddev_unnormalized


# Compute PostNewtonian initial values
def omega_and_adot(r, q, chiA, chiB):
    """
    Calculates the orbital frequency and normalized rate of change of separation
    for a binary black hole system. Supports both scalar and vectorized inputs
    for performance optimization.

    Args:
        r: the separation(s) between the two black holes
        q: the mass ratio(s) of the two black holes, defined as M1/M2
        chiA: length-3 spin vector [chiA_x, chiA_y, chiA_z] OR (N, 3) array
            of spin vectors for the first black hole
        chiB: length-3 spin vector [chiB_x, chiB_y, chiB_z] OR (N, 3) array
            of spin vectors for the second black hole

    Returns:
        Omega: the orbital frequency (or frequencies) of the binary system
        adot: the rate of change of separation normalized by the separation,
            i.e., separation_dot / r
    """

    # Detect whether inputs are scalar (backward compatible path) or arrays (vectorized path)
    # to decide which code path to take.
    # ndim == 0 catches zero-dimensional numpy arrays (e.g. np.float64(1.0)),
    # which np.isscalar() doesn't always catch
    is_scalar = np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0)

    if is_scalar:
        # SCALAR PATH
        # Original single-system behavior for single values (backward compatible)
        # Ensure chiA and chiB are lists or arrays of length 3
        assert len(chiA) == 3, "chiA must have 3 elements"
        assert len(chiB) == 3, "chiB must have 3 elements"

        # Construct the 14 element BBH state vector expected by PostNewtonian.BBH:
        # [0-1] mass fractions M1 / (M1+M2), M2 / (M1+M2)
        # [2-4] chiA spin components
        # [5-7] chiB spin components
        # [8-13] initial orbital conditions (unit separation vector + unit velocity
        #        vector in the orbital plane)
        state = np.array(
            [
                q / (1.0 + q),  # mass fraction of the primary
                1.0 / (1.0 + q),  # mass fraction of the secondary
                *chiA,
                *chiB,
                1,
                0,
                0,
                0,
                1,
                0,  # initial orbital frame
            ]
        )
        # Ensure the state array has the correct length
        assert len(state) == 14, "State array must have 14 elements"

        # Initialize the post-Newtonian system
        pn = PostNewtonian.BBH(state)

        # Overwrite the default separation with the requested value r.
        # separation_inverse converts r to the internal coordinate used by the
        # PN integrator before setting state[12].
        pn.state[12] = PostNewtonian.separation_inverse(r, pn)

        # Normalize separation_dot by r to get adot = rdot/r (dimensionless rate)
        # Return orbital frequency and normalized separation rate
        return PostNewtonian.Omega(pn), PostNewtonian.separation_dot(pn) / r

    else:
        # VECTORIZED PATH
        # Process N systems in a loop. Still calls Julia once per row, but eliminates
        # the Python iterrows() overhead
        r = np.asarray(r)
        q = np.asarray(q)
        chiA = np.asarray(chiA)  # epected shape: (N, 3)
        chiB = np.asarray(chiB)  # expected shape: (N, 3)

        N = len(r)
        assert chiA.shape == (
            N,
            3,
        ), f"chiA must have shape ({N}, 3), got {chiA.shape}"
        assert chiB.shape == (
            N,
            3,
        ), f"chiB must have shape ({N}, 3), got {chiB.shape}"
        assert len(q) == N, f"q must have length   {N}.    , got {len(q)}"

        omegas = np.zeros(N)
        adots = np.zeros(N)

        for i in range(N):
            # Build the same 14-element state vector as the scalar path,
            # indexing into the i-th row of each array
            state = np.array(
                [
                    q[i] / (1.0 + q[i]),
                    1.0 / (1.0 + q[i]),
                    chiA[i, 0],
                    chiA[i, 1],
                    chiA[i, 2],
                    chiB[i, 0],
                    chiB[i, 1],
                    chiB[i, 2],
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]
            )
            pn = PostNewtonian.BBH(state)
            pn.state[12] = PostNewtonian.separation_inverse(r[i], pn)
            omegas[i] = PostNewtonian.Omega(pn)
            adots[i] = PostNewtonian.separation_dot(pn) / r[i]

        return omegas, adots


# Function to compute initial PN values
# Compute additional PN terms explicitly instead of relying on PostNewtonian.BBH
def omegaAndAdot(r, q, chiA, chiB, rPrime0):
    """
    Calculates PN orbital frequency and separation rate with explicit PN terms.
    Supports both scalar and vectorized inputs for performance optimization.

    Args:
        r (float or array): The separation(s) between the two black holes.
        q (float or array): The mass ratio(s) of the two black holes, defined as M1/M2.
        chiA (array-like): Length-3 spin vector OR (N, 3) array of spin vectors for the first black hole.
        chiB (array-like): Length-3 spin vector OR (N, 3) array of spin vectors for the second black hole.
        rPrime0 (float): Reference separation for logarithmic PN term.

    Returns:
        mega (float or array): The orbital frequency (or frequencies).
        adot (float or array): The normalized separation rate(s), i.e., (dr/dt) / r.
    """
    # Check whether inputs are scalars (backward compatible path) or arrays (vectorized path)
    # to decide which code path to take.
    # ndim == 0 catches zero dimensional numpy arrays (e.g. np.float64(1.0)),
    # which np.isscalar() doesn't always catch.
    is_scalar = np.isscalar(r) or (isinstance(r, np.ndarray) and r.ndim == 0)

    if is_scalar:
        # SCALAR PATH
        # Original behavior for single values (backward compatible)
        # Ensure chiA and chiB are numpy arrays
        chiA = np.array(chiA)
        chiB = np.array(chiB)

        # Unit vector along the orbital angular momentum (z-axis for aligned spins)
        LHat = np.array([0.0, 0.0, 1.0])

        # Project spins onto the orbital angular momentum direction -
        # only the aligned components enter the PN expressions below
        chiAL = np.dot(chiA, LHat)
        chiBL = np.dot(chiB, LHat)
        chiAB = np.dot(chiA, chiB)

        # Compute the reciprocal of the separation, used for PN expansions
        rInv = 1.0 / r

        # Mass related quantities
        mA = q / (1.0 + q)
        mB = 1.0 / (1.0 + q)
        eta = mA * mB
        deltaM = mA - mB

        # Linear and mass difference weighted spin projections
        SL = np.dot(LHat, mA**2.0 * chiA + mB**2.0 * chiB)
        SigmaL = np.dot(LHat, mB * chiB - mA * chiA)

        # Post-Newtonian terms
        # See equation 4.2 of http://arxiv.org/abs/1212.5520v1 for the PN expression for omega(r) for circular orbits.
        # Note that the 2.5PN term disagrees with equation 5.10, 5.11a, 5.11b of http://journals.aps.org/prd/pdf/10.1103/PhysRevD.74.104033
        # We use the version in equation 4.2 of the first reference since it is more recent.
        # We also include the spin-spin term (2PN order) from equation 4.5 of Equation 4.5 of http://arxiv.org/abs/gr-qc/9506022
        # Equation 228 of c also gives omega(r), without spin terms.
        A1 = (-3.0 + eta) * rInv
        A1p5 = (
            (
                -chiAL * (2.0 * mA**2.0 + 3.0 * eta)
                - chiBL * (2.0 * mB**2.0 + 3.0 * eta)
            )
            * rInv
            * np.sqrt(rInv)
        )
        A2 = (
            (
                6.0
                + 41 * eta / 4.0
                + eta**2
                - 1.5 * eta * chiAB
                + 4.5 * eta * chiAL * chiBL
            )
            * rInv
            * rInv
        )
        A2p5 = (
            (22.5 - 13.5 * eta) * SL + (13.5 - 6.5 * eta) * deltaM * SigmaL
        ) * (rInv**2.5)
        A3 = (
            (
                -10.0
                + (
                    -75707.0 / 840
                    + 41 * np.pi**2 / 64.0
                    + 22.0 * np.log(r / rPrime0)
                )
                * eta
                + 9.5 * eta**2
                + eta**3.0
            )
            * rInv
            * rInv
            * rInv
        )
        A3p5 = (
            (1.0 / 8.0)
            * (
                (-495.0 - 561.0 * eta - 51 * eta**2) * SL
                + (-297.0 - 341 * eta - 21 * eta**2) * deltaM * SigmaL
            )
            * (rInv**3.5)
        )
        omega = np.sqrt(rInv**3.0 * (1.0 + A1 + A1p5 + A2 + A2p5 + A3 + A3p5))

        # adot
        # adot0 = (dr/dt)/r, given in equation 4.12 of
        # http://arxiv.org/abs/gr-qc/9506022
        B1 = -(1.0 / 336.0) * (1751 + 588 * eta) * rInv
        B1p5 = (
            -(
                (7.0 / 12.0)
                * (
                    chiAL * (19.0 * mA**2.0 + 15.0 * eta)
                    + chiBL * (19.0 * mB**2.0 + 15.0 * eta)
                )
                - 4.0 * np.pi
            )
            * rInv
            * np.sqrt(rInv)
        )
        B2 = (
            (-5.0 / 48.0)
            * eta
            * (59.0 * chiAB - 173.0 * chiAL * chiBL)
            * rInv
            * rInv
        )

        dr_dt = (-64.0 / 5.0) * eta * rInv * rInv * rInv * (1 + B1 + B1p5 + B2)
        adot = dr_dt / r

        return omega, adot

    else:
        # VECTORIZED PATH - Process arrays for performance (all operations are already vectorized with NumPy!)
        r = np.asarray(r)
        q = np.asarray(q)
        chiA = np.asarray(chiA)  # Should be (N, 3)
        chiB = np.asarray(chiB)  # Should be (N, 3)

        N = len(r)
        assert chiA.shape == (
            N,
            3,
        ), f"chiA must have shape ({N}, 3), got {chiA.shape}"
        assert chiB.shape == (
            N,
            3,
        ), f"chiB must have shape ({N}, 3), got {chiB.shape}"
        assert len(q) == N, f"q must have length {N}, got {len(q)}"

        LHat = np.array([0.0, 0.0, 1.0])
        # Vectorized dot products using broadcasting
        chiAL = chiA @ LHat  # (N,) array
        chiBL = chiB @ LHat  # (N,) array
        chiAB = np.sum(chiA * chiB, axis=1)  # (N,) array
        rInv = 1.0 / r

        # Mass related quantities (all vectorized)
        mA = q / (1.0 + q)
        mB = 1.0 / (1.0 + q)
        eta = mA * mB
        deltaM = mA - mB

        # Linear and mass difference weighted spin projections
        SL = (mA**2)[:, np.newaxis] * chiA + (mB**2)[:, np.newaxis] * chiB
        SL = SL @ LHat  # (N,) array
        SigmaL = mB[:, np.newaxis] * chiB - mA[:, np.newaxis] * chiA
        SigmaL = SigmaL @ LHat  # (N,) array

        # Post-Newtonian terms (all vectorized)
        A1 = (-3.0 + eta) * rInv
        A1p5 = (
            (
                -chiAL * (2.0 * mA**2.0 + 3.0 * eta)
                - chiBL * (2.0 * mB**2.0 + 3.0 * eta)
            )
            * rInv
            * np.sqrt(rInv)
        )
        A2 = (
            (
                6.0
                + 41 * eta / 4.0
                + eta**2
                - 1.5 * eta * chiAB
                + 4.5 * eta * chiAL * chiBL
            )
            * rInv
            * rInv
        )
        A2p5 = (
            (22.5 - 13.5 * eta) * SL + (13.5 - 6.5 * eta) * deltaM * SigmaL
        ) * (rInv**2.5)
        A3 = (
            (
                -10.0
                + (
                    -75707.0 / 840
                    + 41 * np.pi**2 / 64.0
                    + 22.0 * np.log(r / rPrime0)
                )
                * eta
                + 9.5 * eta**2
                + eta**3.0
            )
            * rInv
            * rInv
            * rInv
        )
        A3p5 = (
            (1.0 / 8.0)
            * (
                (-495.0 - 561.0 * eta - 51 * eta**2) * SL
                + (-297.0 - 341 * eta - 21 * eta**2) * deltaM * SigmaL
            )
            * (rInv**3.5)
        )
        omega = np.sqrt(rInv**3.0 * (1.0 + A1 + A1p5 + A2 + A2p5 + A3 + A3p5))

        # adot (vectorized)
        B1 = -(1.0 / 336.0) * (1751 + 588 * eta) * rInv
        B1p5 = (
            -(
                (7.0 / 12.0)
                * (
                    chiAL * (19.0 * mA**2.0 + 15.0 * eta)
                    + chiBL * (19.0 * mB**2.0 + 15.0 * eta)
                )
                - 4.0 * np.pi
            )
            * rInv
            * np.sqrt(rInv)
        )
        B2 = (
            (-5.0 / 48.0)
            * eta
            * (59.0 * chiAB - 173.0 * chiAL * chiBL)
            * rInv
            * rInv
        )

        dr_dt = (-64.0 / 5.0) * eta * rInv * rInv * rInv * (1 + B1 + B1p5 + B2)
        adot = dr_dt / r

        return omega, adot


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


# Do both training and eigenvalue analysis in one step
# uses the functions defined above: train_gpr_model, normalize_data, and predict_with_gpr_model
def train_model_and_eigenvalue_analysis(
    df, input_col, output_col, output_col_initial=None, use_diff=False, gpr=None
):
    """
    Trains a GPR model, predicts, plots, and shows the kernel eigenvalue decay.

    Args:
        df: pd.DataFrame
        input_col: str
        output_col: str
        utput_col_initial: str or None (optional)
            if provided with use_diff=True, output is computed as output_col - output_col_initial
        use_diff: bool (optional)
        gpr: module (optional) - module containing GPR functions
    """

    # Prepare input data
    X = df[input_col].values

    # If use_diff is set, train on the correction (delta) instead of the raw output value -
    # this is useful when the GPR should learn how much to adjust an existing PN guess
    # rather than predicting the absolute value (which we end up not doing anyway)
    if use_diff and output_col_initial:
        Y = df[output_col].values - df[output_col_initial].values
        y_label = f"$\\Delta${output_col}"
    else:
        Y = df[output_col].values
        y_label = output_col

    # Train GPR
    model, likelihood = gpr.train_gpr_model(X, Y)

    # Normalize values - needed for eigenvalue analysis
    X_normalized, Y_normalized, scaler_X, scaler_Y = gpr.normalize_data(X, Y)
    train_X = torch.from_numpy(X_normalized).float()

    # Dense grid over the input range for a smooth predicted curve
    dense_X = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    dense_X_normalized = scaler_X.transform(dense_X)
    mean_pred, stddev_pred = gpr.predict_with_gpr_model(
        dense_X, model, likelihood
    )

    # Sort by X for plotting
    sorted_indices = np.argsort(dense_X.flatten())
    sorted_dense_X = dense_X.flatten()[sorted_indices]
    sorted_mean = mean_pred.flatten()[sorted_indices]
    sorted_stddev = stddev_pred.flatten()[sorted_indices]

    # Plot GPR fit with 2-sigma confidence band
    plt.figure(figsize=(8, 6))
    plt.plot(X, Y, "o", label="Original Test Data", color="orange")
    plt.plot(sorted_dense_X, sorted_mean, "b", label="GPR Prediction")
    plt.fill_between(
        sorted_dense_X,
        sorted_mean - 2 * sorted_stddev,
        sorted_mean + 2 * sorted_stddev,
        alpha=0.5,
        color="blue",
        label="Confidence Interval",
    )
    plt.xlabel(input_col, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f"GPR: {input_col} → {y_label}", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Kernel eigenvalue analysis
    # The eigenvalue decay of the kernel matrix K reveals the effective dimensionality
    # of the GP - fast decay means the model is dominated by a small number of modes
    # whereas slow decay means it needs more
    with torch.no_grad():
        K = model.covar_module(train_X).evaluate()
        # Add a small jitter to the diagonal to ensure numerical stability
        # before computing eigenvalues (avoids near-zero eigenvalues)
        K += 1e-6 * torch.eye(K.size(-1))
        eigenvalues = torch.sort(
            torch.linalg.eigvalsh(K), descending=True
        ).values

    # Plot the eigenvalues
    plt.figure(figsize=(8, 5))
    plt.semilogy(eigenvalues.cpu().numpy(), marker="o")
    plt.title("Eigenvalues of the GP Kernel Matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue (log scale)")
    plt.grid(True)
    plt.show()


# Leave one out predictions
# ie, leave out one data point and do GPR on the remaining points, then repeat for all
# points. This is used to test model performance
def loo_predictions(
    filtered_df, inputVar, outputVar, outputVar_initial=None, use_diff=False
):
    """
    Compute GPR Leave-One-Out predictions and uncertainties for any input variable.

    Args:
        filtered_df (pd.DataFrame): DataFrame with input data
        inputVar    (str): Name of the input variable
        outputVar   (str): Name of the output variable (optional)
            If provided with use_diff=True, gives output as a difference: outputVar - outputVar_initial
            use_diff (bool): Whether to compute output as a difference (optional)
    Returns:
        X: values of the input variable
        Y: true (possible delta) output values
        predictions_LOO: predicted output values (unnormalized, one per point)
        uncertainties_LOO: stddev of prediction (unnormalized, one per point)
    """

    # Prepare input and output arrays
    X = filtered_df[inputVar].values
    if use_diff and outputVar_initial is not None:
        Y = (
            filtered_df[outputVar].values
            - filtered_df[outputVar_initial].values
        )
    else:
        Y = filtered_df[outputVar].values

    predictions_LOO = np.zeros_like(X, dtype=float)
    uncertainties_LOO = np.zeros_like(X, dtype=float)

    for i in range(len(X)):
        # Boolean mask: True for all points except the i-th (held out) point
        train_indices = np.ones(len(X), dtype=bool)
        train_indices[i] = False

        X_train = X[train_indices]
        Y_train = Y[train_indices]
        X_test = X[i : i + 1]  # shape (1,) - slice preserves dimensions

        # Train model on the raw, unnormalized N-1 data
        model, likelihood = train_gpr_model(X_train, Y_train)
        model.eval()
        likelihood.eval()

        # Normalize X_test using the same normalization used in train_gpr_model
        X_test_norm = (X_test - model.input_mean) / model.input_std
        X_test_tensor = torch.from_numpy(X_test_norm).float()

        with torch.no_grad():
            observed_pred = likelihood(model(X_test_tensor))

        pred_mean = observed_pred.mean.numpy()
        pred_std = observed_pred.variance.sqrt().numpy()

        # Denormalize predictions using model output normalization
        pred_mean_un = pred_mean * model.output_std + model.output_mean
        pred_std_un = pred_std * model.output_std

        # .item() extracts the scalar from a length-1 array
        predictions_LOO[i] = pred_mean_un.item()
        uncertainties_LOO[i] = pred_std_un.item()

    return X, Y, predictions_LOO, uncertainties_LOO


# Import simulations of interest to apply GPR corrections to
def parse_test_runs(run_strings):
    """
    Parse lines like 'RunID=0000 ZwickyDays=10 q=8.0 chiA=... chiB=... D0=...
    Omega0=... adot0=...' into a DataFrame for GPR correction. Field order does
    not matter, relies on key lookup and if there are unknown fields, they get
    ignored.

    Args:
        run_strings (list of str): Takes in simulation info in string format.

    Returns:
        pd.DataFrame: DataFrame with parsed parameter columns.
    """
    test_runs = []

    for run in run_strings:
        # Split on whitespace to get key=value tokens, then build a dict -
        # field order does not matter and uknown keys are ignored
        parts = dict(token.split("=", 1) for token in run.split())
        RunID = parts["RunID"]  # identify simulation
        q = float(parts["q"])  # mass ratio
        chiA_x, chiA_y, chiA_z = map(
            float, parts["chiA"].split(",")
        )  # spin of object A
        chiB_x, chiB_y, chiB_z = map(
            float, parts["chiB"].split(",")
        )  # spin of object B
        D0 = float(parts["D0"])  # separation
        Omega0 = float(
            parts["Omega0"]
        )  # spec pn guess orbital frequency to be corrected
        adot0 = float(parts["adot0"])  # spec pn guess adot to be corrected

        # Prepare test dataframe and rename to match columns
        test_runs.append(
            {
                "name": f"test_{RunID}",
                "initial_separation": D0,
                "spec_pn_guess_omega": Omega0,
                "spec_pn_guess_adot": adot0,
                "initial_mass1": None,
                "initial_mass2": None,
                "mass_ratio": q,
                "S1x": chiA_x,
                "S1y": chiA_y,
                "S1z": chiA_z,
                "S2x": chiB_x,
                "S2y": chiB_y,
                "S2z": chiB_z,
                "eccentricity": None,
            }
        )
    return pd.DataFrame(test_runs)


# Apply GPR corrections using the previously trained model
def apply_gpr_corrections(
    df_test,
    model_omega,
    likelihood_omega,
    model_adot,
    likelihood_adot,
    input_columns=None,
):
    """
    Apply trained GPR delta corrections to the input DataFrame
    to produce corrected PN values.

    Args:
        df_test (pd.DataFrame):         test DataFrame with raw initial values
        model_omega:                    trained GPR model for omega
        likelihood_omega:               likelihood for omega model
        model_adot:                     trained GPR model for adot
        likelihood_adot:                likelihood for adot model.
        input_columns (list, optional): columns for the GPR input.
                                        Defaults to the standard 8 features.

    Returns:
        pd.DataFrame: DataFrame with added columns:
            delta_pred_omega
            delta_pred_adot
            gpr_corrected_omega
            gpr_corrected_adot
    """
    # Prepare test input X - same features as training
    if input_columns is None:
        # Standard 8-feature input: separation, mass ratio, and 3D spins for
        # both objects - must match the columns used during training
        input_columns = [
            "initial_separation",
            "mass_ratio",
            "S1x",
            "S1y",
            "S1z",
            "S2x",
            "S2y",
            "S2z",
        ]

    # Use a distinct name X_test so as to not overwrite the X used in the training
    X_test = df_test[input_columns].values

    # Predict the corrections (deltas) for omega and adot
    delta_omega_pred, _ = predict_with_gpr_model(
        X_test, model_omega, likelihood_omega
    )
    delta_adot_pred, _ = predict_with_gpr_model(
        X_test, model_adot, likelihood_adot
    )

    df_test["delta_pred_omega"] = delta_omega_pred
    df_test["delta_pred_adot"] = delta_adot_pred

    # Add corrections: PN initial guess + GPR-predicted correction
    df_test["gpr_corrected_omega"] = (
        df_test["spec_pn_guess_omega"] + delta_omega_pred
    )
    df_test["gpr_corrected_adot"] = (
        df_test["spec_pn_guess_adot"] + delta_adot_pred
    )

    return df_test


# Save the GPR corrected values to a txt file
def save_gpr_corrected(
    df,
    output_file,
    omega_col="gpr_corrected_omega",
    adot_col="gpr_corrected_adot",
    zwicky_days=10,
):
    """ "
    Write and save GPR corrected runs to a text file.
    Each line contains RunID, ZwickyDays, q, chiA, chiB, D0, Omega0, adot0.

    Args:
        df (pd.DataFrame): DataFrame containing corrected simsulations
        output_file (str): Output file path (e.g., "GPR_corrected_sims.txt")
        omega_col (str): Column name for the corrected omega
        adot_col (str): Column name for the corrected adot
        zwicky_days (int): ZwickyDays value to include (default: 10)

    Returns:
        None
    """
    lines = []
    for _, row in df.iterrows():
        # Strip the "test_" prefix added by parse_test_runs to recover the
        # original RunID (e.g. "test_0111" -> "0111")
        runid = row["name"].replace("test_", "")

        q = row["mass_ratio"]
        chiA = f"{row['S1x']},{row['S1y']},{row['S1z']}"
        chiB = f"{row['S2x']},{row['S2y']},{row['S2z']}"
        D0 = row["initial_separation"]
        Omega0 = row[omega_col]
        adot0 = row[adot_col]

        # High precision formatting: D0 to 10 demical places,
        # Omega0 and adot0 to 18 decimal places to match the precision
        # expeced by SpEC
        line = (
            f"RunID={runid} "
            f"ZwickyDays={zwicky_days} "
            f"q={q} "
            f"chiA={chiA} "
            f"chiB={chiB} "
            f"D0={D0:.10f} "
            f"Omega0={Omega0:.18f} "
            f"adot0={adot0:.18e}"
        )
        lines.append(line)

    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Exported {len(lines)} simulations to {output_file}")


# Leave one out cross validation to run after GPR
def loo_crossval(
    X: np.ndarray,
    Y: np.ndarray,
    train_gpr_function,
    predict_with_gpr_function,
    target_name="Target",
):
    """
    Perform Leave-One-Out Cross-Validation. Train N models (each omits one point),
    predict the held-out point, and collect predictions and uncertainties.

    Args:
        X (np.ndarray): Input features (N, D)
        Y (np.ndarray): Target variable (N, )
        train_gpr_funcion (callable): Function to train the GPR model,
            must return (model, likelihood).
        predict_gpr_function (callable): Function to predict using the GPR model.
        target_name (str): Label for plots and print output.

    Returns:
        predictions_loo: ndarray of shape (N, )
        uncertainties_loo: ndarray of shape (N, )
        rmse: float
        mae: float
        r_squared: float
    """
    N = len(Y)
    predictions_loo = np.zeros_like(Y)
    uncertainties_loo = np.zeros_like(Y)

    print(f"Processing {N} LOO iterations for {target_name}...")
    for i in range(N):
        # Progress update every 10 iterations so long runs are easily trackable
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N} complete")

        # Create train and test split
        # Boolean mask: all True except index i (held out point)
        train_mask = np.ones(N, dtype=bool)
        train_mask[i] = False

        X_train = X[train_mask]
        Y_train = Y[train_mask]
        X_test = X[
            i : i + 1
        ]  # slice preserves the 2D shape needed by the model

        # Train and predict
        model_loo, likelihood_loo = train_gpr_function(X_train, Y_train)
        pred_mean, pred_std = predict_with_gpr_function(
            X_test, model_loo, likelihood_loo
        )

        predictions_loo[i] = pred_mean[0]
        uncertainties_loo[i] = pred_std[0]

    Y_loo = Y  # Same as the original Y for the multi input case

    # Plot correlation
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_loo, predictions_loo, alpha=0.6, s=20)

    # Perfect correlation/prediction line (y = x)
    min_val = min(Y_loo.min(), predictions_loo.min())
    max_val = max(Y_loo.max(), predictions_loo.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2,
        label="Perfect Correlation",
    )

    # Labels and formatting
    plt.xlabel(f"True Δ{target_name}", fontsize=12)
    plt.ylabel(f"LOO Predicted Δ{target_name}", fontsize=12)
    plt.title(f"LOO: GPR Predictions vs True ({target_name})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Calculate and display R^2
    # Computed from the Pearson correlation coefficient - equivalent to the coefficient
    # of determination for a linear fit through the origin
    correlation = np.corrcoef(Y_loo, predictions_loo)[0, 1]
    r_squared_loo = correlation**2
    plt.text(
        0.95,
        0.95,
        f"R² = {r_squared_loo:.4f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        horizontalalignment="right",
    )

    plt.tight_layout()
    plt.show()

    # Print metrics with goal values
    rmse_loo = np.sqrt(np.mean((Y_loo - predictions_loo) ** 2))
    mae_loo = np.mean(np.abs(Y_loo - predictions_loo))
    y_range = Y_loo.max() - Y_loo.min()  # used to contextualize RMSE/MAE

    print(f"=== LEAVE-ONE-OUT CROSS-VALIDATION RESULTS ({target_name}) ===")
    print(
        f"RMSE: {rmse_loo:.6f} (goal: < 1 % of target range, lower is better)"
    )
    print(f"MAE: {mae_loo:.6f} (goal: < 1 % of target range, lower is better)")
    print(
        f"R²: {r_squared_loo:.4f} (goal: > 0.95 excellent, > 0.90 good, < 0.70"
        " poor)"
    )

    # Additional LOO specific info
    print(f"\n Dataset size: {len(Y_loo)} points")
    print(
        f"Each model is trained on {len(Y_loo)-1} points, and tested on 1 point"
    )
    print("This provides an unbiased generalization estimate.")

    return predictions_loo, uncertainties_loo, rmse_loo, mae_loo, r_squared_loo


# Function to compute and plot the residuals
def plot_loo_residuals(Y_loo, predictions_loo, target_name="Target", show=True):
    """
    Calculate LOO prediction residuals, plot a histogram, and print statistics.

    Args:
        Y_loon(np.ndarray): true target values from LOO cross validation
        predictions_loo (np.ndarray): predicted values from LOO cross validation
        target_name (str): Name of target variable

    Returns:
        residuals_loo (np.ndarray): residuals
    """

    # Compute residuals: LOO prediction error
    # Residual = true - predicted
    residuals_loo = Y_loo - predictions_loo

    # Make histogram
    # plt.figure(figsize=(8, 5))
    plt.hist(residuals_loo, bins=20, color="skyblue", edgecolor="k", alpha=0.8)
    # Vertical line at zero highlights systematic bias - ideally the histogram is
    # centered on this line
    plt.axvline(0, color="r", linestyle="--", label="Zero Error")

    plt.title(f"LOO Residuals Histogram for {target_name}")
    plt.xlabel(" Residuals", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.tight_layout()
    if show:
        plt.show()

    # Print statistics
    print(f"Residual statistics for {target_name}:")
    print(f"Mean residual:          {np.mean(residuals_loo):.4e}")
    print(f"Std of residuals:       {np.std (residuals_loo):.4e}")
    print(f"Max residual:           {np.max (residuals_loo):.4e}")
    print(f"Min residual:           {np.min (residuals_loo):.4e}")

    return residuals_loo


# Load, open, and read saved GPR from disk
def load_gpr_checkpoint(ckpt_path):
    """
    Loads a saved Gaussian Process Regression (GPR) model from disk.
    Restores model weights, likelihood, normalization parameters,
    and the raw training data (optional).

    Args:
        ckpt_path (str): Path to the checkpoint file.

    Returns:
        model (GPRegressionModel): Loaded GPR model.
        likelihood (GaussianLikelihood): Loaded likelihood.
        meta (dict): Metadata including input features.
    """
    # Load the checkpoint file
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Unpack metadata
    meta = ckpt["metadata"]
    features = meta["input_features"]
    D = len(features)  # number of input dimensions

    # Build dummy input and output tensors as placeholders to construct
    # the model object correctly. These get replaced with the trained values
    # saved in the checkpoint later in model.load_state_dict and likelihood.load_state_dict
    dummy_x = torch.zeros(1, D)  # ensures correct dimension input features
    dummy_y = torch.zeros(1)  # ensures scalar output

    # Initialize likelihood and model
    likelihood = (
        gpytorch.likelihoods.GaussianLikelihood()
    )  # represents assumed noise model of the data
    model = GPRegressionModel(dummy_x, dummy_y, likelihood) # constructs model object

    # Load trained parameters back into model and likelihood
    # and overwrite dummy inputs
    model.load_state_dict(ckpt["model_state_dict"])
    likelihood.load_state_dict(ckpt["likelihood_state_dict"])

    # Restore the normalization statistics used during training
    # so that the predictions are correctly scaled back to the original units
    norm = ckpt["normalization"]
    model.set_normalization(
        input_mean=np.array(norm["input_mean"]),  # mean of training features
        input_std=np.array(
            norm["input_std"]
        ),  # standard deviation of training features
        output_mean=norm["output_mean"],  # mean of training targets (deltas)
        output_std=norm[
            "output_std"
        ],  # standard deviation of training targets (deltas)
    )

    # Switch to evaluation mode before inference
    model.eval()
    likelihood.eval()

    return model, likelihood, meta
