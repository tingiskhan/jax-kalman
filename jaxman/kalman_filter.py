from typing import Tuple, Optional
import jax.numpy as jnp
from jax import lax
from numpyro.distributions import MultivariateNormal


class KalmanFilter:
    """
    A JAX-based Kalman Filter class with a similar API to pykalman.KalmanFilter.

    Features:
    - Uses jnp.linalg.solve instead of directly computing matrix inverses to improve numerical stability.
    - Applies the Joseph form for covariance updates for better numerical robustness.
    - Handles missing observations by replacing them with the predicted observation means.
    - Computes the log-likelihood of the observed data under the linear-Gaussian model using numpyro distributions.
    """

    def __init__(
            self,
            transition_matrices: jnp.ndarray,
            observation_matrices: jnp.ndarray,
            transition_covariance: jnp.ndarray,
            observation_covariance: jnp.ndarray,
            initial_state_mean: jnp.ndarray,
            initial_state_covariance: jnp.ndarray,
    ) -> None:
        """
        Initialize the Kalman filter parameters.

        Parameters
        ----------
        transition_matrices : jnp.ndarray of shape (n_dim_state, n_dim_state)
            State transition matrix (often denoted as A).
        observation_matrices : jnp.ndarray of shape (n_dim_obs, n_dim_state)
            Observation matrix (often denoted as C or H).
        transition_covariance : jnp.ndarray of shape (n_dim_state, n_dim_state)
            Covariance matrix of the process noise (often Q).
        observation_covariance : jnp.ndarray of shape (n_dim_obs, n_dim_obs)
            Covariance matrix of the observation noise (often R).
        initial_state_mean : jnp.ndarray of shape (n_dim_state,)
            Mean vector of the initial state distribution.
        initial_state_covariance : jnp.ndarray of shape (n_dim_state, n_dim_state)
            Covariance matrix of the initial state distribution.

        All parameters should be JAX arrays (or will be converted to JAX arrays),
        and they should be compatible with JIT compilation and vectorized operations.
        """
        self.transition_matrices = jnp.array(transition_matrices)
        self.observation_matrices = jnp.array(observation_matrices)
        self.transition_covariance = jnp.array(transition_covariance)
        self.observation_covariance = jnp.array(observation_covariance)
        self.initial_state_mean = jnp.array(initial_state_mean)
        self.initial_state_covariance = jnp.array(initial_state_covariance)

        # Determine dimensions
        self.n_dim_state: int = self.transition_matrices.shape[0]
        self.n_dim_obs: int = self.observation_matrices.shape[0]

    def _filter_step(
            self,
            carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            obs_t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        One step of the forward filtering using the Kalman filter equations.

        Instead of computing inverses directly, we use `jnp.linalg.solve` for numerical stability.
        The Joseph form is employed for covariance updates.
        Missing observations are replaced by predicted observation means.

        Parameters
        ----------
        carry : tuple
            A tuple (pred_mean, pred_cov, ll_cum) representing the predicted state mean,
            predicted state covariance, and cumulative log-likelihood from previous steps.
        obs_t : jnp.ndarray of shape (n_dim_obs,)
            The observation at the current time step. NaNs represent missing values.

        Returns
        -------
        new_carry : tuple
            (new_mean, new_cov, new_ll_cum) The updated filtered state mean, covariance,
            and cumulative log-likelihood after incorporating the current observation.
        outputs : tuple
            (new_mean, new_cov, log_likelihood_t) The filtered state mean and covariance
            for the current time step, and the incremental log-likelihood.
        """
        A = self.transition_matrices
        C = self.observation_matrices
        Q = self.transition_covariance
        R = self.observation_covariance

        pred_mean, pred_cov, ll_cum = carry

        # Prediction step
        pred_mean = A @ pred_mean
        pred_cov = A @ pred_cov @ A.T + Q

        # Identify missing observations and replace them with predicted means
        mask = ~jnp.isnan(obs_t)
        yhat = C @ pred_mean
        obs_t_filled = jnp.where(mask, obs_t, yhat)

        # Compute innovation and S
        innovation = obs_t_filled - yhat
        S = C @ pred_cov @ C.T + R

        # Compute log-likelihood increment
        log_likelihood_t = MultivariateNormal(loc=yhat, covariance_matrix=S).log_prob(obs_t_filled)

        # Compute Kalman gain using solve: S K^T = (C P)^T => K = (S \ (C P))^T
        K = jnp.linalg.solve(S, (C @ pred_cov).T).T

        # Joseph form for covariance
        I = jnp.eye(self.n_dim_state)
        new_mean = pred_mean + K @ innovation
        new_cov = (I - K @ C) @ pred_cov @ (I - K @ C).T + K @ R @ K.T

        new_ll_cum = ll_cum + log_likelihood_t

        return (new_mean, new_cov, new_ll_cum), (new_mean, new_cov, log_likelihood_t)

    def filter(
            self,
            observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Run the Kalman filter forward pass for a sequence of observations.

        Parameters
        ----------
        observations : jnp.ndarray of shape (n_timesteps, n_dim_obs)
            The observations over time, possibly containing NaN for missing values.

        Returns
        -------
        filtered_state_means : jnp.ndarray of shape (n_timesteps, n_dim_state)
            The filtered state means at each time step.
        filtered_state_covariances : jnp.ndarray of shape (n_timesteps, n_dim_state, n_dim_state)
            The filtered state covariances at each time step.
        log_likelihood : float
            The total log-likelihood of the observed data under the model.
        """
        observations = jnp.asarray(observations)
        init_carry = (self.initial_state_mean, self.initial_state_covariance, 0.0)

        (final_mean, final_cov, final_ll), (filtered_means, filtered_covs, _) = lax.scan(
            self._filter_step, init_carry, observations
        )

        return filtered_means, filtered_covs, final_ll

    def _smooth_step(
            self,
            carry: Tuple[jnp.ndarray, jnp.ndarray],
            args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One step of the backward smoothing (RTS smoother).

        Parameters
        ----------
        carry : tuple (next_smoothed_mean, next_smoothed_cov)
            The smoothed mean and covariance from the next time step.
        args : tuple (filtered_mean_t, filtered_cov_t, predicted_mean_next, predicted_cov_next)
            Quantities from the forward pass and predictions for the next time step.

        Returns
        -------
        (smoothed_mean_t, smoothed_cov_t) for this time step, repeated as output.
        """
        A = self.transition_matrices
        next_smoothed_mean, next_smoothed_cov = carry
        filtered_mean_t, filtered_cov_t, predicted_mean_next, predicted_cov_next = args

        # Compute J using solve
        # J = filtered_cov_t @ A.T @ inv(predicted_cov_next)
        # Solve predicted_cov_next * X^T = (A @ filtered_cov_t.T)
        J = jnp.linalg.solve(predicted_cov_next, (A @ filtered_cov_t.T).T).T

        smoothed_mean_t = filtered_mean_t + J @ (next_smoothed_mean - predicted_mean_next)
        smoothed_cov_t = filtered_cov_t + J @ (next_smoothed_cov - predicted_cov_next) @ J.T

        return (smoothed_mean_t, smoothed_cov_t), (smoothed_mean_t, smoothed_cov_t)

    def smooth(
            self,
            observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Apply RTS smoothing to the filtered estimates.

        Smoothing does not alter the likelihood of the data; it simply refines the state
        estimates using future observations.

        Parameters
        ----------
        observations : jnp.ndarray of shape (n_timesteps, n_dim_obs)

        Returns
        -------
        smoothed_state_means : jnp.ndarray of shape (n_timesteps, n_dim_state)
            The smoothed state means at each time step.
        smoothed_state_covariances : jnp.ndarray of shape (n_timesteps, n_dim_state, n_dim_state)
            The smoothed state covariances at each time step.
        log_likelihood : float
            The log-likelihood of the observed data, same as obtained from filtering.
        """
        observations = jnp.asarray(observations)
        filtered_means, filtered_covs, ll = self.filter(observations)

        A = self.transition_matrices
        Q = self.transition_covariance

        # Compute predicted means and covariances
        def pred_body(
                carry: Tuple[jnp.ndarray, jnp.ndarray],
                x: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
            pm, pc = carry
            fm_t, fc_t = x
            pm_next = A @ pm
            pc_next = A @ pc @ A.T + Q
            return (fm_t, fc_t), (pm_next, pc_next)

        init_carry = (self.initial_state_mean, self.initial_state_covariance)
        (_, _), (predicted_means, predicted_covs) = lax.scan(pred_body, init_carry,
                                                             (filtered_means[:-1], filtered_covs[:-1]))

        # Insert the first prediction at time 0
        first_pm = A @ self.initial_state_mean
        first_pc = A @ self.initial_state_covariance @ A.T + Q
        predicted_means = jnp.concatenate([jnp.reshape(first_pm, (1, -1)), predicted_means], axis=0)
        predicted_covs = jnp.concatenate(
            [jnp.reshape(first_pc, (1, self.n_dim_state, self.n_dim_state)), predicted_covs], axis=0)

        # Backward smoothing
        final_carry = (filtered_means[-1], filtered_covs[-1])
        args = (filtered_means[:-1], filtered_covs[:-1], predicted_means[1:], predicted_covs[1:])
        rev_args = tuple(a[::-1] for a in args)
        (sm_last_mean, sm_last_cov), (rev_sm_means, rev_sm_covs) = lax.scan(
            self._smooth_step, final_carry, list(zip(*rev_args))
        )

        smoothed_means = jnp.concatenate([rev_sm_means[::-1], jnp.reshape(sm_last_mean, (1, -1))], axis=0)
        smoothed_covariances = jnp.concatenate(
            [rev_sm_covs[::-1], jnp.reshape(sm_last_cov, (1, self.n_dim_state, self.n_dim_state))], axis=0)

        return smoothed_means, smoothed_covariances, ll

    def filter_update(
            self,
            filtered_state_mean: jnp.ndarray,
            filtered_state_covariance: jnp.ndarray,
            observation: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Update the Kalman filter with a single new observation step.

        Parameters
        ----------
        filtered_state_mean : jnp.ndarray of shape (n_dim_state,)
            The previous filtered state mean.
        filtered_state_covariance : jnp.ndarray of shape (n_dim_state, n_dim_state)
            The previous filtered state covariance.
        observation : jnp.ndarray of shape (n_dim_obs,), optional
            The new observation. If None or NaN values are present, missing entries are replaced by
            the predicted observation means.

        Returns
        -------
        new_filtered_state_mean : jnp.ndarray of shape (n_dim_state,)
            The updated filtered state mean after incorporating the new observation.
        new_filtered_state_covariance : jnp.ndarray of shape (n_dim_state, n_dim_state)
            The updated filtered state covariance.
        log_likelihood_t : float
            The log-likelihood contribution of this single observation step.
        """
        if observation is None:
            observation = jnp.full((self.n_dim_obs,), jnp.nan)

        A = self.transition_matrices
        C = self.observation_matrices
        Q = self.transition_covariance
        R = self.observation_covariance

        # Prediction
        pred_mean = A @ filtered_state_mean
        pred_cov = A @ filtered_state_covariance @ A.T + Q

        # Identify missing and fill
        mask = ~jnp.isnan(observation)
        yhat = C @ pred_mean
        obs_t_filled = jnp.where(mask, observation, yhat)

        innovation = obs_t_filled - yhat
        S = C @ pred_cov @ C.T + R

        # Log-likelihood
        log_likelihood_t = MultivariateNormal(loc=yhat, covariance_matrix=S).log_prob(obs_t_filled)

        # Kalman gain via solve
        K = jnp.linalg.solve(S, (C @ pred_cov).T).T

        # Joseph form
        I = jnp.eye(self.n_dim_state)
        new_mean = pred_mean + K @ innovation
        new_cov = (I - K @ C) @ pred_cov @ (I - K @ C).T + K @ R @ K.T

        return new_mean, new_cov, float(log_likelihood_t)
