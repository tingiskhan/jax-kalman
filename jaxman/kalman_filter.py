from typing import Optional, Tuple

import jax.numpy as jnp
from jax import lax
from numpyro.distributions import MultivariateNormal
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class KalmanFilter:
    """
    A JAX-based Kalman Filter class with a similar API to pykalman.KalmanFilter.

    Features:
    - Uses jnp.linalg.solve instead of directly computing matrix inverses to improve numerical stability.
    - Applies the Joseph form for covariance updates for better numerical robustness.
    - Handles missing observations by replacing them with the predicted observation means.
    - Computes the log-likelihood of the observed data under the linear-Gaussian model using numpyro distributions.
    - Supports optional transition and observation offsets.
    """

    def __init__(
        self,
        transition_matrices: jnp.ndarray,
        observation_matrices: jnp.ndarray,
        transition_covariance: jnp.ndarray,
        observation_covariance: jnp.ndarray,
        initial_state_mean: jnp.ndarray,
        initial_state_covariance: jnp.ndarray,
        transition_offset: Optional[jnp.ndarray] = None,
        observation_offset: Optional[jnp.ndarray] = None,
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
        transition_offset : jnp.ndarray of shape (n_dim_state,), optional
            A constant offset added to the state after applying the transition matrix.
        observation_offset : jnp.ndarray of shape (n_dim_obs,), optional
            A constant offset added to the predicted observations after applying the observation matrix.
        """
        self.transition_matrices = jnp.array(transition_matrices)
        self.observation_matrices = jnp.array(observation_matrices)
        self.transition_covariance = jnp.array(transition_covariance)
        self.observation_covariance = jnp.array(observation_covariance)
        self.initial_state_mean = jnp.array(initial_state_mean)
        self.initial_state_covariance = jnp.array(initial_state_covariance)

        self.n_dim_state: int = self.transition_matrices.shape[0]
        self.n_dim_obs: int = self.observation_matrices.shape[0]

        if transition_offset is None:
            transition_offset = jnp.zeros(self.n_dim_state)

        if observation_offset is None:
            observation_offset = jnp.zeros(self.n_dim_obs)

        self.transition_offset = transition_offset
        self.observation_offset = observation_offset

    def tree_flatten(self):
        children = (
            self.transition_matrices,
            self.observation_matrices,
            self.transition_covariance,
            self.observation_offset,
            self.initial_state_mean,
            self.initial_state_covariance,
            self.transition_offset,
            self.observation_offset,
        )

        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def _filter_step(
        self, carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], obs_t: jnp.ndarray
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """
        One step of the forward filtering using the Kalman filter equations.

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
        a = self.transition_matrices
        q = self.transition_covariance

        c = self.observation_matrices
        r = self.observation_covariance

        prev_mean, prev_cov, ll_cum = carry

        # Prediction step
        pred_mean = self.transition_offset + a @ prev_mean
        pred_cov = a @ prev_cov @ a.T + q

        # Identify missing observations and replace them with predicted means
        mask = jnp.isfinite(obs_t)

        yhat = self.observation_offset + c @ pred_mean
        obs_t_filled = jnp.where(mask, obs_t, yhat)

        # Compute innovation and S
        innovation = obs_t_filled - yhat
        s = c @ pred_cov @ c.T + r

        # Compute log-likelihood increment
        log_likelihood_t = MultivariateNormal(loc=yhat, covariance_matrix=s).log_prob(obs_t_filled)

        # Compute Kalman gain using solve: S K^T = (C P)^T => K = (S \ (C P))^T
        k = jnp.linalg.solve(s, (c @ pred_cov).T).T

        # Joseph form for covariance
        eye = jnp.eye(self.n_dim_state)
        new_mean = pred_mean + k @ innovation
        new_cov = (eye - k @ c) @ pred_cov @ (eye - k @ c).T + k @ r @ k.T

        new_ll_cum = ll_cum + log_likelihood_t

        return (new_mean, new_cov, new_ll_cum), (new_mean, new_cov, log_likelihood_t)

    def filter(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
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
        self, carry: Tuple[jnp.ndarray, jnp.ndarray], args: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        One step of the backward smoothing (RTS smoother).

        Parameters
        ----------
        carry : tuple (next_smoothed_mean, next_smoothed_cov)
            The smoothed mean and covariance from the next time step.
        args : tuple (filtered_mean_t, filtered_cov_t, predicted_mean_next, predicted_cov_next)
            Quantities from the forward pass and predictions for the next time step.
        """
        a = self.transition_matrices
        next_smoothed_mean, next_smoothed_cov = carry
        filtered_mean_t, filtered_cov_t, predicted_mean_next, predicted_cov_next = args

        # Compute J using solve
        j = jnp.linalg.solve(predicted_cov_next, (a @ filtered_cov_t.T).T).T

        smoothed_mean_t = filtered_mean_t + j @ (next_smoothed_mean - predicted_mean_next)
        smoothed_cov_t = filtered_cov_t + j @ (next_smoothed_cov - predicted_cov_next) @ j.T

        return (smoothed_mean_t, smoothed_cov_t), (smoothed_mean_t, smoothed_cov_t)

    def smooth(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Apply RTS smoothing to the filtered estimates.

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

        a = self.transition_matrices
        q = self.transition_covariance

        # Compute predicted means and covariances
        def pred_body(
            carry: Tuple[jnp.ndarray, jnp.ndarray], x: Tuple[jnp.ndarray, jnp.ndarray]
        ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
            pm, pc = carry
            fm_t, fc_t = x
            pm_next = a @ pm
            if self.transition_offset is not None:
                pm_next = pm_next + self.transition_offset
            pc_next = a @ pc @ a.T + q
            return (fm_t, fc_t), (pm_next, pc_next)

        init_carry = (self.initial_state_mean, self.initial_state_covariance)
        (_, _), (predicted_means, predicted_covs) = lax.scan(
            pred_body, init_carry, (filtered_means[:-1], filtered_covs[:-1])
        )

        # Insert the first prediction at time 0
        first_pm = a @ self.initial_state_mean
        if self.transition_offset is not None:
            first_pm = first_pm + self.transition_offset
        first_pc = a @ self.initial_state_covariance @ a.T + q
        predicted_means = jnp.concatenate([jnp.reshape(first_pm, (1, -1)), predicted_means], axis=0)
        predicted_covs = jnp.concatenate(
            [jnp.reshape(first_pc, (1, self.n_dim_state, self.n_dim_state)), predicted_covs], axis=0
        )

        # Backward smoothing
        final_carry = (filtered_means[-1], filtered_covs[-1])
        args = (filtered_means[:-1], filtered_covs[:-1], predicted_means[1:], predicted_covs[1:])
        rev_args = tuple(a[::-1] for a in args)
        (sm_last_mean, sm_last_cov), (rev_sm_means, rev_sm_covs) = lax.scan(self._smooth_step, final_carry, rev_args)

        smoothed_means = jnp.concatenate([rev_sm_means[::-1], jnp.reshape(sm_last_mean, (1, -1))], axis=0)
        smoothed_covariances = jnp.concatenate(
            [rev_sm_covs[::-1], jnp.reshape(sm_last_cov, (1, self.n_dim_state, self.n_dim_state))], axis=0
        )

        return smoothed_means, smoothed_covariances, ll

    def filter_update(
        self,
        filtered_state_mean: jnp.ndarray,
        filtered_state_covariance: jnp.ndarray,
        observation: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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

        carry = (filtered_state_mean, filtered_state_covariance, 0.0)
        state, _ = self._filter_step(carry, observation)

        return state


__all__ = [
    "KalmanFilter",
]
