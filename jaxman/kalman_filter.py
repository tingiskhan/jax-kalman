from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax

from .results import FilterResult, SmoothingResult


def _inflate_missing(
    non_valid_mask: jnp.ndarray, obs: jnp.ndarray, H: jnp.ndarray, R: jnp.ndarray, missing_value: float = 1e12
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Masks missing dimensions by zeroing corresponding rows of H and inflating the diagonal of R.

    Args:
        non_valid_mask: Boolean mask of shape (obs_dim,) indicating missing dimensions.
        obs: Observation vector of shape (obs_dim,). Missing entries are NaN.
        H: Observation matrix of shape (obs_dim, state_dim).
        R: Observation covariance matrix of shape (obs_dim, obs_dim).
        missing_value: Large scalar to add on the diagonal for missing dimensions.

    Returns:
        A tuple of:
          - obs_masked: Same shape as obs, with missing entries replaced by 0.0.
          - H_masked: Same shape as H, rows zeroed out for missing dimensions.
          - R_masked: Same shape as R, diagonal entries inflated for missing dimensions.
    """

    valid_mask = ~non_valid_mask
    valid_mask_f = valid_mask.astype(obs.dtype)

    H_masked = H * valid_mask_f[:, None]
    diag_inflation = (1.0 - valid_mask_f) * missing_value
    R_masked = R + jnp.diag(diag_inflation)
    obs_masked = jnp.where(valid_mask, obs, 0.0)

    return obs_masked, H_masked, R_masked


class KalmanFilter:
    """
    A JAX-based Kalman Filter supporting partial missing data, offsets, optional noise transform,
    integrated log-likelihood, and RTS smoothing. Uses pseudo-inverse to handle degenerate covariances.
    """

    def __init__(
        self,
        initial_mean: jnp.ndarray,
        initial_cov: jnp.ndarray,
        transition_matrix: Any,
        transition_cov: Any,
        observation_matrix: Any,
        observation_cov: Any,
        transition_offset: Optional[Any] = None,
        observation_offset: Optional[Any] = None,
        noise_transform: Optional[Any] = None,
    ):
        """
        Initializes a KalmanFilter.

        Args:
            initial_mean: Mean of the initial state, shape (state_dim,).
            initial_cov: Covariance of the initial state, shape (state_dim, state_dim).
            transition_matrix: F_t, shape (state_dim, state_dim) or callable returning it.
            transition_cov: Q_t, shape (noise_dim, noise_dim) or callable returning it.
            observation_matrix: H_t, shape (obs_dim, state_dim) or callable returning it.
            observation_cov: R_t, shape (obs_dim, obs_dim) or callable returning it.
            transition_offset: b_t, shape (state_dim,) or callable returning it. Default is None.
            observation_offset: d_t, shape (obs_dim,) or callable returning it. Default is None.
            noise_transform: G_t, shape (state_dim, noise_dim) or callable returning it.
                If None, an identity matrix of size (state_dim, state_dim) is used.
        """

        self.initial_mean = initial_mean
        self.initial_cov = initial_cov

        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov
        self.observation_matrix = observation_matrix
        self.observation_cov = observation_cov

        self.transition_offset = transition_offset
        self.observation_offset = observation_offset

        if noise_transform is None:
            state_dim = initial_mean.shape[0]
            noise_transform = jnp.eye(state_dim)

        self.noise_transform = noise_transform

    def _get_transition_matrix(self, t: int, x_prev: jnp.ndarray) -> jnp.ndarray:
        if callable(self.transition_matrix):
            return self.transition_matrix(t, x_prev)

        return self.transition_matrix

    def _get_transition_cov(self, t: int) -> jnp.ndarray:
        if callable(self.transition_cov):
            return self.transition_cov(t)

        return self.transition_cov

    def _get_observation_matrix(self, t: int) -> jnp.ndarray:
        if callable(self.observation_matrix):
            return self.observation_matrix(t)

        return self.observation_matrix

    def _get_observation_cov(self, t: int) -> jnp.ndarray:
        if callable(self.observation_cov):
            return self.observation_cov(t)

        return self.observation_cov

    def _get_transition_offset(self, t: int) -> jnp.ndarray:
        if self.transition_offset is None:
            return 0.0

        if callable(self.transition_offset):
            return self.transition_offset(t)

        return self.transition_offset

    def _get_observation_offset(self, t: int) -> jnp.ndarray:
        if self.observation_offset is None:
            return 0.0

        if callable(self.observation_offset):
            return self.observation_offset(t)

        return self.observation_offset

    def _get_noise_transform(self, t: int) -> jnp.ndarray:
        if callable(self.noise_transform):
            return self.noise_transform(t)

        return self.noise_transform

    def _predict(self, mean: jnp.ndarray, cov: jnp.ndarray, t: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        F_t = self._get_transition_matrix(t, mean)
        Q_t = self._get_transition_cov(t)
        b_t = self._get_transition_offset(t)
        G_t = self._get_noise_transform(t)

        mean_pred = F_t @ mean + b_t
        cov_pred = F_t @ cov @ F_t.T + G_t @ Q_t @ G_t.T

        return mean_pred, cov_pred

    def _update(
        self,
        mean_pred: jnp.ndarray,
        cov_pred: jnp.ndarray,
        obs_masked: jnp.ndarray,
        H_masked: jnp.ndarray,
        R_masked: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        S = H_masked @ cov_pred @ H_masked.T + R_masked
        S_pinv = jnp.linalg.pinv(S)

        K = cov_pred @ H_masked.T @ S_pinv
        residual = obs_masked - (H_masked @ mean_pred)

        mean_up = mean_pred + K @ residual
        cov_up = cov_pred - K @ H_masked @ cov_pred

        return mean_up, cov_up

    def _forward_pass(
        self, observations: jnp.ndarray, missing_value: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def scan_fn(carry, obs_t):
            t, mean_tm1, cov_tm1, ll_tm1 = carry

            mean_t, cov_t = self._predict(mean_tm1, cov_tm1, t)

            H_t = self._get_observation_matrix(t)
            R_t = self._get_observation_cov(t)
            d_t = self._get_observation_offset(t)

            obs_mask = jnp.isnan(obs_t)
            obs_masked, H_masked, R_masked = _inflate_missing(obs_mask, obs_t, H_t, R_t, missing_value)

            pred_mean_masked = H_masked @ mean_t + jnp.where(obs_mask, 0.0, d_t)
            pred_cov_masked = H_masked @ cov_t @ H_masked.T + R_masked

            dist_y = dist.MultivariateNormal(loc=pred_mean_masked, covariance_matrix=pred_cov_masked)
            step_log_prob = dist_y.log_prob(obs_masked)
            ll_t = ll_tm1 + step_log_prob

            corrected_mean_t, corrected_cov_t = self._update(mean_t, cov_t, obs_masked, H_masked, R_masked)

            return (t + 1, corrected_mean_t, corrected_cov_t, ll_t), (mean_t, cov_t, corrected_mean_t, corrected_cov_t)

        init_carry = (1, self.initial_mean, self.initial_cov, 0.0)
        final_carry, outputs = lax.scan(scan_fn, init_carry, observations)

        _, _, _, total_ll = final_carry

        predicted_means = outputs[0]
        predicted_covs = outputs[1]
        filtered_means = outputs[2]
        filtered_covs = outputs[3]

        return predicted_means, predicted_covs, filtered_means, filtered_covs, total_ll

    def filter(self, observations: jnp.ndarray, missing_value: float = 1e12) -> FilterResult:
        """
        Runs forward filtering over a sequence of observations, returning a named tuple
        FilterResult(means, covariances, log_likelihood).
        """

        (_, __, filtered_means, filtered_covs, total_ll) = self._forward_pass(observations, missing_value)

        return FilterResult(filtered_means, filtered_covs, total_ll)

    # TODO: fix
    def smooth(
        self, observations: jnp.ndarray, missing_value: float = 1e12
    ) -> SmoothingResult:
        """
        Runs forward filtering + RTS backward pass for smoothing, returning SmoothResult(means, covariances, log_likelihood).

        The backward pass uses a reverse-time lax.scan to fill in smoothed results for each time step.
        """
        predicted_means, predicted_covs, filter_means, filter_covs, ll = self._forward_pass(observations, missing_value)

        num_timesteps = observations.shape[0]

        def rts_step(carry, aux_t):
            """
            carry: (mean_next, cov_next, smeans, scovs)
            t_rev: the reversed time index, e.g. from T-2 down to 0
            """
            mean_next, cov_next = carry
            t, mean_f, cov_f, mean_p, cov_p = aux_t

            F_t = self._get_transition_matrix(t + 1, mean_f)
            cov_p_inv = jnp.linalg.pinv(cov_p)

            A_t = cov_f @ F_t.T @ cov_p_inv
            curr_mean_smooth = mean_f + A_t @ (mean_next - mean_p)
            curr_cov_smooth = cov_f + A_t @ (cov_next - cov_p) @ A_t.T

            return (curr_mean_smooth, curr_cov_smooth), (curr_mean_smooth, curr_cov_smooth)

        carry_init = (filter_means[-1], filter_covs[-1])

        time_inds = jnp.arange(num_timesteps)[:-1]
        xs = (
            time_inds,
            filter_means[:-1],
            filter_covs[:-1],
            predicted_means[1:],
            predicted_covs[1:],
        )

        _, (smoothed_means, smoothed_covariances) = lax.scan(rts_step, carry_init, xs, reverse=True)

        smoothed_means = jnp.concatenate([smoothed_means, jnp.expand_dims(filter_means[-1], axis=0)], axis=0)
        smoothed_covariances = jnp.concatenate([smoothed_covariances, jnp.expand_dims(filter_covs[-1], axis=0)], axis=0)

        return SmoothingResult(smoothed_means, smoothed_covariances, ll)

    def sample(self, rng_key: jax.random.PRNGKey, num_timesteps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Samples from the state-space model for num_timesteps.
        """

        def sample_step(carry, _):
            t, x_prev, rng_prev = carry

            F_t = self._get_transition_matrix(t, x_prev)
            Q_t = self._get_transition_cov(t)
            b_t = self._get_transition_offset(t)
            G_t = self._get_noise_transform(t)
            H_t = self._get_observation_matrix(t)
            R_t = self._get_observation_cov(t)
            d_t = self._get_observation_offset(t)

            rng_proc, rng_obs = jax.random.split(rng_prev)
            noise_dim = Q_t.shape[0]
            w_t = dist.MultivariateNormal(loc=jnp.zeros(noise_dim), covariance_matrix=Q_t).sample(rng_proc)

            x_t = F_t @ x_prev + b_t + G_t @ w_t
            obs_dim = R_t.shape[0]
            v_t = dist.MultivariateNormal(loc=jnp.zeros(obs_dim), covariance_matrix=R_t).sample(rng_obs)

            y_t = H_t @ x_t + d_t + v_t

            return (t + 1, x_t, rng_proc), (x_t, y_t)

        def init_state(key):
            return dist.MultivariateNormal(loc=self.initial_mean, covariance_matrix=self.initial_cov).sample(key)

        x0 = init_state(rng_key)
        init_carry = (0, x0, rng_key)

        _, (xs, ys) = lax.scan(sample_step, init_carry, None, length=num_timesteps)

        return xs, ys
