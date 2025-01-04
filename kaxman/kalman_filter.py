from typing import Tuple

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.tree_util import register_pytree_node_class
from jaxtyping import Array, Float

from .results import FilterResult, SmoothingResult


def _inflate_missing(non_valid_mask: jnp.ndarray, r: jnp.ndarray, inflation: float = 1e12) -> jnp.ndarray:
    """
    Masks missing dimensions by inflating the diagonal of R.

    Args:
        non_valid_mask: Boolean mask of shape (obs_dim,) indicating missing dimensions.
        r: Observation covariance matrix of shape (obs_dim, obs_dim).
        inflation: Large scalar to add on the diagonal for missing dimensions.

    Returns:
        A tuple of:
          - r_masked: Same shape as R, diagonal entries inflated for missing dimensions.
    """

    diag_inflation = jnp.where(non_valid_mask, inflation, 0.0)
    r_masked = r + jnp.diag(diag_inflation)

    return r_masked


@register_pytree_node_class
class KalmanFilter:
    """
    A JAX-based Kalman Filter supporting partial missing data, offsets, optional noise transform,
    integrated log-likelihood, and RTS smoothing. Uses pseudo-inverse to handle degenerate covariances.

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
        variance_inflation: Inflation factor for missing dimensions in the observation covariance.
    """

    def __init__(
        self,
        initial_mean: Float[Array, "state_dim"],  # noqa: 821
        initial_cov: Float[Array, "state_dim state_dim"],  # noqa: F722
        transition_matrix: Float[Array, "state_dim state_dim"],  # noqa: F722
        transition_cov: Float[Array, "state_dim state_dim"],  # noqa: F722
        observation_matrix: Float[Array, "obs_dim state_dim"],  # noqa: F722
        observation_cov: Float[Array, "obs_dim obs_dim"],  # noqa: F722
        transition_offset: Float[Array, "state_dim"] = None,  # noqa: F821
        observation_offset: Float[Array, "obs_dim"] = None,  # noqa: F821
        noise_transform: Float[Array, "state_dim noise_dim"] = None,  # noqa: F722
        variance_inflation: float = 1e8,
    ):
        self.initial_mean = initial_mean
        self.initial_cov = initial_cov

        self.transition_matrix = transition_matrix
        self.transition_cov = transition_cov
        self.observation_matrix = observation_matrix
        self.observation_cov = observation_cov

        self.transition_offset = transition_offset if transition_offset is not None else jnp.zeros(1)
        self.observation_offset = observation_offset if observation_offset is not None else jnp.zeros(1)

        if noise_transform is None:
            state_dim = initial_mean.shape[0]
            noise_transform = jnp.eye(state_dim)

        self.noise_transform = noise_transform
        self.variance_inflation = variance_inflation

    def tree_flatten(self):
        """
        Splits the object into (children, aux_data).
        children must be a tuple/list of JAX array leaves.
        aux_data holds everything else needed to reconstruct the class.
        """
        children = (
            self.initial_mean,
            self.initial_cov,
            self.transition_matrix,
            self.transition_cov,
            self.observation_matrix,
            self.observation_cov,
            self.transition_offset,
            self.observation_offset,
            self.noise_transform,
        )

        aux_data = dict(variance_inflation=self.variance_inflation)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Reconstructs the class instance from aux_data (static) + children (JAX arrays).
        The order of children must match tree_flatten above.
        """
        (
            initial_mean,
            initial_cov,
            transition_matrix,
            transition_cov,
            observation_matrix,
            observation_cov,
            transition_offset,
            observation_offset,
            noise_transform,
        ) = children

        return cls(
            initial_mean=initial_mean,
            initial_cov=initial_cov,
            transition_matrix=transition_matrix,
            transition_cov=transition_cov,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
            transition_offset=transition_offset,
            observation_offset=observation_offset,
            noise_transform=noise_transform,
            **aux_data,
        )

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
        if callable(self.transition_offset):
            return self.transition_offset(t)

        return self.transition_offset

    def _get_observation_offset(self, t: int) -> jnp.ndarray:
        if callable(self.observation_offset):
            return self.observation_offset(t)

        return self.observation_offset

    def _predict(self, mean: jnp.ndarray, cov: jnp.ndarray, t: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        f_t = self._get_transition_matrix(t, mean)
        q_t = self._get_transition_cov(t)
        b_t = self._get_transition_offset(t)
        g_t = self.noise_transform

        mean_pred = f_t @ mean + b_t
        cov_pred = f_t @ cov @ f_t.T + g_t @ q_t @ g_t.T

        return mean_pred, cov_pred

    def _update(
        self,
        mean_pred: jnp.ndarray,
        cov_pred: jnp.ndarray,
        obs_t: jnp.ndarray,
        h: jnp.ndarray,
        s_inverse: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        gain = cov_pred @ h.T @ s_inverse
        residual = obs_t - h @ mean_pred

        corrected_mean = mean_pred + gain @ residual
        corrected_cov = cov_pred - gain @ h @ cov_pred

        return corrected_mean, corrected_cov

    def _forward_pass(
        self, observations: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        def scan_fn(carry, obs_t):
            t, mean_tm1, cov_tm1, ll_tm1 = carry

            x_pred_mean, x_pred_cov = self._predict(mean_tm1, cov_tm1, t)

            h_t = self._get_observation_matrix(t)
            r_t = self._get_observation_cov(t)
            d_t = self._get_observation_offset(t)

            nan_mask = jnp.isnan(obs_t)

            # TODO: need to verify this...
            r_t = _inflate_missing(nan_mask, r_t, inflation=self.variance_inflation)

            y_pred_mean = h_t @ x_pred_mean + d_t
            y_pred_cov = h_t @ x_pred_cov @ h_t.T + r_t

            obs_masked = jnp.where(nan_mask, y_pred_mean, obs_t)
            s_inv = jnp.linalg.pinv(y_pred_cov)

            dist_y = dist.MultivariateNormal(loc=y_pred_mean, covariance_matrix=y_pred_cov)
            step_log_prob = dist_y.log_prob(obs_masked)
            ll_t = ll_tm1 + step_log_prob

            corrected_mean_t, corrected_cov_t = self._update(x_pred_mean, x_pred_cov, obs_masked, h_t, s_inv)

            carry_t = (t + 1, corrected_mean_t, corrected_cov_t, ll_t)
            return carry_t, (x_pred_mean, x_pred_cov, corrected_mean_t, corrected_cov_t)

        init_carry = (1, self.initial_mean, self.initial_cov, 0.0)
        final_carry, (predicted_means, predicted_covs, filtered_means, filtered_covs) = lax.scan(
            scan_fn, init_carry, observations
        )

        _, _, _, total_ll = final_carry

        return predicted_means, predicted_covs, filtered_means, filtered_covs, total_ll

    def filter(self, observations: jnp.ndarray) -> FilterResult:
        """
        Runs forward filtering over a sequence of observations, returning a named tuple
        FilterResult(means, covariances, log_likelihood).
        """

        (_, __, filtered_means, filtered_covs, total_ll) = self._forward_pass(observations)

        return FilterResult(filtered_means, filtered_covs, total_ll)

    # TODO: fix
    def smooth(self, observations: jnp.ndarray, missing_value: float = 1e12) -> SmoothingResult:
        """
        Runs forward filtering + RTS backward pass for smoothing, returning SmoothResult(means, covariances, log_likelihood).

        The backward pass uses a reverse-time lax.scan to fill in smoothed results for each time step.
        """
        predicted_means, predicted_covs, filter_means, filter_covs, ll = self._forward_pass(observations)

        def rts_step(carry, aux_t):
            """
            carry: (mean_next, cov_next, smeans, scovs)
            t_rev: the reversed time index, e.g. from T-2 down to 0
            """
            mean_next, cov_next = carry
            t, mean_f, cov_f, mean_p, cov_p = aux_t

            f_t = self._get_transition_matrix(t + 1, mean_f)
            cov_inv = jnp.linalg.pinv(cov_p)

            a_t = cov_f @ f_t.T @ cov_inv
            curr_mean_smooth = mean_f + a_t @ (mean_next - mean_p)
            curr_cov_smooth = cov_f + a_t @ (cov_next - cov_p) @ a_t.T

            return (curr_mean_smooth, curr_cov_smooth), (curr_mean_smooth, curr_cov_smooth)

        carry_init = (filter_means[-1], filter_covs[-1])

        num_timesteps = observations.shape[0]
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

            f_t = self._get_transition_matrix(t, x_prev)
            q_t = self._get_transition_cov(t)
            b_t = self._get_transition_offset(t)
            g_t = self.noise_transform
            h_t = self._get_observation_matrix(t)
            r_t = self._get_observation_cov(t)
            d_t = self._get_observation_offset(t)

            rng_proc, rng_obs = jax.random.split(rng_prev)
            noise_dim = q_t.shape[0]
            w_t = dist.MultivariateNormal(loc=jnp.zeros(noise_dim), covariance_matrix=q_t).sample(rng_proc)

            x_t = f_t @ x_prev + b_t + g_t @ w_t
            obs_dim = r_t.shape[0]
            v_t = dist.MultivariateNormal(loc=jnp.zeros(obs_dim), covariance_matrix=r_t).sample(rng_obs)

            y_t = h_t @ x_t + d_t + v_t

            return (t + 1, x_t, rng_proc), (x_t, y_t)

        def init_state(key):
            return dist.MultivariateNormal(loc=self.initial_mean, covariance_matrix=self.initial_cov).sample(key)

        x0 = init_state(rng_key)
        init_carry = (0, x0, rng_key)

        _, (xs, ys) = lax.scan(sample_step, init_carry, None, length=num_timesteps)

        return xs, ys


__all__ = ["KalmanFilter"]
