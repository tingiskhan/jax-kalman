import jax
import jax.numpy as jnp
import pytest
from jaxman import KalmanFilter


@pytest.fixture
def simple_kf():
    return KalmanFilter(
        transition_matrices=jnp.array([[1.0]]),
        observation_matrices=jnp.array([[1.0]]),
        transition_covariance=jnp.array([[1e-2]]),
        observation_covariance=jnp.array([[1e-1]]),
        initial_state_mean=jnp.array([0.0]),
        initial_state_covariance=jnp.array([[1.0]])
    )


def test_filter_with_no_missing(simple_kf):
    true_states = jnp.linspace(0, 10, 50)
    observations = true_states + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (50,))

    filtered_means, filtered_covs, ll = simple_kf.filter(observations)
    assert filtered_means.shape == (50, 1)
    assert filtered_covs.shape == (50, 1, 1)
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)
    # Check that filtered means are somewhat close
    assert jnp.mean(jnp.abs(filtered_means.squeeze() - true_states)) < 0.5


def test_filter_with_missing(simple_kf):
    true_states = jnp.linspace(0, 10, 50)
    observations = true_states + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (50,))
    observations = observations.at[10].set(jnp.nan)
    observations = observations.at[20].set(jnp.nan)

    filtered_means, filtered_covs, ll = simple_kf.filter(observations)
    assert filtered_means.shape == (50, 1)
    assert filtered_covs.shape == (50, 1, 1)
    assert not jnp.isnan(filtered_means).all()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)


def test_smooth_with_missing(simple_kf):
    true_states = jnp.linspace(0, 10, 50)
    observations = true_states + 0.1 * jax.random.normal(jax.random.PRNGKey(0), (50,))
    observations = observations.at[10].set(jnp.nan)
    observations = observations.at[20].set(jnp.nan)

    smoothed_means, smoothed_covs, ll = simple_kf.smooth(observations)
    assert smoothed_means.shape == (50, 1)
    assert smoothed_covs.shape == (50, 1, 1)
    assert not jnp.isnan(smoothed_means).all()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)


def test_filter_update(simple_kf):
    fm = simple_kf.initial_state_mean
    fc = simple_kf.initial_state_covariance
    obs = jnp.array([1.0])
    new_mean, new_cov, ll = simple_kf.filter_update(fm, fc, obs)
    assert new_mean.shape == (1,)
    assert new_cov.shape == (1, 1)
    assert not jnp.isnan(new_mean).any()
    assert not jnp.isnan(new_cov).any()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)
