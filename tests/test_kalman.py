import jax
import jax.numpy as jnp
import pytest

from jaxman import KalmanFilter


@pytest.fixture
def kf_params():
    """
    Define consistent parameters for the Kalman filter and data generation.
    """
    n_dim_state = 1
    n_dim_obs = 1

    # Filter parameters
    transition_matrices = jnp.array([[1.0]])  # Identity transition
    observation_matrices = jnp.array([[1.0]])  # Direct observation of the state
    transition_covariance = jnp.array([[1e-1]])  # State transition noise variance
    observation_covariance = jnp.array([[1e-1]])  # Observation noise variance = 0.1

    # Initial state parameters
    initial_state_mean = jnp.array([0.0])
    initial_state_covariance = jnp.array([[1.0]])

    # For data generation, we need standard deviation from covariance
    obs_noise_std = jnp.sqrt(observation_covariance[0, 0])  # sqrt(0.1) ~ 0.316
    return {
        "transition_matrices": transition_matrices,
        "observation_matrices": observation_matrices,
        "transition_covariance": transition_covariance,
        "observation_covariance": observation_covariance,
        "initial_state_mean": initial_state_mean,
        "initial_state_covariance": initial_state_covariance,
        "obs_noise_std": obs_noise_std,
    }


@pytest.fixture
def simple_kf(kf_params):
    return KalmanFilter(
        transition_matrices=kf_params["transition_matrices"],
        observation_matrices=kf_params["observation_matrices"],
        transition_covariance=kf_params["transition_covariance"],
        observation_covariance=kf_params["observation_covariance"],
        initial_state_mean=kf_params["initial_state_mean"],
        initial_state_covariance=kf_params["initial_state_covariance"],
    )


def test_filter_with_no_missing(simple_kf, kf_params):
    # Generate synthetic data with no missing observations
    key = jax.random.PRNGKey(123)
    true_states = jnp.linspace(0, 10, 50)
    noise = jax.random.normal(key, (50,))
    observations = true_states + kf_params["obs_noise_std"] * noise

    filtered_means, filtered_covs, ll = simple_kf.filter(observations)
    assert filtered_means.shape == (50, 1)
    assert filtered_covs.shape == (50, 1, 1)
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)

    # Check that filtered means are close to true states
    # With given parameters, we expect decent accuracy
    mae = jnp.mean(jnp.abs(filtered_means.squeeze() - true_states))
    assert mae < 0.5, f"Mean absolute error is too large: {mae}"


def test_filter_with_missing(simple_kf, kf_params):
    key = jax.random.PRNGKey(1)
    true_states = jnp.linspace(0, 10, 50)
    noise = jax.random.normal(key, (50,))
    observations = true_states + kf_params["obs_noise_std"] * noise

    # Introduce missing data
    observations = observations.at[10].set(jnp.nan)
    observations = observations.at[20].set(jnp.nan)

    filtered_means, filtered_covs, ll = simple_kf.filter(observations)
    assert filtered_means.shape == (50, 1)
    assert filtered_covs.shape == (50, 1, 1)
    assert not jnp.isnan(filtered_means).all()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)

    # Check that even with missing observations, the estimates are not too far off
    mae = jnp.mean(jnp.abs(filtered_means.squeeze() - true_states))
    # Relax criterion since missing data can degrade performance
    assert mae < 1.0, f"Mean absolute error is too large with missing data: {mae}"


def test_smooth_with_missing(simple_kf, kf_params):
    key = jax.random.PRNGKey(2)
    true_states = jnp.linspace(0, 10, 50)
    noise = jax.random.normal(key, (50,))
    observations = true_states + kf_params["obs_noise_std"] * noise

    # Introduce missing data
    observations = observations.at[10].set(jnp.nan)
    observations = observations.at[20].set(jnp.nan)

    smoothed_means, smoothed_covs, ll = simple_kf.smooth(observations)
    assert smoothed_means.shape == (50, 1)
    assert smoothed_covs.shape == (50, 1, 1)
    assert not jnp.isnan(smoothed_means).all()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)

    # Smoothing should improve estimates compared to filtering
    mae = jnp.mean(jnp.abs(smoothed_means.squeeze() - true_states))
    assert mae < 1.0, f"Mean absolute error too large after smoothing with missing data: {mae}"


def test_filter_update(simple_kf, kf_params):
    # Single-step update test
    fm = simple_kf.initial_state_mean
    fc = simple_kf.initial_state_covariance

    key = jax.random.PRNGKey(3)
    obs = 1.0 + kf_params["obs_noise_std"] * jax.random.normal(key, (1,))

    new_mean, new_cov, ll = simple_kf.filter_update(fm, fc, obs)
    assert new_mean.shape == (1,)
    assert new_cov.shape == (1, 1)
    assert not jnp.isnan(new_mean).any()
    assert not jnp.isnan(new_cov).any()
    assert isinstance(ll, float) or isinstance(ll, jnp.ndarray)
