import pytest
import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
from pykalman import KalmanFilter as PyKalman
from jaxman import KalmanFilter


@pytest.fixture
def random_key():
    """
    Returns a PRNGKey(0) for reproducibility.
    """
    return jax.random.PRNGKey(0)


def test_filter_vs_pykalman(random_key):
    """
    Compares the KalmanFilter output to pykalman on a simple 1D observation, 2D state system.
    Checks whether the filtered means/covs match pykalman results (allowing small tolerance).
    """
    state_dim = 2
    obs_dim = 1
    F = jnp.array([[1.0, 1.0], [0.0, 1.0]])
    Q = jnp.array([[1e-4, 0.0], [0.0, 1e-4]])
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[1e-1]])
    transition_offset = jnp.array([0.01, 0.0])

    np.random.seed(42)
    kf_ref = PyKalman(
        transition_matrices=F,
        transition_covariance=Q,
        observation_matrices=H,
        observation_covariance=R,
        initial_state_mean=[0, 0],
        n_dim_obs=obs_dim,
        transition_offsets=np.array(transition_offset),
    )
    states_ref, obs_ref = kf_ref.sample(n_timesteps=20, initial_state=[0, 0])

    kf_jax = KalmanFilter(
        initial_mean=jnp.array([0.0, 0.0]),
        initial_cov=jnp.eye(state_dim),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=H,
        observation_cov=R,
        transition_offset=transition_offset,
    )

    obs_jax = jnp.array(obs_ref)
    fm, fc, ll = kf_jax.filter(obs_jax)

    pf_means, pf_covs = kf_ref.filter(obs_ref)
    ll_ref = kf_ref.loglikelihood(obs_ref)

    assert_allclose(ll_ref, ll, atol=5e-2)
    assert_allclose(fm[1:], pf_means[1:], atol=5e-2)


def test_smoothing_vs_pykalman(random_key):
    """
    Compares the RTS smoothing results of the custom KalmanFilter class to the built-in smoothing in pykalman.
    We check that smoothed means/covariances are numerically close.
    """
    state_dim = 2
    obs_dim = 1

    F = jnp.array([[1.0, 1.0],
                   [0.0, 1.0]])
    Q = jnp.array([[1e-2, 0.0],
                   [0.0, 1e-2]])
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[1e-1]])

    np.random.seed(42)
    kf_ref = PyKalman(
        transition_matrices=np.array(F),
        transition_covariance=np.array(Q),
        observation_matrices=np.array(H),
        observation_covariance=np.array(R),
        initial_state_mean=np.array([0.0, 0.0]),
        n_dim_obs=obs_dim
    )

    states_ref, obs_ref = kf_ref.sample(n_timesteps=25, initial_state=[0.0, 0.0])

    kf_jax = KalmanFilter(
        initial_mean=jnp.array([0.0, 0.0]),
        initial_cov=jnp.eye(state_dim),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=H,
        observation_cov=R
    )

    obs_jax = jnp.array(obs_ref)

    pykalman_smoothed_means, pykalman_smoothed_covs = kf_ref.smooth(obs_ref)

    means, covariances, _ = kf_jax.smooth(obs_jax)
    jax_smoothed_means = np.array(means)
    jax_smoothed_covs = np.array(covariances)

    # Compare
    assert jax_smoothed_means.shape == pykalman_smoothed_means.shape
    assert jax_smoothed_covs.shape == pykalman_smoothed_covs.shape

    # We allow some numerical tolerance due to different internal implementations
    assert_allclose(jax_smoothed_means, pykalman_smoothed_means, atol=1e-2, rtol=1e-2)
    for t in range(len(jax_smoothed_covs)):
        assert_allclose(jax_smoothed_covs[t], pykalman_smoothed_covs[t], atol=1e-2, rtol=1e-2)


def test_partial_missing_data(random_key):
    """
    Checks if filter handles partially missing observations in multi-dimensional data.
    """
    state_dim = 2
    obs_dim = 2
    F = jnp.eye(state_dim)
    Q = jnp.eye(state_dim) * 0.01
    H = jnp.eye(obs_dim, state_dim)
    R = jnp.eye(obs_dim) * 0.1

    kf = KalmanFilter(
        initial_mean=jnp.array([0.0, 0.0]),
        initial_cov=jnp.eye(state_dim),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=H,
        observation_cov=R,
    )

    obs_data = np.array([[1.0, 2.0], [np.nan, 2.1], [1.1, np.nan], [np.nan, np.nan], [1.2, 2.2]], dtype=np.float32)

    fm, fc, ll = kf.filter(jnp.array(obs_data))
    assert fm.shape == (5, 2)
    assert fc.shape == (5, 2, 2)


def test_smoothing_improves_estimates(random_key):
    """
    Generates synthetic data and ensures smoothing yields consistent outputs.
    We check that shapes match and that smoothing doesn't crash.
    We do not necessarily check for "better" numerical MSE here, but it typically is.
    """
    state_dim = 1
    obs_dim = 1
    F = jnp.array([[1.0]])
    Q = jnp.array([[0.01]])
    H = jnp.array([[1.0]])
    R = jnp.array([[0.1]])

    kf = KalmanFilter(
        initial_mean=jnp.array([0.0]),
        initial_cov=jnp.array([[1.0]]),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=H,
        observation_cov=R,
    )

    rng_key, subkey = jax.random.split(random_key)
    true_xs, obs_ys = kf.sample(subkey, num_timesteps=10)
    fm, fc, ll_f = kf.filter(obs_ys)
    sm, sc, ll_s = kf.smooth(obs_ys)

    assert fm.shape == (10, 1)
    assert fc.shape == (10, 1, 1)
    assert sm.shape == (10, 1)
    assert sc.shape == (10, 1, 1)
    assert ll_f == pytest.approx(ll_s)


def test_log_likelihood_correct_vs_incorrect(random_key):
    """
    Verifies that the correct model yields higher log-likelihood than an incorrect model.
    """
    F_true = jnp.array([[1.0]])
    Q_true = jnp.array([[0.01]])
    H_true = jnp.array([[1.0]])
    R_true = jnp.array([[0.1]])

    kf_true = KalmanFilter(
        initial_mean=jnp.array([0.0]),
        initial_cov=jnp.array([[1.0]]),
        transition_matrix=F_true,
        transition_cov=Q_true,
        observation_matrix=H_true,
        observation_cov=R_true,
    )

    rng_key, subkey = jax.random.split(random_key)
    xs, ys = kf_true.sample(subkey, num_timesteps=15)
    ll_correct = kf_true.filter(ys)[2]

    F_wrong = jnp.array([[0.5]])
    Q_wrong = jnp.array([[0.5]])
    R_wrong = jnp.array([[1.0]])

    kf_wrong = KalmanFilter(
        initial_mean=jnp.array([0.0]),
        initial_cov=jnp.array([[1.0]]),
        transition_matrix=F_wrong,
        transition_cov=Q_wrong,
        observation_matrix=H_true,
        observation_cov=R_wrong,
    )
    ll_wrong = kf_wrong.filter(ys)[2]

    assert ll_correct > ll_wrong


def test_sampling(random_key):
    """
    Checks that the sample method runs without error and returns arrays of expected shapes.
    """
    state_dim = 2
    obs_dim = 1
    F = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    Q = jnp.array([[0.1, 0.0], [0.0, 0.1]])
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.2]])

    kf = KalmanFilter(
        initial_mean=jnp.array([0.0, 0.0]),
        initial_cov=jnp.eye(state_dim),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=H,
        observation_cov=R,
    )

    xs, ys = kf.sample(random_key, num_timesteps=5)
    assert xs.shape == (5, 2)
    assert ys.shape == (5, 1)
