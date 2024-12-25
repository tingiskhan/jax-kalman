import os

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import Normal, Uniform, HalfNormal
from jaxman import KalmanFilter

numpyro.set_platform("cpu")
numpyro.set_host_device_count(os.cpu_count())


@pytest.fixture
def ar1_true_params():
    phi = 0.99
    sigma = 0.05
    mu = 0.0
    obs_std = 0.1
    return phi, sigma, mu, obs_std


@pytest.fixture
def simulated_data(ar1_true_params):
    phi, sigma, mu, obs_std = ar1_true_params
    n_timesteps = 500
    rng = np.random.RandomState(0)

    states = np.zeros(n_timesteps)
    states[0] = mu + sigma / np.sqrt(1.0 - phi**2.0) * rng.randn()

    alpha = mu * (1.0 - phi)

    for t in range(1, n_timesteps):
        states[t] = alpha + phi * states[t - 1] + sigma * rng.randn()

    observations = states + obs_std * rng.randn(n_timesteps)
    return states, observations


@jax.jit
def build_filter(A, C, Q, R, initial_mean, initial_cov, transition_offset, observation_offset):
    return KalmanFilter(
        transition_matrices=A,
        observation_matrices=C,
        transition_covariance=Q,
        observation_covariance=R,
        initial_state_mean=initial_mean,
        initial_state_covariance=initial_cov,
        transition_offset=transition_offset,
        observation_offset=observation_offset,
    )


def model(observations):
    phi = numpyro.sample("phi", Uniform(0.0, 1.0))
    sigma = numpyro.sample("sigma", HalfNormal())
    mu = numpyro.sample("mu", Normal())

    obs_std = jnp.array(0.1)

    A = phi[None, None]
    Q = sigma[None, None] ** 2

    C = jnp.array([[1.0]])
    R = obs_std[None, None] ** 2

    transition_offset = (mu * (1.0 - phi))[None]
    observation_offset = None

    initial_mean = mu[None]
    initial_cov = (sigma**2.0 / (1.0 - phi**2.0))[None, None]

    kf = build_filter(A, C, Q, R, initial_mean, initial_cov, transition_offset, observation_offset)

    means, _, ll = kf.filter(observations)
    numpyro.factor("obs_ll", ll)
    numpyro.deterministic("x", means)

    return


@pytest.mark.parametrize("num_warmup,num_samples", [(500, 500)])
def test_numpyro_parameter_inference(simulated_data, ar1_true_params, num_warmup, num_samples):
    states, observations = simulated_data
    phi_true, sigma_true, mu_true, obs_std_true = ar1_true_params

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, chain_method="parallel", num_chains=4)
    mcmc.run(jax.random.PRNGKey(0), observations)
    samples = mcmc.get_samples()

    phi_est = np.mean(samples["phi"])
    sigma_est = np.mean(samples["sigma"])
    mu_est = np.mean(samples["mu"])

    # Check that posterior means are close to true parameters
    # With enough samples and a well-posed model, they should be reasonably close.
    assert jnp.abs(phi_est - phi_true) < 0.05, f"phi estimate off: {phi_est}, true: {phi_true}"
    assert jnp.abs(sigma_est - sigma_true) < 0.01, f"sigma estimate off: {sigma_est}, true: {sigma_true}"
    assert jnp.abs(mu_est - mu_true) < 0.05, f"mu estimate off: {mu_est}, true: {mu_true}"
