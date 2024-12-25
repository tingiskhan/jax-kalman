import pytest
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jaxman import KalmanFilter


@pytest.fixture
def random_key():
    return jax.random.PRNGKey(0)


def _simulate_data(true_params, rng_key, num_timesteps=30):
    F = jnp.array([[true_params["F"]]])      # shape (1,1)
    Q = jnp.array([[true_params["Q"]]])      # shape (1,1)
    R = jnp.array([[true_params["R"]]])      # shape (1,1)

    kf = KalmanFilter(
        initial_mean=jnp.array([true_params["initial_mean"]]),
        initial_cov=jnp.array([[true_params["initial_cov"]]]),
        transition_matrix=F,
        transition_cov=Q,
        observation_matrix=jnp.array([[1.0]]),
        observation_cov=R
    )
    xs, ys = kf.sample(rng_key, num_timesteps=num_timesteps)
    return xs, ys


def test_infer_state_space_parameters_with_numpyro(random_key):
    true_params = {
        "F": 0.95,
        "Q": 0.1,
        "R": 0.5,
        "initial_mean": 0.0,
        "initial_cov": 1.0
    }

    rng_key, data_key = jax.random.split(random_key)
    xs, ys = _simulate_data(true_params, data_key, num_timesteps=100)

    def model(observations):
        F = numpyro.sample("F", dist.Beta(2.0, 1.0))

        sigma_dist = dist.TransformedDistribution(
            dist.Normal().expand((2,)),
            [dist.transforms.OrderedTransform(), dist.transforms.ExpTransform()]
        )
        Q, R = numpyro.sample("sigmas", sigma_dist)

        kf = KalmanFilter(
            initial_mean=jnp.array([true_params["initial_mean"]]),
            initial_cov=jnp.array([[true_params["initial_cov"]]]),
            transition_matrix=jnp.array([[F]]),
            transition_cov=jnp.array([[Q]]),
            observation_matrix=jnp.array([[1.0]]),
            observation_cov=jnp.array([[R]])
        )

        ll = kf.filter(observations)[2]
        numpyro.factor("kalman_likelihood", ll)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1_000, num_samples=500, num_chains=2, progress_bar=False)

    rng_key, mcmc_key = jax.random.split(rng_key)
    mcmc.run(mcmc_key, observations=ys)

    posterior_samples = mcmc.get_samples()

    F_mean = jnp.mean(posterior_samples["F"])
    Q_mean, R_mean = jnp.mean(posterior_samples["sigmas"], axis=0)

    assert pytest.approx(true_params["F"], abs=0.1) == F_mean
    assert pytest.approx(true_params["Q"], abs=0.02) == Q_mean
    assert pytest.approx(true_params["R"], abs=0.05) == R_mean
