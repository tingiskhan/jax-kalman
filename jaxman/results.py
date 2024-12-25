from typing import NamedTuple
import jax.numpy as jnp


class FilterResult(NamedTuple):
    means: jnp.ndarray
    covariances: jnp.ndarray
    log_likelihood: jnp.ndarray


class SmoothResult(NamedTuple):
    means: jnp.ndarray
    covariances: jnp.ndarray
    log_likelihood: jnp.ndarray