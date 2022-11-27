from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class Prediction(object):
    mean: jnp.ndarray
    covariance: jnp.ndarray


@dataclass
class Correction(object):
    mean: jnp.ndarray
    covariance: jnp.ndarray
    gain: jnp.ndarray
    loglikelihood: jnp.ndarray

