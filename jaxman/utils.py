from typing import Sequence, Tuple
import jax.numpy as jnp
import jax
from functools import partial

from .typing import ArrayLike
from .result import Correction


def to_array(*x: ArrayLike) -> Sequence[jnp.ndarray]:
    """
    Coerces all inputs to be a `jax.numpy.ndarray`.

    Returns:
        jnp.ndarray: a sequence of `jax.numpy.ndarray`. 
    """

    return tuple(jnp.array(x_) for x_ in x)


def coerce_covariance(cov: jnp.ndarray) -> jnp.ndarray:
    """
    Coerces the covariance to be a proper covariance matrix.

    Args:
        cov (jnp): covariance matrix.

    Returns:
        jnp.ndarray: coerced covariance.
    """

    if cov.ndim >= 2:
        return cov

    dim = 1 if cov.ndim == 0 else cov.shape[0]
    
    return cov * jnp.eye(dim)


@partial(jax.jit, static_argnums=(1, 2))
def coerce_matrix(a: jnp.ndarray, left_dim: int, right_dim: int) -> jnp.ndarray:
    """
    Coerces the transition/observation matrix.

    Args:
        a (jnp.ndarray): transition/observation matrix.
        left_dim (int): dimension of first.
        right_dim (int): dimension of right.

    Returns:
        jnp.ndarray: coerced matrix.
    """

    shape = (left_dim, right_dim)
    shape = tuple(d for d in shape if d != 0)

    return jnp.ones(shape) * a


@jax.jit
def predict(x: jnp.ndarray, p: jnp.ndarray, f: jnp.ndarray, q: jnp.ndarray):
    """
    Predicts the mean and covariance, see [here](https://en.wikipedia.org/wiki/Kalman_filter)

    Args:
        x (jnp.ndarray): previous state.
        p (jnp.ndarray): previous covariance.
        f (jnp.ndarray): state transition model.
        q (jnp.ndarray): state transition covariance

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: mean and covariance.
    """

    mean = (f @ x[..., None]).squeeze(-1)
    cov = f @ p @ f.transpose(-1, -2) + q

    return mean, cov


@jax.jit
def correct(x: jnp.ndarray, h: jnp.ndarray, p: jnp.ndarray, r: jnp.ndarray, y: jnp.ndarray):
    """
    Corrects the Kalman step, see [here](https://en.wikipedia.org/wiki/Kalman_filter).

    Args:
        x (jnp.ndarray): predicted state mean.
        h (jnp.ndarray): observation model.
        p (jnp.ndarray): predicted state covariance.
        r (jnp.ndarray): observation covariance.
        y (jnp.ndarray): observation.
    """

    z_pred = y - (h @ x[..., None]).squeeze(-1)
    
    h_transpose = h.transpose(-1, -2)
    s_pred = h @ p @ h_transpose + r

    inv_s_pred = jnp.linalg.pinv(s_pred)
    gain = p @ h_transpose @ inv_s_pred
    
    x_corr = x + (gain @ z_pred[..., None]).squeeze(-1)
    p_corr = (jnp.eye(x.shape[-1]) - gain @ h) @ p

    return x_corr, p_corr, gain
