import jax.numpy as jnp
import numpy as np
from typing import Tuple
import jax.random as jrnd


from .typing import ArrayLike
from .utils import to_array, coerce_covariance, coerce_matrix, predict, correct
from .result import Correction, Prediction, Result


class KalmanFilter(object):
    r"""
    Implements a standard [Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter) which enables [Single Instruction Multiple Data](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) in JAX.
    Inspired by [simdkalman](https://github.com/oseiskar/simdkalman) and [pykalman](https://pykalman.github.io/).
    """

    def __init__(
        self,
        trans_mat: ArrayLike,
        trans_cov: ArrayLike,
        obs_mat: ArrayLike,
        obs_cov: ArrayLike,
        init_mean: ArrayLike = None,
        init_cov: ArrayLike = None,
    ):
        """
        Internal initializer for :class:`KalmanFilter`.

        Args:
            trans_mat (ArrayLike): transition matrix.
            trans_cov (ArrayLike): transition covariance.
            obs_mat (ArrayLike): observation matrix.
            obs_cov (ArrayLike): observation covariance.
            init_mean (ArrayLike, optional): initial mean. Defaults to None.
            init_cov (ArrayLike, optional): initial covariance. Defaults to None.
        """

        trans_mat, trans_cov, obs_mat, obs_cov = to_array(trans_mat, trans_cov, obs_mat, obs_cov)

        self.trans_cov = coerce_covariance(trans_cov)
        self.trans_ndim = self.trans_cov.shape[-1]
        self.trans_mat = coerce_matrix(trans_mat, self.trans_cov.shape[-2], self.trans_cov.shape[-1])

        self.obs_cov = coerce_covariance(obs_cov)
        self.obs_ndim = self.obs_cov.shape[-1]
        self.obs_mat = coerce_matrix(obs_mat, self.obs_cov.shape[-1], self.trans_cov.shape[-1])

        if init_mean is None:
            init_mean = jnp.zeros(self.trans_mat.shape[-1])

        if init_cov is None:
            init_cov = jnp.eye(self.trans_cov.shape[-1])

        self.init_cov = coerce_covariance(init_cov)
        self.init_mean = coerce_matrix(init_mean, 0, self.trans_mat.shape[-1])

    def initialize(self) -> Correction:
        """
        Generates the initial correction.

        Returns:
            Correction: initial correction.
        """

        return Correction(self.init_mean, self.init_cov, 0.0, 0.0)

    def predict(self, correction: Correction) -> Prediction:
        """
        Predicts from the latest correction.

        Args:
            correction (Correction): latest correction.

        Returns:
            Prediction: new prediction.
        """

        mean, cov = predict(correction.mean, correction.covariance, self.trans_mat, self.trans_cov)

        return Prediction(mean, cov)

    def correct(self, y: jnp.ndarray, prediction: Prediction) -> Correction:
        """
        Corrects latest prediction.

        Args:
            y (jnp.ndarray): latest observation.
            prediction (Prediction): latest prediction.

        Returns:
            Correction: corrected Kalman state.
        """

        mean, cov, gain = correct(prediction.mean, self.obs_mat, prediction.covariance, self.obs_cov, y)

        # TODO: Fix log-likelihood
        return Correction(mean, cov, gain, 0.0)

    def filter(self, y: jnp.ndarray) -> Result:
        """
        Filters the data `y`.

        Args:
            y (jnp.ndarray): data to filter, must be of shape `{time, [batch], [dim]}`

        Returns:
            Result: result object.
        """

        c = self.initialize()
        result = Result()

        result.append(None, c)

        for yt in y:
            p = self.predict(c)
            c = self.correct(yt, p)

            result.append(p, c)

        return result

    def sample(
        self, timesteps: int, prng_key: jrnd.PRNGKey, batch_shape: tuple = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Samples the LDS `timesteps` into the future.

        Args:
            timesteps (int): number of future timesteps.
            prng_key (jax.random.PRNGKey): random number generator key.
            batch_shape (tuple): batch shape.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: returns the sampled states for the latent and observable processes.
        """

        x = jrnd.multivariate_normal(prng_key, self.init_mean, self.init_cov, batch_shape)

        x_res = tuple()
        y_res = tuple()

        for t in range(timesteps):
            _, prng_key = jrnd.split(prng_key)

            x = jrnd.multivariate_normal(
                prng_key, (self.trans_mat @ x[..., None]).squeeze(-1), self.trans_cov, method="svd"
            )
            y = jrnd.multivariate_normal(
                prng_key, (self.obs_mat @ x[..., None]).squeeze(-1), self.obs_cov, method="svd"
            )

            x_res += (x,)
            y_res += (y,)

        return jnp.stack(x_res), jnp.stack(y_res)
