from dataclasses import dataclass, field
from typing import List, Tuple
import jax.numpy as jnp


@dataclass(frozen=True)
class Prediction(object):
    mean: jnp.ndarray
    covariance: jnp.ndarray


@dataclass(frozen=True)
class Correction(object):
    mean: jnp.ndarray
    covariance: jnp.ndarray
    gain: jnp.ndarray
    loglikelihood: jnp.ndarray


@dataclass
class Result(object):
    predictions: List[Prediction] = field(default_factory=list, repr=False)
    corrections: List[Correction] = field(default_factory=list, repr=False)

    def append(self, p: Prediction, c: Correction):
        """
        Appends prediction and correction.

        Args:
            p (Prediction): prediction to append.
            c (Correction): correction to append.
        """

        self.predictions.append(p)
        self.corrections.append(c)

    def filtered_means(self) -> jnp.ndarray:
        """
        Returns the filtered means of the result.

        Returns:
            jnp.ndarray: filtered means.
        """

        means = [c.mean for c in self.corrections]

        return jnp.stack(means)

    def filtered_covariances(self) -> jnp.ndarray:
        """
        Returns the filtered covariances of the result.

        Returns:
            jnp.ndarray: filtered covariances.
        """

        covs = [c.covariance for c in self.corrections]

        return jnp.stack(covs)

    def filtered(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Returns both the filtered means and covariances.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray]: filtered means and covariances.
        """

        return self.filtered_means(), self.filtered_covariances()
