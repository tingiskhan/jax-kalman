import pytest as pt
import jax.numpy as jnp
import jaxman as jxk



def covariances():
    yield jnp.array(1.0), (1, 1)
    yield jnp.ones(1), (1, 1)
    yield jnp.eye(2), (2, 2)


def matrices():
    yield 1.0, (1, 1), jnp.ones((1, 1))
    yield jnp.array([1.0, 2.0]), (1, 2), jnp.array([[1.0, 2.0]])
    yield jnp.eye(4), (4, 4), jnp.eye(4)



class TestUtils(object):
    @pt.mark.parametrize("cov_and_shape", covariances())
    def test_coerce_covariance(self, cov_and_shape):
        cov, shape = cov_and_shape
        cov = jxk.utils.coerce_covariance(cov)

        assert cov.shape == shape

    @pt.mark.parametrize("a", matrices())
    def test_coerce_matrix(self, a):
        x, shape, expected = a

        x = jxk.utils.coerce_matrix(x, *shape)
        assert (x == expected).all()
    