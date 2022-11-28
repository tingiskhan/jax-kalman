import jax.numpy as jnp
from jaxman import KalmanFilter
import pytest as pt


def kalman_configurations():
    yield (
        (0.5, 1.0, 1.0, 1.0),
        (1, 1)
    )

    yield (
        (0.5 * jnp.eye(2), jnp.eye(2), jnp.ones(2), 1.0),
        (2, 1)
    )

    yield (
        (0.5 * jnp.eye(2), jnp.eye(2), jnp.ones((3, 2)), jnp.eye(3)),
        (2, 3)
    )


class TestKalman(object):
    @pt.mark.parametrize("conf_and_expected", kalman_configurations())
    def test_initializer(self, conf_and_expected):
        conf, exp = conf_and_expected
        kf = KalmanFilter(*conf)

        assert kf.trans_cov.shape == (exp[0], exp[0])
        assert kf.trans_mat.shape == kf.trans_cov.shape

        assert kf.obs_cov.shape == (exp[1], exp[1])
        assert kf.obs_mat.shape == (exp[1], exp[0])

    @pt.mark.parametrize("conf_and_expected", kalman_configurations())
    @pt.mark.parametrize("batch_shape", [(), (100,), (1_000,)])
    def test_predict_correct(self, conf_and_expected, batch_shape):
        conf, shape = conf_and_expected
        kf = KalmanFilter(*conf)

        c = kf.initialize()
        p = kf.predict(c)
        
        c = kf.correct(jnp.ones((*batch_shape, shape[-1])), p)
 