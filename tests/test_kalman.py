import jax.numpy as jnp
from jaxman import KalmanFilter
import pytest as pt
import pykalman as pk
import numpy as np


def error(y_true, y_hat):
    return jnp.abs((y_true - y_hat) / y_true).mean()



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


def kalman_filters_with_pykalman():
    trans_mat = 1.0
    obs_mat = 1.0

    trans_cov = 0.1
    obs_cov = 0.1
    
    yield (
        pk.KalmanFilter(transition_covariance=trans_cov, transition_matrices=trans_mat, observation_covariance=obs_cov, observation_matrices=obs_mat),
        KalmanFilter(trans_mat, trans_cov, obs_mat, obs_cov)
    )


class TestKalman(object):
    # TODO: Figure out a better eps
    EPS = 1e-3

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

        assert c.mean.shape == (*batch_shape, shape[0])
        assert c.covariance.shape == (shape[0], shape[0])

    @pt.mark.parametrize("pykf_kf", kalman_filters_with_pykalman())
    @pt.mark.parametrize("replicas", [1, 10, 50])
    def test_compare_with_pykalman(self, pykf_kf, replicas):
        pykf, kf = pykf_kf

        _, y_ = pykf.sample(100)

        y = np.stack(replicas * (y_,), axis=1).squeeze(1)

        m = np.empty((y.shape[0], replicas, y.shape[-1]))
        c = np.empty(((y.shape[0], replicas, y.shape[-1], y.shape[-1])))
        
        m_, c_ = pykf.filter(y_)
        # TODO: Use repeat...

        if replicas > 1:
            for i in range(replicas):            
                m[:, i] = m_
                c[:, i] = c_
        else:
            m = m_
            c = c_

        jax_result = kf.filter(y)
        m_jax = jax_result.filtered_means()
        c_jax = jax_result.filtered_covariances()

        m_error = error(m, m_jax)
        c_error = error(c, c_jax)
        
        assert m_error < self.EPS and c_error < self.EPS
