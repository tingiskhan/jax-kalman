import jax.numpy as jnp
from jaxman import KalmanFilter
import pytest as pt
import pykalman as pk
import numpy as np
import jax.random as jxrnd


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

    trans_mat = np.eye(1, 5)
    trans_mat = np.concatenate((trans_mat, np.eye(4, 5)))
    trans_cov = 0.1 * trans_mat

    obs_mat = np.ones(trans_mat.shape[-1]).cumsum()[::-1]
    obs_mat /= obs_mat.sum()
    
    yield (
        pk.KalmanFilter(transition_covariance=trans_cov, transition_matrices=trans_mat, observation_covariance=obs_cov, observation_matrices=obs_mat),
        KalmanFilter(trans_mat, trans_cov, obs_mat, obs_cov) 
    )


class TestKalman(object):
    EPS = 1e-6

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
        assert c.covariance.shape == (*batch_shape, shape[0], shape[0])

    @pt.mark.parametrize("pykf_kf", kalman_filters_with_pykalman())
    @pt.mark.parametrize("replicas", [1, 10, 100])
    def test_compare_with_pykalman(self, pykf_kf, replicas):
        pykf, kf = pykf_kf

        m = list()
        c = list()

        _, y = kf.sample(100, jxrnd.PRNGKey(123), (replicas,))
        for i in range(replicas):            
            m_, c_ = pykf.filter(y[:, i])

            m.append(m_)
            c.append(c_)

        m = np.stack(m, axis=1)
        c = np.stack(c, axis=1)

        jax_result = kf.filter(y)
        m_jax, c_jax = jax_result.filtered()

        # NB: We only compare the last 50% of the series as the approach is somewhat different.
        ind = int(y.shape[0] * 0.5)

        m_error = error(m[ind:], m_jax[ind:])
        c_error = error(c[ind:], c_jax[ind:])
        
        assert m_error < self.EPS and c_error < self.EPS
