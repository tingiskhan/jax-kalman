from jaxman import KalmanFilter
import jax.random as jaxrnd

trans_mat = 1.0
trans_cov = 0.05 ** 2

obs_mat = 1.0
obs_cov = 0.1 ** 2

kf = KalmanFilter(trans_mat, trans_cov, obs_mat, obs_cov)

_, y = kf.sample(100, jaxrnd.PRNGKey(123), batch_shape=(100_000,))


result = kf.filter(y)