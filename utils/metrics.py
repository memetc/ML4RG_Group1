import jax.numpy as jnp
from jax import jit


# Mean Absolute Error (MAE)
@jit
def mean_absolute_error(y_true, y_pred):
    return jnp.mean(jnp.abs(y_true - y_pred))


# Mean Squared Error (MSE)
@jit
def mean_squared_error(y_true, y_pred):
    return jnp.mean(jnp.square(y_true - y_pred))


# Root Mean Squared Error (RMSE)
@jit
def root_mean_squared_error(y_true, y_pred):
    return jnp.sqrt(mean_squared_error(y_true, y_pred))


# R-squared (R²)
@jit
def r2_score(y_true, y_pred):
    ss_res = jnp.sum(jnp.square(y_true - y_pred))
    ss_tot = jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
    return 1 - ss_res / ss_tot


# Adjusted R-squared
@jit
def adjusted_r2_score(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    return 1 - ((1 - r2) * (n - 1)) / (n - k - 1)


# Mean Absolute Percentage Error (MAPE)
@jit
def mean_absolute_percentage_error(y_true, y_pred):
    return jnp.mean(jnp.abs((y_true - y_pred) / y_true)) * 100


# Mean Bias Deviation (MBD)
@jit
def mean_bias_deviation(y_true, y_pred):
    return jnp.mean(y_true - y_pred)


# Explained Variance Score
@jit
def explained_variance_score(y_true, y_pred):
    return 1 - jnp.var(y_true - y_pred) / jnp.var(y_true)
