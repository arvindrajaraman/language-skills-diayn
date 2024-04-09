from functools import partial

from icecream import ic
from jax import jit
import jax
import jax.numpy as jnp

@jit
def classify_goal(x):
    conditions = jnp.array([
        x < -0.33,
        (x >= -0.33) & (x < 0.33),
        x >= 0.33
    ])
    choices = jnp.array([0, 1, 2])
    return jnp.sum(conditions.T * choices, axis=1, dtype=jnp.int32)

@jit
def embedding_1he(x):
    return jnp.array([
        x < -0.33,
        (x >= -0.33) & (x < 0.33),
        x >= 0.33
    ], dtype=jnp.float32).T
    
def normalize_freq_matrix(freq_matrix):
    if jnp.sum(freq_matrix) < 1e-9:
        return jnp.zeros((3, 3))
    
    conf_matrix = freq_matrix / (jnp.sum(freq_matrix, axis=1, keepdims=True) + 1e-9)
    return conf_matrix

def I_zs_score(freq_matrix):
    freq_matrix = normalize_freq_matrix(freq_matrix)

    total_freq = jnp.sum(freq_matrix)
    if total_freq < 1e-9:
        return 0.0
    
    p_y = jnp.sum(freq_matrix, axis=0) / total_freq
    p_yhat = jnp.sum(freq_matrix, axis=1) / total_freq
    p_y_yhat = freq_matrix / total_freq
    
    mi = 0.0
    for i in range(3):
        for j in range(3):
            if p_y_yhat[i, j] > 0 and p_y[i] > 0 and p_yhat[j] > 0:
                mi += p_y_yhat[i, j] * jnp.log(p_y_yhat[i, j] / (p_y[i] * p_yhat[j] + 1e-9))

    return mi

@partial(jit, static_argnames=('qlocal', 'batch_size', 'skill_size', 'action_size'))
def I_za_score(qlocal, qlocal_params, state, skill_gt, batch_size, skill_size, action_size):
    # H_a
    all_probs = jnp.empty((skill_size, batch_size, action_size))
    for skill_idx in range(skill_size):
        skill = jnp.tile(jax.nn.one_hot(skill_idx, skill_size), (batch_size, 1))
        logits = qlocal.apply(qlocal_params, state, skill, train=False)
        probs = jax.nn.softmax(logits)
        all_probs = all_probs.at[skill_idx].set(probs)
    
    averaged_probs = all_probs.mean(axis=0)
    H_a = jax.scipy.special.entr(averaged_probs).sum(axis=1).mean()

    # H_az
    logits = qlocal.apply(qlocal_params, state, skill_gt, train=False)
    probs = jax.nn.softmax(logits)
    H_az = jax.scipy.special.entr(probs).sum(axis=1).mean()

    return H_a - H_az
