from jax import random
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm
from collections import deque
from typing import NamedTuple
from flax.dataclass import dataclass
from flax.training import train_state
from functools import partial
from typing import Tuple, List
import ml_collections

from structures import *
from models import QSkillNet, Discriminator

@partial(jit, static_argnames=('qlocal', 'action_size'))
def diayn_act(
    key: random.PRNGKey,
    qlocal: QSkillNet,
    qlocal_params: optax.Params,
    state: jnp.ndarray,
    skill: jnp.ndarray,
    action_size: int,
    eps: float = 0.0
) -> Tuple[random.PRNGKey, jnp.ndarray]:
    def _random_action(subkey):
        action = random.choice(subkey, jnp.arange(action_size))
        return action

    def _greedy_action(_):
        qvalues = qlocal.apply(qlocal_params, state, skill)
        action = jnp.argmax(qvalues, axis=-1)
        return action
    
    key, subkey = random.split(key)
    explore = eps > random.uniform(subkey)
    
    key, subkey = random.split(key)
    action = jax.lax.cond(explore, _random_action, _greedy_action, operand=subkey)

    return key, action

def diayn_train(
    key: random.PRNGKey,
    config: ml_collections.FrozenConfigDict
):
    # 1. Initialize training state
    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    qlocal = QSkillNet(
        action_size=config.action_size,
        hidden1_size=config.policy_hidden_size,
        hidden2_size=config.policy_hidden_size,
        dropout_rate=config.dropout_rate
    )
    key, subkey = jax.random.split(key)
    qlocal_params = qlocal.init(subkey, dummy_state)
    qlocal_opt = optax.adam(
        learning_rate=config.policy_lr
    )

    key, subkey = jax.random.split(key)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QSkillNet(
        action_size=config.action_size,
        hidden1_size=config.policy_hidden_size,
        hidden2_size=config.policy_hidden_size,
        dropout_rate=config.dropout_rate
    )
    key, subkey = jax.random.split(key)
    qtarget_params = qtarget.init(subkey, dummy_state)

    discrim = Discriminator(
        skill_size=config.skill_size,
        hidden1_size=config.discim_hidden_size,
        hidden2_size=config.discim_hidden_size,
        dropout_rate=config.dropout_rate
    )
    key, subkey = jax.random.split(key)
    discrim_params = discrim.init(subkey, dummy_embedding, dummy_skill)
    discrim_opt = optax.adam(
        learning_rate=config.discrim_lr
    )

    key, subkey = jax.random.split(key)
    discrim_opt_state = discrim_opt.init(discrim_params)

    # 2. Run training
    