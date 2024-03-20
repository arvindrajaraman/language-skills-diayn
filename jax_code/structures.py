import jax
import jax.numpy as jnp
import optax
from jax import random, jit, vmap
from jax.tree_util import tree_map
from flax import struct
from functools import partial
import ml_collections
from typing import Tuple, List

from models import QSkillNet, Discriminator

@struct.dataclass
class Experience:
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.ndarray
    next_state_embedding: jnp.ndarray
    skill: jnp.ndarray
    done: jnp.ndarray

@struct.dataclass
class BufferState:
    states: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_states: jnp.ndarray
    next_state_embeddings: jnp.ndarray
    skills: jnp.ndarray
    dones: jnp.ndarray
    size: int

def buffer_init(
    config: ml_collections.ConfigDict
) -> BufferState:
    return BufferState(
        states=jnp.empty((config.buffer_size, config.state_size), dtype=jnp.float32),
        actions=jnp.empty((config.buffer_size, config.action_size), dtype=jnp.int32),
        rewards=jnp.empty((config.buffer_size,), dtype=jnp.float32),
        next_states=jnp.empty((config.buffer_size, config.state_size), dtype=jnp.float32),
        next_state_embeddings=jnp.empty((config.buffer_size, config.embedding_size), dtype=jnp.float32),
        skills=jnp.empty((config.buffer_size, config.skill_size), dtype=jnp.float32),
        dones=jnp.empty((config.buffer_size,), dtype=jnp.bool_),
        size=0
    )

def buffer_add(
    config: ml_collections.ConfigDict,
    buffer_state: dict,
    experience: Experience,
    idx: int
) -> BufferState:
    idx = idx % config.buffer_size

    buffer_state['states'] = buffer_state["states"].at[idx].set(experience.state)
    buffer_state['actions'] = buffer_state["actions"].at[idx].set(experience.action)
    buffer_state['rewards'] = buffer_state["rewards"].at[idx].set(experience.reward)
    buffer_state['next_states'] = buffer_state["next_states"].at[idx].set(experience.next_state)
    buffer_state['next_states_embedding'] = buffer_state["next_states_embedding"].at[idx].set(experience.next_state_embedding)
    buffer_state['skills'] = buffer_state["skills"].at[idx].set(experience.skill)
    buffer_state['dones'] = buffer_state["dones"].at[idx].set(experience.done)
    buffer_state['size'] = min(buffer_state['size'] + 1, config.buffer_size)

    return buffer_state

def buffer_sample(
    key: random.PRNGKey,
    config: ml_collections.ConfigDict,
    buffer_state: dict
) -> Tuple[random.PRNGKey, List[Experience]]:
    @partial(vmap, in_axes=(0, None))
    def sample_batch(indexes, buffer):
        """
        For a given index, extracts all the values from the buffer
        """
        return tree_map(lambda x: x[indexes], buffer)
    
    key, subkey = random.split(key)
    indexes = random.randint(
        subkey,
        shape=(config.batch_size,),
        minval=0,
        max_val=buffer_state['size']
    )
    experiences = sample_batch(indexes, buffer_state)
    
    return subkey, experiences

