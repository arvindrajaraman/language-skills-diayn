import jax
import jax.numpy as jnp
import chex
from icecream import ic
from sentence_transformers import SentenceTransformer
from functools import partial
from typing import Optional, Tuple, Union, Any

from craftax_classic.constants import *
from gymnax.environments import environment, spaces

from crafter_constants import blocks_labels, mobs_labels, inventory_labels

def separate_features_by_skill(features, skills):
    skill_idxs = jnp.argmax(skills, axis=1)

    i = jnp.argsort(skill_idxs)
    sorted_features = features[i]
    sorted_skill_idxs = skill_idxs[i]

    changes = jnp.where(jnp.diff(sorted_skill_idxs) != 0)[0] + 1
    grouped_features = jnp.split(sorted_features, changes)
    return grouped_features

def obs_to_features(obs):
    maps, metadata = jnp.split(obs, [7 * 9 * 21], axis=1)
    
    maps = jnp.reshape(maps, [-1, 7, 9, 21])
    maps = jnp.transpose(maps, [0, 3, 1, 2])
    maps = jnp.reshape(maps, [-1, 21, 7 * 9])
    maps = maps.sum(axis=2)

    features = jnp.concatenate((maps, metadata), axis=-1)
    return features

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class ParallelizedBatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info

# OBS_SIZE = 1345
# MAP_HEIGHT = 7
# MAP_WIDTH = 9
# MAP_FEATURES = 21

# obs_feature_labels = {
#     -22: 'inv/wood',
#     -21: 'inv/stone',
#     -20: 'inv/coal',
#     -19: 'inv/iron',
#     -18: 'inv/diamond',
#     -17: 'inv/sapling',
#     -16: 'inv/wood_pickaxe',
#     -15: 'inv/stone_pickaxe',
#     -14: 'inv/iron_pickaxe',
#     -13: 'inv/wood_sword',
#     -12: 'inv/stone_sword',
#     -11: 'inv/iron_sword',
#     -10: 'int/player_health',
#     -9: 'int/player_food',
#     -8: 'int/player_drink',
#     -7: 'int/player_energy',
#     -6: 'dir/1',
#     -5: 'dir/2',
#     -4: 'dir/3',
#     -3: 'dir/4',
#     -2: 'light_level',
#     -1: 'is_sleeping',
# }

# map_feature_idxs = {
#     0: 'block/invalid',
#     1: 'block/out_of_bounds',
#     2: 'block/grass',
#     3: 'block/water',
#     4: 'block/stone',
#     5: 'block/tree',
#     6: 'block/wood',
#     7: 'block/path',
#     8: 'block/coal',
#     9: 'block/iron',
#     10: 'block/diamond',
#     11: 'block/crafting_table',
#     12: 'block/furnace',
#     13: 'block/sand',
#     14: 'block/lava',
#     15: 'block/plant',
#     16: 'block/ripe_plant',
#     17: 'mob/zombie',
#     18: 'mob/cow',
#     19: 'mob/skeleton',
#     20: 'mob/arrow'
# }

# # Replace negative indices with positive indices
# for idx in obs_feature_labels:
#     if idx < 0:
#         obs_feature_labels[idx + OBS_SIZE] = obs_feature_labels[idx]

# for h in range(MAP_HEIGHT):
#     for w in range(MAP_WIDTH):
#         for f in range(MAP_FEATURES):
#             map_feature_name = map_feature_idxs[f]
#             idx = (h * MAP_WIDTH * MAP_FEATURES) + (w * MAP_FEATURES) + f
#             obs_feature_labels[idx] = f'{map_feature_name}/h{h}-w{w}'

health_labels = ['very unhealthy', 'unhealthy', 'at okay health', 'healthy', 'very healthy']
food_labels = ['very hungry', 'hungry', 'at okay hunger', 'full', 'very full']
thirsty_labels = ['very thirsty', 'thirsty', 'at okay thirst', 'thirsty', 'very thirsty']
energy_labels = ['very tired', 'tired', 'at okay energy', 'well-rested', 'very well-rested']

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embedding_crafter(next_obs):
    batch_size = next_obs.shape[0]
    maps, metadata = jnp.split(next_obs, [7 * 9 * 21], axis=1)
    
    # For each block/mob type, count how many are in frame
    maps = jnp.reshape(maps, [-1, 7, 9, 21])
    maps = jnp.transpose(maps, [0, 3, 1, 2])
    maps = jnp.reshape(maps, [-1, 21, 7 * 9])
    maps = maps.sum(axis=2)
    maps = jnp.round(maps).astype(jnp.int32)
    blocks, mobs = jnp.split(maps, [17], axis=1)

    # Extract and format metadata
    inventory, intrinsics, direction, light_level, is_sleeping = jnp.split(metadata, [12, 16, 20, 21], axis=1)
    inventory = jnp.round(inventory * 10.0).astype(jnp.int32)
    intrinsics *= 10.0

    health, food, drink, energy = jnp.split(intrinsics, [1, 2, 3], axis=1)

    # Construct embedding
    sentences = []
    for i in range(batch_size):
        blocks_gt0 = jnp.argwhere(blocks[i] > 0).reshape(-1)
        mobs_gt0 = jnp.argwhere(mobs[i] > 0).reshape(-1)
        inventory_gt0 = jnp.argwhere(inventory[i] > 0).reshape(-1)

        blocks_str = 'You see the following blocks: {}.'.format(', '.join([blocks_labels[b] for b in blocks_gt0]))
        mobs_str = 'You see the following mobs: {}.'.format(', '.join([mobs_labels[m] for m in mobs_gt0]))
        inventory_str = 'You have in your inventory: {}.'.format(', '.join([inventory_labels[i] for i in inventory_gt0]))

        health_str = 'Your health level is: {}.'.format(health_labels[round(health[i].item() / 2.0) - 1])
        food_str = 'Your hunger level is: {}.'.format(food_labels[round(food[i].item() / 2.0) - 1])
        drink_str = 'Your thirst level is: {}.'.format(thirsty_labels[round(drink[i].item() / 2.0) - 1])
        energy_str = 'Your energy level is: {}.'.format(energy_labels[round(energy[i].item() / 2.0) - 1])

        desc = [health_str, food_str, drink_str, energy_str]
        if len(blocks_gt0) > 0:
            desc.append(blocks_str)
        if len(mobs_gt0) > 0:
            desc.append(mobs_str)
        if len(inventory_gt0) > 0:
            desc.append(inventory_str)

        desc = ' '.join(desc)
        sentences.append(desc)

    sentences = embedding_model.encode(sentences)
    sentences = jnp.array(sentences)
    return sentences
