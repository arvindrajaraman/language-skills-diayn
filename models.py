from typing import List

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

# from model_utils import ResidualLayer

class QNetClassic(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.action_size)(x)
        return x

class QNet(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, skill, train):
        x = jnp.concatenate((state, skill), axis=-1)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.action_size)(x)
        return x
    
class QNetCraftax(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, skill, train):
        maps, metadata = jnp.split(state, [7 * 9 * 21], axis=1)
        maps = maps.reshape((-1, 7, 9, 21))
        
        maps = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.Dropout(rate=0.1, deterministic=not train)(maps)
        maps = nn.max_pool(maps, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        maps = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.Dropout(rate=0.1, deterministic=not train)(maps)
        maps = nn.max_pool(maps, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        maps_features = maps.reshape((maps.shape[0], -1))

        y = jnp.concatenate((maps_features, metadata, skill), axis=-1)
        y = nn.LayerNorm()(y)
        y = nn.Dense(self.hidden1_size)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        y = nn.LayerNorm()(y)
        y = nn.Dense(self.hidden2_size)(y)
        y = nn.relu(y)
        y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
        y = nn.Dense(self.action_size)(y)
        y = nn.softmax(y)
        return y

class Discriminator(nn.Module):
    skill_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train):
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x
