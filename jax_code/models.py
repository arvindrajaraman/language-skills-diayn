import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np
from typing import List

# from model_utils import ResidualLayer

class QNet(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, skill, train):
        x = jnp.concatenate((state, skill), axis=-1)
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        # x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        # x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.action_size)(x)
        return x
    
# class QNetConv(nn.Module):
#     out_channels: int
#     action_size: int
#     hidden1_size: int
#     hidden2_size: int
#     dropout_rate: float

#     res_layers: List[ResidualLayer]

#     def setup(self):
#         res1 = ResidualLayer(out_channels=self.out_channels,     stride=1, num_blocks=3)
#         res2 = ResidualLayer(out_channels=self.out_channels * 2, stride=2, num_blocks=4)
#         res3 = ResidualLayer(out_channels=self.out_channels * 4, stride=2, num_blocks=6)
#         res4 = ResidualLayer(out_channels=self.out_channels * 8, stride=2, num_blocks=3)
#         self.res_layers = [res1, res2, res3, res4]

#     def __call__(self, state, skill, train):
#         x = state
#         for res_layer in self.res_layers:
#             x = res_layer(x)

#         x = nn.avg_pool(x)
#         x = jnp.concatenate((x, skill), axis=1)
#         x = nn.Dense(features=self.hidden1_size)(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
#         x = nn.Dense(features=self.hidden2_size)(x)
#         x = nn.relu(x)
#         x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
#         x = nn.Dense(features=self.action_size)(x)

#         return x

class Discriminator(nn.Module):
    skill_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x, train):
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x
