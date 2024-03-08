import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import numpy as np
from tqdm import tqdm

class QNet(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.action_size)(x)
        return x
    
class QSkillNet(nn.Module):
    action_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, state, skill):
        x = jnp.concatenate((state, skill), axis=-1)
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.action_size)(x)
        return x

class Discriminator(nn.Module):
    skill_size: int
    hidden1_size: int
    hidden2_size: int
    dropout_rate: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden1_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.hidden2_size)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x
