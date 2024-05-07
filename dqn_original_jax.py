import jax
import jax.numpy as jnp
from jax import jit, random, vmap, tree_util
import flax
import flax.linen as nn
import optax
from collections import deque
import gym
from functools import partial
import numpy as np

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network

N_EPISODES = 2000
MAX_T = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

SEED = 42
ACTION_SIZE = 4
STATE_SIZE = 8
FC1_UNITS = 64
FC2_UNITS = 64

class QNetwork(nn.Module):
    action_size: int
    fc1_units: int
    fc2_units: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.fc1_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.fc2_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_size)(x)
        return x
    
def get_size(data) -> int:
    sizes = tree_util.tree_map(lambda arr: len(arr), data)
    return max(tree_util.tree_leaves(sizes))
    
class Dataset(flax.core.FrozenDict):
    """
    A class for storing (and retrieving batches of) data in nested dictionary format.

    Example:
        dataset = Dataset({
            'observations': {
                'image': np.random.randn(100, 28, 28, 1),
                'state': np.random.randn(100, 4),
            },
            'actions': np.random.randn(100, 2),
        })

        batch = dataset.sample(32)
        # Batch will have nested shape: {
        # 'observations': {'image': (32, 28, 28, 1), 'state': (32, 4)},
        # 'actions': (32, 2)
        # }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

    def sample(self, batch_size: int, indx=None):
        """
        Sample a batch of data from the dataset. Use `indx` to specify a specific
        set of indices to retrieve. Otherwise, a random sample will be drawn.

        Returns a dictionary with the same structure as the original dataset.
        """
        if indx is None:
            indx = np.random.randint(self.size, size=batch_size)
        return self.get_subset(indx)

    def get_subset(self, indx):
        return tree_util.tree_map(lambda arr: arr[indx], self._dict)

class ReplayBuffer(Dataset):
    """
    Dataset where data is added to the buffer.

    Example:
        example_transition = {
            'observations': {
                'image': np.random.randn(28, 28, 1),
                'state': np.random.randn(4),
            },
            'actions': np.random.randn(2),
        }
        buffer = ReplayBuffer.create(example_transition, size=1000)
        buffer.add_transition(example_transition)
        batch = buffer.sample(32)

    """

    @classmethod
    def create(cls, transition, size: int):
        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        buffer_dict = tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: dict, size: int):
        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_transition(self, transition):
        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element

        tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

def train():
    key = random.PRNGKey(SEED)
    key, qlocal_key, qtarget_key = random.split(key, num=3)
    # Init models
    qlocal = QNetwork(
        action_size=ACTION_SIZE,
        fc1_units=FC1_UNITS,
        fc2_units=FC2_UNITS
    )
    qlocal_params = qlocal.init(qlocal_key, jnp.zeros((STATE_SIZE,)))
    qlocal_opt = optax.adam(learning_rate=LR)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)
    
    qtarget = QNetwork(
        action_size=ACTION_SIZE,
        fc1_units=FC1_UNITS,
        fc2_units=FC2_UNITS
    )
    qtarget_params = qtarget.init(qtarget_key, jnp.zeros((STATE_SIZE,)))

    buffer = ReplayBuffer.create({
        'obs': np.zeros((STATE_SIZE,)),
        'action': np.zeros((1,), dtype=np.int32),
        'reward': np.zeros((1,)),
        'next_obs': np.zeros((STATE_SIZE,)),
        'done': np.zeros((1,)),
    }, size=BUFFER_SIZE)

    @jit
    def act(key, qlocal_params, obs, eps=0.0):    
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs)),
            random.choice(action_key, jnp.arange(ACTION_SIZE))
        )
        return key, action
    
    @jit
    def update(qlocal_params, qlocal_opt_state, qtarget_params, batch):
        def loss_fn(qlocal_params, batch):
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['action'], batch['reward'], batch['next_obs'], batch['done']

            Q_targets_next = qtarget.apply(qtarget_params, next_obs).max(axis=1).reshape(-1, 1)
            Q_targets = rewards + (GAMMA * Q_targets_next * (1.0 - dones))

            Q_expected = qlocal.apply(qlocal_params, obs)[jnp.arange(BATCH_SIZE), actions.reshape(-1)].reshape(-1, 1)

            loss = optax.squared_error(Q_targets, Q_expected).mean()
            q_value_mean = jnp.mean(Q_expected)
            target_value_mean = jnp.mean(Q_targets)
            reward_mean = jnp.mean(rewards)
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qlocal_params, batch)
        updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, updates)

        # Polyak average the qtarget_params
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, TAU)

        return qlocal_params, qlocal_opt_state, qtarget_params, aux_metrics
    
    env = gym.make('LunarLander-v2')

    t_step = 0
    eps = EPS_START
    score_window = deque(maxlen=100)
    for episode in range(N_EPISODES):
        obs = env.reset()
        score = 0
        for t in range(MAX_T):
            key, action = act(key, qlocal_params, obs, eps)
            next_obs, reward, done, info = env.step(action.item())
            buffer.add_transition({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done
            })

            t_step += 1
            if t_step % UPDATE_EVERY == 0:
                batch = buffer.sample(BATCH_SIZE)
                qlocal_params, qlocal_opt_state, qtarget_params, (q_value_mean, target_value_mean, reward_mean) = update(qlocal_params, qlocal_opt_state, qtarget_params, batch)
            obs = next_obs
            score += reward
            if done:
                break
        score_window.append(score)
        eps = max(eps * EPS_DECAY, EPS_END)
        print('\rEpisode {}\tTimestep {}\tAverage Score: {:.2f}\tQ-value mean: {:.2f}\tTarget Q-value mean: {:.2f}\tReward mean: {:.2f}\tEpsilon: {:.2f}'.format(episode, t_step, jnp.mean(jnp.array(score_window)), q_value_mean, target_value_mean, reward_mean, eps), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tTimestep {}\tAverage Score: {:.2f}\tQ-value mean: {:.2f}\tTarget Q-value mean: {:.2f}\tReward mean: {:.2f}\tEpsilon: {:.2f}'.format(episode, t_step, jnp.mean(jnp.array(score_window)), q_value_mean, target_value_mean, reward_mean, eps))

if __name__ == '__main__':
    train()
