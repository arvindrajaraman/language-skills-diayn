# python
import argparse
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

# packages
from craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from dotenv import load_dotenv
from environment_base.wrappers import AutoResetEnvWrapper, BatchEnvWrapper
import flashbax as fbx
from flax import linen as nn
from flax.training import orbax_utils
from functools import partial
import gym
from icecream import ic
from jax import random, jit
import jax
import jax.numpy as jnp
import jax.lax as lax
from ml_collections import ConfigDict
import numpy as onp
import optax
import orbax.checkpoint
from tqdm import tqdm
import wandb

# files
from models import QNetClassic
from utils import grad_norm

@partial(jit, static_argnames=('qlocal', 'action_size', 'num_envs'))
def act(key, qlocal, qlocal_params, state, action_size, num_envs, eps=0.0):
    def _random_action(action_key):
        action = random.choice(action_key, jnp.arange(action_size))
        return action

    def _greedy_action():
        qvalues = qlocal.apply(qlocal_params, state, train=False)
        action = jnp.argmax(qvalues, axis=-1)
        return action
    
    key, explore_key, action_key = random.split(key, num=3)
    explore = eps > random.uniform(explore_key, shape=(num_envs,))
    action = jnp.where(
        explore,
        _random_action(action_key),
        _greedy_action()
    )

    return key, action

def train(key, config, run_name, log):
    # 1. Initialize training state
    key, qlocal_key, qtarget_key, env_key = jax.random.split(key, num=4)

    dummy_state = jnp.zeros((1, config.state_size))

    qlocal = QNetClassic(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        dropout_rate=config.dropout_rate
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state, train=False)
    qlocal_opt = optax.adam(
        learning_rate=config.policy_lr
    )
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetClassic(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        dropout_rate=config.dropout_rate
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state, train=False)

    buffer = fbx.make_item_buffer(
        config.buffer_size,
        config.batch_size,
        config.batch_size,
        add_sequences=False,
        add_batches=True
    )
    buffer_state = buffer.init({
        'obs': jnp.zeros((config.state_size,)),
        'action': jnp.array(0, dtype=jnp.int32),
        'reward': jnp.array(0, dtype=jnp.float32),
        'next_obs': jnp.zeros((config.state_size,)),
        'done': jnp.array(False, dtype=jnp.bool),
    })
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    # 2. Define model update functions
    @jit
    def update_qlocal(dropout_key, qlocal_params, qlocal_opt_state, qtarget_params, obs, actions, rewards, next_obs, dones):
        def loss_fn(qlocal_params, obs, actions, rewards, next_obs, dones):
            Q_target_outputs = qtarget.apply(qtarget_params, next_obs, train=False)
            Q_targets_next = jnp.max(Q_target_outputs, axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)

            Q_expected = qlocal.apply(qlocal_params, obs, train=True, rngs={
                'dropout': dropout_key
            })[jnp.arange(config.batch_size), actions]

            reg_penalty = 1e-1
            reg_loss = sum(jnp.abs(w).sum() for w in jax.tree_util.tree_leaves(qlocal_params))

            loss = optax.squared_error(Q_targets, Q_expected).mean() + (reg_penalty * reg_loss)
            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            qlocal_params, obs, actions, rewards, next_obs, dones)
        updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, updates)

        q_value_mean, target_value_mean, reward_mean = aux_metrics
        qlocal_metrics = {
            'critic_loss': loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(grads),
            'reward': reward_mean
        }

        return qlocal_params, qlocal_opt_state, qlocal_metrics

    @jit
    def update_qtarget(qlocal_params, qtarget_params):
        new_qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)
        return new_qtarget_params

    # Divide frequencies by num_envs
    config.qlocal_update_freq = math.ceil(config.qlocal_update_freq / config.num_envs)
    config.qtarget_update_freq = math.ceil(config.qtarget_update_freq / config.num_envs)
    config.vis_freq = math.ceil(config.vis_freq / config.num_envs)
    config.save_freq = math.ceil(config.save_freq / config.num_envs)
    config.warmup_steps = math.ceil(config.warmup_steps / config.num_envs)

    # 2. Run training
    episodes = 0
    eps = config.eps_start

    if config.gym_np:
        env = gym.vector.make(config.env_name, num_envs=config.num_envs, asynchronous=True)
        obs = env.reset()
        obs = jnp.array(obs)
        state, next_state = None, None
        score = jnp.zeros(config.num_envs)
    else:
        env = BatchEnvWrapper(AutoResetEnvWrapper(CraftaxClassicSymbolicEnv()), num_envs=config.num_envs)
        env_params = env.default_params
        obs, state = env.reset(env_key, env_params)

    for t_step in tqdm(range(config.steps // config.num_envs)):
        key, step_key, buffer_key, qlocal_dropout_key = random.split(key, num=4)
        if t_step > config.warmup_steps:
            eps = jnp.maximum(eps * config.eps_decay, config.eps_end)

        key, action = act(key, qlocal, qlocal_params, obs, config.action_size, config.num_envs, eps)
        if config.gym_np:
            action = onp.asarray(action)
            next_obs, reward, done, _ = env.step(action)
            next_obs, done, reward = jnp.array(next_obs), jnp.array(done), jnp.array(reward, dtype=jnp.float32)
        else:
            next_obs, next_state, reward, done, _ = env.step(step_key, state, action, env_params)
        
        metrics = {
            't_step': t_step,
            'steps': t_step * config.num_envs,
            'eps': eps,
            'episodes': episodes,
        }

        score += reward
        done_idx = jnp.argwhere(done).reshape(-1)
        if done_idx.shape[0] > 0:
            metrics['score'] = score[done_idx].mean()
        score = score.at[done_idx].set(0.0)
        episodes += jnp.sum(done)

        buffer_state = buffer.add(buffer_state, {
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
        })

        experiences = buffer.sample(buffer_state, buffer_key)
        e_obs = experiences.experience['obs']
        e_actions = experiences.experience['action']
        e_rewards = experiences.experience['reward']
        e_next_obs = experiences.experience['next_obs']
        e_dones = experiences.experience['done']

        if t_step > config.warmup_steps:   
            if t_step % config.qlocal_update_freq == 0:
                qlocal_params, qlocal_opt_state, qlocal_metrics = update_qlocal(
                    qlocal_dropout_key, qlocal_params, qlocal_opt_state, qtarget_params, e_obs, e_actions, e_rewards, e_next_obs, e_dones
                )
                metrics.update(qlocal_metrics)
            
            if t_step % config.qtarget_update_freq == 0:
                qtarget_params = update_qtarget(qlocal_params, qtarget_params)
        
        if log:
            wandb.log(metrics, step=t_step*config.num_envs)
            if t_step % config.save_freq == 0:
                checkpoint = {'qlocal': qlocal_params, 'qtarget': qtarget_params}
                save_args = orbax_utils.save_args_from_target(checkpoint)
                orbax_checkpointer.save(f'{os.getcwd()}/data/{run_name}/checkpoint_{t_step*config.num_envs}', checkpoint, save_args=save_args)

        obs = next_obs
        if not config.gym_np:
            state = next_state



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--nolog', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(Path(os.path.join('./config', args.config)).read_text())
    if "seed" not in config:
        config['seed'] = int(datetime.now().timestamp())
    config = ConfigDict(config)

    if not args.nolog:
        run_name = '{}_dqn_{}'.format(config.env_name, int(datetime.now().timestamp()))
        os.makedirs(f'./data/{run_name}')

        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        run = wandb.init(
            project="language-skills",
            entity="arvind6902",
            name=run_name,
            config=config
        )
        wandb.config = config
    else:
        run_name = None

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
