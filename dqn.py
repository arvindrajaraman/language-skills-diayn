# python
import argparse
from collections import deque
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

# packages
from craftax.craftax_env import make_craftax_env_from_name
from dotenv import load_dotenv
from flax import linen as nn
from flax.training import checkpoints
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
from tqdm import tqdm
import wandb

# files
from models import QNetClassic, QNetClassicCraftax
from structures import ReplayBuffer
from utils import grad_norm
import lunar_vis
import crafter_constants
import crafter_utils
import crafter_vis

def train(key, config, run_name, log):
    # region 1. Initialize training state
    key, qlocal_key, qtarget_key, env_key = jax.random.split(key, num=4)

    QNetModel = QNetClassicCraftax if config.env_name == 'Craftax-Classic-Symbolic-v1' else QNetClassic
    dummy_state = jnp.zeros((1, config.state_size))

    qlocal = QNetModel(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state)
    qlocal_opt = optax.adam(learning_rate=config.policy_lr)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetModel(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state)

    buffer = ReplayBuffer.create({
        'obs': onp.zeros((config.state_size,)),
        'action': onp.zeros((1,), dtype=onp.int32),
        'reward': onp.zeros((1,)),
        'next_obs': onp.zeros((config.state_size,)),
        'done': onp.zeros((1,)),
    }, size=config.buffer_size)

    if config.env_name == 'LunarLander-v2':
        min_x = -1.
        max_x = 1.
        min_y = 0.
        max_y = 1.
        num_bins_x = num_bins_y = 50
        bin_size_x = (max_x - min_x) / num_bins_x
        bin_size_y = (max_y - min_y) / num_bins_y
        visits = onp.zeros((num_bins_x, num_bins_y))
    elif config.env_name == 'Craftax-Classic-Symbolic-v1':
        achievement_counts_total = onp.zeros((22,), dtype=jnp.int32)
    # endregion

    # 2. Define model usage/update functions
    @jit
    def act(key, qlocal_params, obs, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs)),
            random.choice(action_key, jnp.arange(config.action_size), shape=(1,))
        )
        return key, action

    @jit
    def update_q(qlocal_params, qlocal_opt_state, qtarget_params, batch):
        def loss_fn(qlocal_params, batch):
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['action'], batch['reward'], batch['next_obs'], batch['done']

            Q_targets_next = qtarget.apply(qtarget_params, next_obs).max(axis=1).reshape(-1, 1)
            Q_targets = rewards + (config.gamma * Q_targets_next * (1.0 - dones))
            Q_expected = qlocal.apply(qlocal_params, obs)[jnp.arange(config.batch_size), actions.reshape(-1)].reshape(-1, 1)

            loss = optax.squared_error(Q_targets, Q_expected).mean()

            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qlocal_params, batch)
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

        # Update target network
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)

        return qlocal_params, qlocal_opt_state, qtarget_params, qlocal_metrics

    # 2. Run training
    if config.gym_np:
        env = gym.make(config.env_name)
        obs = jnp.array(env.reset())
        state, next_state = None, None
    else:
        env = make_craftax_env_from_name(config.env_name, auto_reset=False)
        env_params = env.default_params
        obs, state = env.reset(env_key, env_params)

    eps = config.eps_start
    score_gt = 0.0
    score_gt_window = deque(maxlen=100)
    steps_per_ep = 0
    episodes = 0
    metrics = dict()
    for t_step in tqdm(range(config.steps)):
        key, action = act(key, qlocal_params, obs.reshape(1, -1), eps)
        if config.gym_np:
            next_obs, reward_gt, done, _ = env.step(action.item())
        else:
            key, step_key = random.split(key)
            next_obs, next_state, reward_gt, done, _ = env.step(step_key, state, action.item(), env_params)
    
        steps_per_ep += 1
        score_gt += reward_gt

        metrics['steps'] = t_step
        metrics['eps'] = eps
        metrics['episodes'] = episodes

        if done or steps_per_ep >= config.max_steps_per_ep:
            episodes += 1
            metrics['score/gt'] = jnp.mean(jnp.array(score_gt_window))
            metrics['steps_per_ep'] = steps_per_ep  

        buffer.add_transition({
            'obs': obs,
            'action': action,
            'reward': reward_gt,
            'next_obs': next_obs,
            'done': done
        })

        if t_step % config.q_update_freq == 0 and t_step > config.warmup_steps:
            batch = buffer.sample(config.batch_size)
            qlocal_params, qlocal_opt_state, qtarget_params, qlocal_metrics = update_q(qlocal_params, qlocal_opt_state, qtarget_params, batch)
            metrics.update(qlocal_metrics)

        if config.env_name == 'LunarLander-v2':
            x_index = int((obs[0] - min_x) / bin_size_x)
            x_index = min(max(x_index, 0), num_bins_x - 1)
            y_index = int((obs[1] - min_y) / bin_size_y)
            y_index = min(max(y_index, 0), num_bins_y - 1)
            visits[y_index, x_index] += 1

            if t_step % config.vis_freq == 10:
                metrics["log_visits_all"] = lunar_vis.plot_visits(visits, min_x, max_x, min_y, max_y, log=True)
                visits = onp.zeros_like(visits)

            if (t_step % config.vis_freq == 10 or t_step == config.steps - 1) and config.save_rollouts:
                rollouts = lunar_vis.gather_rollouts(qlocal, qlocal_params, config.rollouts_per_skill, config.max_steps_per_ep)
                for rollout_num in range(config.rollouts_per_skill):
                    rollout_filepath = rollouts[rollout_num]
                    metrics[f'rollouts/num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=16, format="mp4")
        elif config.env_name == 'Craftax-Classic-Symbolic-v1':
            # Achievement counts
            if done or steps_per_ep >= config.max_steps_per_ep:
                achievement_counts_total += state.achievements
                for idx, label in enumerate(crafter_constants.achievement_labels):
                    metrics[f'achievements/{label}'] = achievement_counts_total[idx] / episodes
                metrics['score/crafter'] = crafter_utils.crafter_score(achievement_counts_total, episodes)

            if (t_step % config.vis_freq == 10 or t_step == config.steps - 1) and config.save_rollouts:
                rollouts = crafter_vis.gather_rollouts(key, qlocal, qlocal_params, config.rollouts_per_skill, config.max_steps_per_ep)
                for rollout_num in range(config.rollouts_per_skill):
                    rollout_filepath = rollouts[rollout_num]
                    metrics[f'rollouts/num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=8, format="mp4")
        

        if log:
            wandb.log(metrics, step=t_step)

        if log and (t_step % config.save_freq == 0 or t_step == config.steps - 1):
            checkpoints.save_checkpoint(ckpt_dir=os.path.join(os.getcwd(), 'data', run_name),
                step=t_step, overwrite=True, keep=10, target={
                'qlocal_params': qlocal_params,
                'qtarget_params': qtarget_params
            })

        obs, state = next_obs, next_state
        if t_step > config.warmup_steps:
            eps = max(eps * config.eps_decay, config.eps_end)
        if done or steps_per_ep >= config.max_steps_per_ep:
            score_gt_window.append(score_gt)
            score_gt = 0.0
            steps_per_ep = 0
            if config.gym_np:
                obs = jnp.array(env.reset())
            else:
                key, env_reset_key = random.split(key)
                obs, state = env.reset(env_reset_key, env_params)


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

    print('Devices visible through JAX:', jax.devices())

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
