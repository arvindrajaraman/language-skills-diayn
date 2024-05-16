#region Imports
import argparse
from collections import deque
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

from craftax.craftax_env import make_craftax_env_from_name
from dotenv import load_dotenv
import flashbax as fbx
from flax import linen as nn
from functools import partial
import gym
from icecream import ic
from jax import random, jit, vmap, pmap
import jax
import jax.numpy as jnp
import jax.lax as lax
from ml_collections import ConfigDict
import numpy as onp
import optax
from tqdm import tqdm
import wandb

import diayn_utils
from models import DiscriminatorCraftax, Discriminator
from structures import ReplayBuffer
import crafter_constants
import crafter_utils
import crafter_vis
import lunar_utils
import lunar_vis
#endregion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(key, config, run_name, log):
    assert config.env_name == 'Craftax-Classic-Symbolic-v1', "Cannot train using this file if the env is not Craftax"

    #region 1. Initialize training state
    key, policy_key, discrim_key, env_key, skill_init_key = jax.random.split(key, num=5)

    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    policy, policy_params, policy_opt, policy_opt_state = diayn_utils.dqn_init_policy(config, policy_key, dummy_state, dummy_embedding, dummy_skill)

    DiscriminatorModel = DiscriminatorCraftax if config.embedding_type == 'identity' else Discriminator
    discrim = DiscriminatorModel(
        skill_size=config.skill_size,
        hidden1_size=config.discrim_units,
        hidden2_size=config.discrim_units
    )
    discrim_params = discrim.init(discrim_key, dummy_embedding)
    discrim_opt = optax.adam(learning_rate=config.discrim_lr)
    discrim_opt_state = discrim_opt.init(discrim_params)

    buffer = ReplayBuffer.create({
        'obs': onp.zeros((config.state_size,)),
        'action': onp.array(0, dtype=onp.int32),
        'skill': onp.zeros((config.skill_size,)),
        'reward_gt': onp.array(0.0),
        'next_obs': onp.zeros((config.state_size,)),
        'next_obs_embedding': onp.zeros((config.embedding_size,)),
        'done': onp.array(False, dtype=onp.bool_),
    }, size=config.buffer_size)
    # buffer = fbx.make_item_buffer(
    #     config.buffer_size,
    #     config.batch_size,
    #     config.batch_size,
    #     add_sequences=False,
    #     add_batches=True
    # )
    # buffer_state = buffer.init({
    #     'obs': jnp.zeros((config.state_size,)),
    #     'action': jnp.array(0, dtype=jnp.int32),
    #     'skill': jnp.zeros((config.skill_size,)),
    #     'reward_gt': jnp.array(0.0),
    #     'next_obs': jnp.zeros((config.state_size,)),
    #     'next_obs_embedding': jnp.zeros((config.embedding_size,)),
    #     'done': jnp.array(False, dtype=jnp.bool_),
    # })

    if config.embedding_type == 'identity':
        embedding_fn = lambda next_obs: (next_obs, None)
    elif config.embedding_type == '1h':
        embedding_fn = lambda next_obs: lunar_utils.embedding_1he(next_obs[:, 0])
    elif config.embedding_type == 'crafter':
        embedding_model = crafter_utils.new_embedding_model()
        embedding_fn = lambda next_obs: crafter_utils.embedding_crafter(embedding_model, next_obs)
    else:
        raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

    action_skill_mtx = onp.zeros((config.skill_size, config.action_size))
    achievement_counts = onp.zeros((config.skill_size, 22), dtype=onp.int32)  # refreshes every config.vis_step steps
    achievement_counts_total = onp.zeros((22,), dtype=onp.int32)  # does not refresh

    # Divide frequencies/steps by vectorization
    config.steps = math.ceil(config.steps / config.vectorization)
    config.warmup_steps = math.ceil(config.warmup_steps / config.vectorization)
    config.save_freq = math.ceil(config.save_freq / config.vectorization)
    config.vis_freq = math.ceil(config.vis_freq / config.vectorization)
    config.eps_decay **= config.vectorization
    model_updates_per_iter = math.ceil(config.vectorization / config.model_update_freq)
    print('model_updates_per_iter', model_updates_per_iter)
    #endregion

    # 2. Run training
    env = make_craftax_env_from_name(config.env_name, auto_reset=True)
    env_params = env.default_params
    env_reset = jit(vmap(env.reset, in_axes=(0, None)))
    env_step = jit(vmap(env.step, in_axes=(0, 0, 0, None)))
    buffer_add = jit(buffer.add)
    obs, state = env_reset(random.split(env_key, config.vectorization), env_params)

    steps_per_ep = jnp.zeros((config.vectorization,), dtype=jnp.int32)
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(config.vectorization,))
    eps = config.eps_start
    episodes = 0

    metrics = dict()
    for t_step in tqdm(range(config.steps), unit_scale=config.vectorization):
        skill = jax.nn.one_hot(skill_idx, config.skill_size)
        key, action = diayn_utils.dqn_act(key, policy, policy_params, obs, skill, config.action_size, config.vectorization, eps)
        key, step_key = random.split(key)
        next_obs, next_state, reward_gt, done, _ = env_step(random.split(step_key, config.vectorization), state, action, env_params)
        next_obs_embedding, _ = embedding_fn(next_obs)
        done_idx = jnp.argwhere(
            jnp.logical_or(done, steps_per_ep >= config.max_steps_per_ep)
        )

        steps_per_ep += 1
        episodes += done_idx.sum()
        metrics['steps_per_ep'] = steps_per_ep[done_idx].mean().item()
        metrics['steps'] = t_step * config.vectorization
        metrics['eps'] = eps
        metrics['episodes'] = episodes

        # buffer_state = buffer_add(buffer_state, {
        #     'obs': obs,
        #     'action': action,
        #     'skill': skill,
        #     'reward_gt': reward_gt,
        #     'next_obs': next_obs,
        #     'next_obs_embedding': next_obs_embedding,
        #     'done': done,
        # })
        buffer.add_transition_batch({
            'obs': obs,
            'action': action,
            'skill': skill,
            'reward_gt': reward_gt,
            'next_obs': next_obs,
            'next_obs_embedding': next_obs_embedding,
            'done': done
        }, batch_size=config.vectorization)

        if t_step > config.warmup_steps:
            for _ in range(model_updates_per_iter):
                batch = buffer.sample(config.batch_size)
                # key, buffer_key = random.split(key)
                # batch = buffer.sample(buffer_state, buffer_key).experience
                policy_params, policy_opt_state, discrim_params, discrim_opt_state, model_metrics = diayn_utils.dqn_update_model(policy, policy_params, policy_opt, policy_opt_state, discrim, discrim_params, discrim_opt, discrim_opt_state, config.batch_size, config.skill_size, config.gamma, config.tau, config.reward_pr_coeff, config.reward_gt_coeff, batch)
            metrics.update(model_metrics)

            action_skill_mtx[skill_idx, action] += 1
            # Achievement counts
            for idx in done_idx:
                achievement_counts[skill_idx] += state.achievements[idx]
                achievement_counts_total += state.achievements[idx]
    
            for idx, label in enumerate(crafter_constants.achievement_labels):
                metrics[f'achievements/{label}'] = achievement_counts_total[idx] / episodes
            metrics['score/crafter'] = crafter_utils.crafter_score(achievement_counts_total, episodes)
            metrics[f'score/crafter_s{skill_idx}'] = crafter_utils.crafter_score(achievement_counts, episodes / config.skill_size)
            
            if (t_step % config.vis_freq == 10 or t_step == config.steps - 1):
                metrics['achievement_counts'] = crafter_vis.achievement_counts_heatmap(achievement_counts, config.skill_size)
                metrics['achievement_counts_log'] = crafter_vis.achievement_counts_heatmap(
                    jnp.where(achievement_counts > 0, jnp.log(achievement_counts), 0), config.skill_size)
                metrics['action_confusion'] = crafter_vis.action_skill_heatmap(action_skill_mtx)
                
                # if config.save_rollouts:
                #     print('Saving rollouts...')
                #     rollouts = crafter_vis.gather_rollouts_by_skill(key, qlocal, qlocal_params, discrim, discrim_params, embedding_fn, config.skill_size, config.rollouts_per_skill, config.max_steps_per_ep)
                #     for idx in range(config.skill_size):
                #         for rollout_num in range(config.rollouts_per_skill):
                #             rollout_filepath = rollouts[idx][rollout_num]
                #             metrics[f'rollouts/skill_{idx}_num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=8, format="mp4")

                # entire_rb = buffer.sample_all()
                # all_skills, all_next_obs, all_next_obs_embeddings = entire_rb['skill'], entire_rb['next_obs'], entire_rb['next_obs_embedding']
                # all_features = crafter_utils.obs_to_features(all_next_obs)
                # metrics['feature_divergences'] = crafter_vis.feature_divergences(all_features, all_skills)

                # tsne_features = crafter_utils.tsne_embeddings(all_next_obs_embeddings)
                # metrics['embeddings/by_skill'] = crafter_vis.plot_embeddings_by_skill(tsne_features, all_skills, config.skill_size)
                # metrics['embeddings/by_timestep'] = crafter_vis.plot_embeddings_by_timestep(tsne_features, all_timesteps)

                achievement_counts = onp.zeros_like(achievement_counts)
                action_skill_mtx = onp.zeros_like(action_skill_mtx)

        if log:
            wandb.log(metrics, step=t_step*config.vectorization)
            metrics.clear()
        
        if log and config.save_checkpoints and (t_step % config.save_freq == 0 or t_step == config.steps - 1):
            jnp.savez(os.path.join(os.getcwd(), 'data', run_name),
                t_step=t_step,
                policy_params=policy_params,
                discrim_params=discrim_params)

        obs, state = next_obs, next_state
        if t_step > config.warmup_steps:
            eps = max(eps * config.eps_decay, config.eps_end)
        steps_per_ep = steps_per_ep.at[done_idx].set(0)
        key, skill_update_key = random.split(key)
        skill_idx = jnp.where(
            done,
            random.randint(skill_update_key, minval=0, maxval=config.skill_size, shape=(config.vectorization,)),
            skill_idx
        )


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
        run_name = '{}_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
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

    devices = jax.devices()
    print(f'{len(devices)} devices visible through JAX: {devices}')
    config.num_workers = len(devices)

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
