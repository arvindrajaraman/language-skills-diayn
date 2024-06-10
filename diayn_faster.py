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
import agents
from models import QNetCraftax, DiscriminatorCraftax, Discriminator, APTForwardDynamicsModel, APTBackwardDynamicsModel, Captioner
from structures import ReplayBuffer
from utils import grad_norm
import crafter_constants
import crafter_utils
import crafter_vis
import lunar_utils
import lunar_vis
#endregion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(key, config, run_name, log):
    assert config.env_name == 'Craftax-Classic-Symbolic-v1', "Cannot train using this file if the env is not Craftax"

    #region Initialize training state
    key, qlocal_key, qtarget_key, discrim_key, env_key, skill_init_key = jax.random.split(key, num=6)

    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    qlocal = QNetCraftax(
        action_size=config.action_size,
        hidden_size=config.policy_units
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill)
    qlocal_opt = optax.adam(learning_rate=config.policy_lr)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetCraftax(
        action_size=config.action_size,
        hidden_size=config.policy_units
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill)

    DiscriminatorModel = DiscriminatorCraftax if config.embedding_type == 'identity' else Discriminator
    discrim = DiscriminatorModel(
        skill_size=config.skill_size,
        hidden_size=config.discrim_units
    )
    discrim_params = discrim.init(discrim_key, dummy_embedding)
    discrim_opt = optax.adam(learning_rate=config.discrim_lr)
    discrim_opt_state = discrim_opt.init(discrim_params)
        
    if config.skill_learning_method == 'rnd':
        key, random_fn_key = jax.random.split(key)
        random_fn = DiscriminatorModel(
            skill_size=config.skill_size,
            hidden_size=config.discrim_units
        )
        random_fn_params = random_fn.init(random_fn_key, dummy_embedding)
    elif config.skill_learning_method == 'apt':
        key, forward_model_key, backward_model_key = jax.random.split(key, num=3)
        forward_model = APTForwardDynamicsModel(
            action_size=config.action_size,
            skill_size=config.state_size,
            hidden_size=config.discrim_units
        )
        forward_model_params = forward_model.init(forward_model_key, dummy_skill, jnp.zeros((1,)))
        forward_model_opt = optax.adam(learning_rate=config.discrim_lr)
        forward_model_opt_state = forward_model_opt.init(forward_model_params)
        backward_model = APTBackwardDynamicsModel(
            action_size=config.action_size,
            hidden_size=config.discrim_units
        )
        backward_model_params = backward_model.init(backward_model_key, dummy_skill, dummy_skill)
        backward_model_opt = optax.adam(learning_rate=config.discrim_lr)
        backward_model_opt_state = backward_model_opt.init(backward_model_params)

    buffer = fbx.make_item_buffer(max_length=config.buffer_size, min_length=1, sample_batch_size=config.batch_size, add_batches=True, add_sequences=False)
    buffer_state = buffer.init({
        'obs': jnp.zeros((config.state_size,)),
        'obs_embedding': jnp.zeros((config.embedding_size,)),
        'action': jnp.array(0, dtype=jnp.int32),
        'skill': jnp.zeros((config.skill_size,)),
        'reward_gt': jnp.array(0.0),
        'next_obs': jnp.zeros((config.state_size,)),
        'next_obs_embedding': jnp.zeros((config.embedding_size,)),
        'done': jnp.array(False, dtype=jnp.bool_),
    })

    if config.embedding_type == 'identity':
        embedding_fn = lambda next_obs: (next_obs, None)
    elif config.embedding_type == '1h':
        embedding_fn = lambda next_obs: lunar_utils.embedding_1he(next_obs[:, 0])
    elif config.embedding_type == 'crafter' or config.embedding_type == 'crafter_and_identity':
        key, captioner_key = jax.random.split(key)
        captioner = Captioner(hidden_size=config.embedding_size)
        captioner_params = jnp.load('crafter_captioner.npy', allow_pickle=True).item()
        if config.embedding_type == 'crafter':
            embedding_fn = lambda next_obs:(captioner.apply(captioner_params, next_obs), None)
        else:
            embedding_fn = lambda next_obs: (jnp.concatenate((captioner.apply(captioner_params, next_obs), next_obs), axis=-1), None)
    else:
        raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

    achievement_counts = onp.zeros((config.skill_size, 22), dtype=onp.int32)  # refreshes every config.vis_step steps
    achievement_counts_total = onp.zeros((22,), dtype=onp.int32)  # does not refresh

    # Divide frequencies/steps by vectorization
    config.steps = math.ceil(config.steps / config.vectorization)
    config.warmup_steps = math.ceil(config.warmup_steps / config.vectorization)
    config.save_freq = math.ceil(config.save_freq / config.vectorization)
    config.vis_freq = math.ceil(config.vis_freq / config.vectorization)
    config.eps_decay = config.eps_end ** (1 / (config.eps_end_perc * config.steps))
    print('Eps decay:', config.eps_decay)
    model_updates_per_iter = math.ceil(config.vectorization / config.model_update_freq)
    print('model_updates_per_iter', model_updates_per_iter)
    #endregion

    #region Model usage functions
    @jit
    def act(key, qlocal_params, obs, skill, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key, shape=(config.vectorization,)) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs, skill), axis=1),
            random.choice(action_key, jnp.arange(config.action_size), shape=(config.vectorization,))
        )
        return key, action

    def diayn_reward_pr_function(discrim_params, batch):
        skills, next_obs_embeddings = batch['skill'], batch['next_obs_embedding']

        skill_idxs = jnp.argmax(skills, axis=1)
        skill_preds = jax.nn.log_softmax(discrim.apply(discrim_params, next_obs_embeddings), axis=1)
        
        selected_logits = skill_preds[jnp.arange(config.batch_size), skill_idxs]
        reward_prs = selected_logits + jnp.log(config.skill_size)
        return reward_prs
    
    def rnd_reward_pr_function(discrim_params, batch):
        next_obs_embeddings = batch['next_obs_embedding']
        
        rnd_rep_target = discrim.apply(discrim_params, next_obs_embeddings)
        rnd_rep_pred = jax.lax.stop_gradient(random_fn.apply(random_fn_params, next_obs_embeddings))
        
        return jnp.sqrt(jnp.square(rnd_rep_target - rnd_rep_pred).sum(axis=1))
    
    def apt_reward_pr_function(discrim_params, batch):
        next_obs_embeddings = batch['next_obs_embedding']
        
        reps = discrim.apply(discrim_params, next_obs_embeddings)
        sum_squares = jnp.square(reps).sum(axis=1)
        dists = jnp.outer(sum_squares, sum_squares) - 2 * jnp.dot(reps, reps.T)
        dists = jnp.sqrt(jnp.maximum(dists, 0.0))
        
        # Within each column of this matrix, find the k smallest values
        dists = jnp.sort(dists, axis=1)
        knns = dists[:, :config.apt_k]
        knns_avg_dist = knns.mean(axis=1)
        
        return jnp.log(knns_avg_dist + 1.0)  # 1.0 for numerical stability

    def update_policy(reward_pr_function, qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, reward_pr_coeff, reward_gt_coeff, batch):
        def loss_fn(qlocal_params, reward_pr_coeff, reward_gt_coeff, batch):
            obs, actions, skills, reward_gts, next_obs, dones = batch['obs'], batch['action'], batch['skill'], batch['reward_gt'], batch['next_obs'], batch['done']

            reward_prs = reward_pr_function(discrim_params, batch)
            rewards = (reward_pr_coeff * reward_prs) + (reward_gt_coeff * reward_gts)

            Q_targets_next = qtarget.apply(qtarget_params, next_obs, skills).max(axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)
            Q_expected = qlocal.apply(qlocal_params, obs, skills)[jnp.arange(config.batch_size), actions]

            loss = optax.squared_error(Q_targets, Q_expected).mean()

            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, (q_value_mean, target_value_mean, reward_mean)), grads = jax.value_and_grad(loss_fn, has_aux=True)(qlocal_params, reward_pr_coeff, reward_gt_coeff, batch)
        qlocal_updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, qlocal_updates)
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)

        policy_metrics = {
            'critic_loss': loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(grads),
            'reward': reward_mean
        }
        return qlocal_params, qlocal_opt_state, qtarget_params, policy_metrics
    
    def update_rnd_network(discrim_params, discrim_opt_state, batch):
        def loss_fn(discrim_params, batch):
            next_obs_embeddings = batch['next_obs_embedding']
            
            rnd_rep_pred = discrim.apply(discrim_params, next_obs_embeddings)
            rnd_rep_target = jax.lax.stop_gradient(random_fn.apply(random_fn_params, next_obs_embeddings))
            
            loss = optax.l2_loss(rnd_rep_pred, rnd_rep_target).mean()
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(discrim_params, batch)
        updates, discrim_opt_state = discrim_opt.update(grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, updates)
        
        skill_learning_metrics = {
            'rnd_loss': loss,
            'rnd_grad_norm': grad_norm(grads)
        }
        return discrim_params, discrim_opt_state, skill_learning_metrics
    
    def update_apt_network(discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, batch):
        def loss_fn(discrim_params, forward_model_params, backward_model_params, batch):
            obs_embeddings, actions, next_obs_embeddings = batch['obs_embedding'], batch['action'], batch['next_obs_embedding']
            
            obs_apt_encoding = discrim.apply(discrim_params, obs_embeddings)
            next_obs_apt_encoding = discrim.apply(discrim_params, next_obs_embeddings)
            
            next_obs_hat = forward_model.apply(forward_model_params, obs_apt_encoding, actions)
            action_hat = backward_model.apply(backward_model_params, obs_apt_encoding, next_obs_apt_encoding)
            
            forward_error = optax.l2_loss(next_obs_hat, next_obs_embeddings).mean()
            backward_error = optax.softmax_cross_entropy_with_integer_labels(action_hat, actions).mean()
            loss = forward_error + backward_error
            return loss, (forward_error, backward_error)
        
        (loss, (forward_error, backward_error)), (discrim_grads, forward_model_grads, backward_model_grads) = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(discrim_params, forward_model_params, backward_model_params, batch)
        
        discrim_updates, discrim_opt_state = discrim_opt.update(discrim_grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, discrim_updates)
        
        forward_model_updates, forward_model_opt_state = forward_model_opt.update(forward_model_grads, forward_model_opt_state)
        forward_model_params = optax.apply_updates(forward_model_params, forward_model_updates)
        
        backward_model_updates, backward_model_opt_state = backward_model_opt.update(backward_model_grads, backward_model_opt_state)
        backward_model_params = optax.apply_updates(backward_model_params, backward_model_updates)
        
        skill_learning_metrics = {
            'apt_loss': loss,
            'forward_error': forward_error,
            'backward_error': backward_error
        }
        return discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, skill_learning_metrics
    
    def update_diayn_discrim(discrim_params, discrim_opt_state, batch):
        def loss_fn(discrim_params, batch):
            next_obs_embeddings, skills = batch['next_obs_embedding'], batch['skill']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings)
            skill_pred_idxs = jnp.argmax(skill_preds, axis=1)

            loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()

            return loss, (acc, pplx)
        
        (loss, (discrim_acc, discrim_pplx)), grads = jax.value_and_grad(loss_fn, has_aux=True)(discrim_params, batch)
        discrim_updates, discrim_opt_state = discrim_opt.update(grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, discrim_updates)

        skill_learning_metrics = {
            'diayn_discrim_loss': loss,
            'diayn_discrim_acc': discrim_acc,
            'diayn_discrim_pplx': discrim_pplx,
            'diayn_discrim_grad_norm': grad_norm(grads)
        }

        return discrim_params, discrim_opt_state, skill_learning_metrics

    @jit
    def full_diayn_update(qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, discrim_opt_state, reward_pr_coeff, reward_gt_coeff, batch):
        qlocal_params, qlocal_opt_state, qtarget_params, policy_metrics = update_policy(diayn_reward_pr_function, qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, reward_pr_coeff, reward_gt_coeff, batch)
        discrim_params, discrim_opt_state, skill_learning_metrics = update_diayn_discrim(discrim_params, discrim_opt_state, batch)
        return qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, policy_metrics, skill_learning_metrics

    @jit
    def full_rnd_update(qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, discrim_opt_state, reward_pr_coeff, reward_gt_coeff, batch):
        qlocal_params, qlocal_opt_state, qtarget_params, policy_metrics = update_policy(rnd_reward_pr_function, qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, reward_pr_coeff, reward_gt_coeff, batch)
        discrim_params, discrim_opt_state, skill_learning_metrics = update_rnd_network(discrim_params, discrim_opt_state, batch)
        return qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, policy_metrics, skill_learning_metrics
    
    @jit
    def full_apt_update(qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, reward_pr_coeff, reward_gt_coeff, batch):
        qlocal_params, qlocal_opt_state, qtarget_params, policy_metrics = update_policy(apt_reward_pr_function, qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, reward_pr_coeff, reward_gt_coeff, batch)
        discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, skill_learning_metrics = update_apt_network(discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, batch)
        return qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, forward_model_params, forward_model_opt_state, backward_model_params, backward_model_opt_state, policy_metrics, skill_learning_metrics
    
    @jit
    def reward_coeff_schedule_discrete(t_step):
        return jax.lax.cond(
            (t_step - config.warmup_steps) < config.pretraining_steps,
            lambda: (1.0, 0.0),
            lambda: (0.0, 1.0)
        )
    
    @jit
    def reward_coeff_schedule_anneal(t_step):
        reward_pr_coeff = 1.0 - ((t_step - config.warmup_steps) / (config.steps - config.warmup_steps))
        reward_gt_coeff = 5.0 * ((t_step - config.warmup_steps) / (config.steps - config.warmup_steps))
        return reward_pr_coeff, reward_gt_coeff
    
    @jit
    def reward_coeff_schedule_constant(t_step):
        return config.reward_pr_coeff, config.reward_gt_coeff
    
    if config.reward_coeff_schedule == 'discrete':
        reward_coeff_schedule = reward_coeff_schedule_discrete
    elif config.reward_coeff_schedule == 'anneal':
        reward_coeff_schedule = reward_coeff_schedule_anneal
    elif config.reward_coeff_schedule == 'constant':
        reward_coeff_schedule = reward_coeff_schedule_constant
    else:
        raise NotImplementedError(f"Invalid reward_coeff_schedule: {config.reward_coeff_schedule}")
    #endregion

    #region Run training
    env = make_craftax_env_from_name(config.env_name, auto_reset=True)
    env_params = env.default_params
    env_reset = jit(vmap(env.reset, in_axes=(0, None)))
    env_step = jit(vmap(env.step, in_axes=(0, 0, 0, None)))
    obs, state = env_reset(random.split(env_key, config.vectorization), env_params)
    obs_embedding, _ = embedding_fn(obs)

    steps_per_ep = jnp.zeros((config.vectorization,), dtype=jnp.int32)
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(config.vectorization,))
    eps = jnp.array(config.eps_start)
    episodes = 0
    achievement_counts = jnp.zeros((config.skill_size, 22), dtype=jnp.int32)
    achievement_counts_total = jnp.zeros((22,), dtype=jnp.int32)
    
    @jit
    def train_step(t_step, val):
        # Env step
        key, obs, state, obs_embedding, steps_per_ep, skill_idx, eps, episodes, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, buffer_state, achievement_counts, achievement_counts_total = val
        skill = jax.nn.one_hot(skill_idx, config.skill_size)
        key, action = act(key, qlocal_params, obs, skill, eps)
        key, step_key = random.split(key)
        next_obs, next_state, reward_gt, done, _ = env_step(random.split(step_key, config.vectorization), state, action, env_params)
        next_obs_embedding, _ = embedding_fn(next_obs)
        done_idx = jnp.logical_or(done, steps_per_ep >= config.max_steps_per_ep)
        
        steps_per_ep += 1
        episodes += jnp.sum(done_idx)
        
        # Buffer update
        buffer_state = buffer.add(buffer_state, {
            'obs': obs,
            'obs_embedding': obs_embedding,
            'action': action,
            'skill': skill,
            'reward_gt': reward_gt,
            'next_obs': next_obs,
            'next_obs_embedding': next_obs_embedding,
            'done': done
        })
        
        # Model update
        reward_pr_coeff, reward_gt_coeff = reward_coeff_schedule(t_step)
        for _ in range(model_updates_per_iter):
            key, buffer_key = jax.random.split(key)
            batch = buffer.sample(buffer_state, buffer_key).experience
            qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, policy_metrics, skill_learning_metrics = full_diayn_update(qlocal_params, qtarget_params, qlocal_opt_state, discrim_params, discrim_opt_state, reward_pr_coeff, reward_gt_coeff, batch)
        
        # Update internal states
        obs, obs_embedding, state = next_obs, next_obs_embedding, next_state
        eps = jnp.maximum(eps * config.eps_decay, config.eps_end)
        steps_per_ep = jnp.where(done_idx, 0, steps_per_ep)
        key, skill_update_key = random.split(key)
        skill_idx = jnp.where(
            done,
            random.randint(skill_update_key, minval=0, maxval=config.skill_size, shape=(config.vectorization,)),
            skill_idx
        )
        
        other_metrics = {
            'steps_per_ep': jnp.where(done_idx, steps_per_ep, 0).mean(),
            'steps': t_step * config.vectorization,
            'eps': eps,
            'episodes': episodes,
            'reward_pr_coeff': reward_pr_coeff,
            'reward_gt_coeff': reward_gt_coeff
        }
        
        return (key, obs, state, obs_embedding, steps_per_ep, skill_idx, eps, episodes, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, buffer_state, achievement_counts, achievement_counts_total), (policy_metrics, skill_learning_metrics, other_metrics)
        
    val = key, obs, state, obs_embedding, steps_per_ep, skill_idx, eps, episodes, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, buffer_state
    for t_step in tqdm(range(config.steps), unit_scale=config.vectorization):
        val, step_metrics = train_step(t_step, val)
        
        metrics = {}
        metrics.update(step_metrics[0])
        metrics.update(step_metrics[1])
        
    #endregion


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
        if 'name' not in config:
            run_name = '{}_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
        else:
            run_name = '{}_{}'.format(config.name, int(datetime.now().timestamp()))

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
