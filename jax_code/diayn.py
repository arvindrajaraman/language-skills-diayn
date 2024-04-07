from jax import random, jit
import jax
import jax.numpy as jnp
import jax.lax as lax
import flashbax as fbx
import optax
from tqdm import tqdm
from craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from environment_base.wrappers import AutoResetEnvWrapper, BatchEnvWrapper
from ml_collections import FrozenConfigDict
import os
from pathlib import Path
import yaml
from datetime import datetime
import argparse
from icecream import ic
from functools import partial
import wandb
from dotenv import load_dotenv

from models import QNet, Discriminator
from utils import grad_norm

import gym
import numpy as onp

@partial(jit, static_argnames=('qlocal', 'action_size', 'num_envs'))
def act(key, qlocal, qlocal_params, state, skill, action_size, num_envs, eps=0.0):
    def _random_action(action_key):
        action = random.choice(action_key, jnp.arange(action_size))
        return action

    def _greedy_action():
        qvalues = qlocal.apply(qlocal_params, state, skill, train=False)
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

def train(key, config, log):
    # 1. Initialize training state
    key, qlocal_key, qtarget_key, discrim_key, env_key, skill_init_key = jax.random.split(key, num=6)

    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    qlocal = QNet(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        dropout_rate=config.dropout_rate
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill, train=False)
    qlocal_opt = optax.adam(
        learning_rate=config.policy_lr
    )
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNet(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        dropout_rate=config.dropout_rate
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill, train=False)

    discrim = Discriminator(
        skill_size=config.skill_size,
        hidden1_size=config.discrim_units,
        hidden2_size=config.discrim_units,
        dropout_rate=config.dropout_rate
    )
    discrim_params = discrim.init(discrim_key, dummy_embedding, train=False)
    discrim_opt = optax.adam(
        learning_rate=config.discrim_lr
    )
    discrim_opt_state = discrim_opt.init(discrim_params)

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
        'skill': jnp.zeros((config.skill_size,)),
        'next_obs': jnp.zeros((config.state_size,)),
        'next_obs_embedding': jnp.zeros((config.embedding_size,)),
        'done': jnp.array(False, dtype=jnp.bool),
    })

    # 2. Define model update functions
    @jit
    def update_qlocal(dropout_key, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, obs, actions, skills, next_obs,
                      next_obs_embeddings, dones):
        def loss_fn(qlocal_params, obs, actions, skills, next_obs, next_obs_embeddings, dones):
            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings, train=False)
            
            selected_probs = skill_preds[jnp.arange(config.batch_size), skill_idxs]
            rewards = jnp.log(selected_probs + 1e-9) - jnp.log(1 / config.skill_size)

            Q_target_outputs = qtarget.apply(qtarget_params, next_obs, skills, train=False)
            Q_targets_next = jnp.max(Q_target_outputs, axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)

            Q_expected = qlocal.apply(qlocal_params, obs, skills, train=True, rngs={
                'dropout': dropout_key
            })[jnp.arange(config.batch_size), actions]

            loss = optax.squared_error(Q_targets, Q_expected).mean()
            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, allow_int=True, has_aux=True)(
            qlocal_params, obs, actions, skills, next_obs, next_obs_embeddings, dones)
        updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, updates)

        q_value_mean, target_value_mean, reward_mean = aux_metrics
        qlocal_metrics = {
            'critic_loss': loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(grads),
            'reward_skill': reward_mean
        }

        return qlocal_params, qlocal_opt_state, qlocal_metrics

    @jit
    def update_qtarget(qlocal_params, qtarget_params):
        new_qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)
        return new_qtarget_params

    @jit
    def update_discrim(dropout_key, discrim_params, discrim_opt_state, embeddings, skills):
        def loss_fn(discrim_params, embeddings, skills):
            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, embeddings, train=True, rngs={
                'dropout': dropout_key
            })
            skill_pred_idxs = jnp.argmax(skill_preds, axis=1)
            
            skill_counts = jnp.sum(skills, axis=0)
            skill_freqs = skill_counts / skill_counts.sum()
            # class_weights = jnp.where(jnp.equal(skill_counts, 0), 0, jnp.reciprocal(skill_counts))
            # weights = class_weights[skill_idxs]

            loss = optax.softmax_cross_entropy(skills, skill_preds).mean()
            # loss = jnp.multiply(optax.softmax_cross_entropy(skills, skill_preds), weights).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            ent = jax.scipy.special.entr(skill_preds).sum(axis=1).mean()
            discrim_input_skills_ent = jax.scipy.special.entr(skill_freqs).sum().mean()
            return loss, (acc, ent, discrim_input_skills_ent)
        
        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            discrim_params, embeddings, skills)
        updates, discrim_opt_state = discrim_opt.update(grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, updates)

        acc, ent, discrim_input_skills_ent = aux_metrics
        discrim_metrics = {
            'discrim_loss': loss,
            'discrim_acc': acc,
            'discrim_ent': ent,
            'discrim_input_skills_ent': discrim_input_skills_ent,
            'discrim_grad_norm': grad_norm(grads)
        }

        return discrim_params, discrim_opt_state, discrim_metrics

    # 2. Run training
    eps = config.eps_start
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(config.num_envs,))

    if config.gym_np:
        env = gym.vector.make(config.env_name, num_envs=config.num_envs, asynchronous=True)
        obs = env.reset()
        obs = jnp.array(obs)
    else:
        env = BatchEnvWrapper(AutoResetEnvWrapper(CraftaxClassicSymbolicEnv()), num_envs=config.num_envs)
        env_params = env.default_params
        obs, state = env.reset(env_key, env_params)

    for t_step in tqdm(range(config.episodes // config.num_envs)):
        key, step_key, buffer_key, skill_update_key, qlocal_dropout_key, discrim_dropout_key = random.split(key, num=6)

        skill = jax.nn.one_hot(skill_idx, config.skill_size)
        eps = jnp.maximum(eps * config.eps_decay, config.eps_end)

        key, action = act(key, qlocal, qlocal_params, obs, skill, config.action_size, config.num_envs, eps)
        if config.gym_np:
            action = onp.asarray(action)
            next_obs, _, done, _ = env.step(action)
            next_obs, done = jnp.array(next_obs), jnp.array(done)
        else:
            next_obs, next_state, _, done, _ = env.step(step_key, state, action, env_params)
        next_obs_embedding = next_obs  # change to use embedding fn later

        buffer_state = buffer.add(buffer_state, {
            'obs': obs,
            'action': action,
            'skill': skill,
            'next_obs': next_obs,
            'next_obs_embedding': next_obs_embedding,
            'done': done,
        })

        experiences = buffer.sample(buffer_state, buffer_key)
        e_obs = experiences.experience['obs']
        e_actions = experiences.experience['action']
        e_skills = experiences.experience['skill']
        e_next_obs = experiences.experience['next_obs']
        e_next_obs_embeddings = experiences.experience['next_obs_embedding']
        e_dones = experiences.experience['done']

        metrics = {
            't_step': t_step,
            'episodes': t_step * config.num_envs,
            'eps': eps,
        }

        if t_step % config.qlocal_update_freq == 0:
            qlocal_params, qlocal_opt_state, qlocal_metrics = update_qlocal(
                qlocal_dropout_key, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, e_obs, e_actions, e_skills, e_next_obs,
                e_next_obs_embeddings, e_dones
            )
            metrics.update(qlocal_metrics)
        
        if t_step % config.qtarget_update_freq == 0:
            qtarget_params = update_qtarget(qlocal_params, qtarget_params)
        
        if t_step % config.discrim_update_freq == 0:
            discrim_params, discrim_opt_state, discrim_metrics = update_discrim(
                discrim_dropout_key, discrim_params, discrim_opt_state, e_next_obs_embeddings, e_skills
            )
            metrics.update(discrim_metrics)

        if log:
            wandb.log(metrics, step=t_step)
    
        obs = next_obs
        if not config.gym_np:
            state = next_state
        skill_idx = jnp.where(
            done,
            random.randint(skill_update_key, minval=0, maxval=config.skill_size, shape=(config.num_envs,)),
            skill_idx
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--nolog', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(Path(os.path.join('../config', args.config)).read_text())
    if "seed" not in config:
        config['seed'] = int(datetime.now().timestamp())
    config = FrozenConfigDict(config)

    if not args.nolog:
        run_name = '{}_{}_{}_{}_{}'.format('CraftaxClassicSymbolic', config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
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
    train(key, config, log=not args.nolog)
    run.finish()
