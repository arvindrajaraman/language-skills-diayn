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
from models import QNet, Discriminator
from utils import grad_norm
import crafter_utils
import lunar_utils
import lunar_vis

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

@partial(jit, static_argnames=('discrim'))
def discriminate(discrim, discrim_params, state_embedding):
    skill_logits = discrim.apply(discrim_params, state_embedding, train=False)
    skill_probs = nn.activation.softmax(skill_logits)
    return skill_probs

def train(key, config, run_name, log):
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

    if config.embedding_type == 'identity':
        embedding_fn = lambda next_obs, next_state: next_obs
    elif config.embedding_type == '1h':
        embedding_fn = lambda next_obs, next_state: lunar_utils.embedding_1he(next_obs[:, 0])
    elif config.embedding_type == 'crafter':
        embedding_fn = lambda next_obs, next_state: crafter_utils.embedding_crafter(next_obs)
    else:
        raise ValueError(f"Invalid embedding type: {config['embedding_type']}")
    
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
        goal_skill_mtx = jnp.zeros((3, 3))

        min_x = -1.
        max_x = 1.
        min_y = 0.
        max_y = 1.
        num_bins_x = num_bins_y = 50
        bin_size_x = (max_x - min_x) / num_bins_x
        bin_size_y = (max_y - min_y) / num_bins_y
        visits = jnp.zeros((config.skill_size + 1, num_bins_x, num_bins_y))

    # 2. Define model update functions
    @jit
    def update_qlocal(dropout_key, qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, obs, actions, skills, next_obs,
                      next_obs_embeddings, dones):
        def loss_fn(qlocal_params, obs, actions, skills, next_obs, next_obs_embeddings, dones):
            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings, train=False)
            
            selected_logits = skill_preds[jnp.arange(config.batch_size), skill_idxs]
            rewards = jax.nn.log_softmax(selected_logits + 1e-9) - jnp.log(1 / config.skill_size)

            # rewards -= reward_mean  # normalize rewards

            Q_target_outputs = qtarget.apply(qtarget_params, next_obs, skills, train=False)
            Q_targets_next = jnp.max(Q_target_outputs, axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)

            Q_expected = qlocal.apply(qlocal_params, obs, skills, train=True, rngs={
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

            loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()
            discrim_skills_pplx = 2. ** jax.scipy.special.entr(skill_freqs).mean()

            return loss, (acc, pplx, discrim_skills_pplx)
        
        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            discrim_params, embeddings, skills)
        updates, discrim_opt_state = discrim_opt.update(grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, updates)

        acc, pplx, discrim_skills_pplx = aux_metrics
        discrim_metrics = {
            'discrim_loss': loss,
            'discrim_acc': acc,
            'discrim_pplx': pplx,
            'discrim_skills_pplx': discrim_skills_pplx,
            'discrim_grad_norm': grad_norm(grads)
        }

        return discrim_params, discrim_opt_state, discrim_metrics
    
    # Divide frequencies by num_envs
    config.qlocal_update_freq = math.ceil(config.qlocal_update_freq / config.num_envs)
    config.qtarget_update_freq = math.ceil(config.qtarget_update_freq / config.num_envs)
    config.discrim_update_freq = math.ceil(config.discrim_update_freq / config.num_envs)
    config.vis_freq = math.ceil(config.vis_freq / config.num_envs)
    config.save_freq = math.ceil(config.save_freq / config.num_envs)
    config.warmup_steps = math.ceil(config.warmup_steps / config.num_envs)

    # 2. Run training
    episodes = 0
    eps = config.eps_start
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(config.num_envs,))
    # reward_mean = 0.0

    if config.gym_np:
        env = gym.vector.make(config.env_name, num_envs=config.num_envs, asynchronous=True)
        obs = env.reset()
        obs = jnp.array(obs)
        state, next_state = None, None
    else:
        env = BatchEnvWrapper(AutoResetEnvWrapper(CraftaxClassicSymbolicEnv()), num_envs=config.num_envs)
        env_params = env.default_params
        obs, state = env.reset(env_key, env_params)

    for t_step in tqdm(range(config.steps // config.num_envs)):
        key, step_key, buffer_key, skill_update_key, qlocal_dropout_key, discrim_dropout_key = random.split(key, num=6)
        skill = jax.nn.one_hot(skill_idx, config.skill_size)
        if t_step > config.warmup_steps:
            eps = jnp.maximum(eps * config.eps_decay, config.eps_end)

        key, action = act(key, qlocal, qlocal_params, obs, skill, config.action_size, config.num_envs, eps)
        if config.gym_np:
            action = onp.asarray(action)
            next_obs, _, done, _ = env.step(action)
            next_obs, done = jnp.array(next_obs), jnp.array(done)
        else:
            next_obs, next_state, _, done, _ = env.step(step_key, state, action, env_params)
        next_obs_embedding = embedding_fn(next_obs, next_state)

        episodes += jnp.sum(done)

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
            'steps': t_step * config.num_envs,
            'eps': eps,
            'episodes': episodes,
            # 'reward_mean': reward_mean
        }

        # if t_step % 20000 == 19999:
        #     _, _, rm_skills, _, rm_next_obs_embeddings, _ = buffer.sample(buffer_state, buffer_key, batch_size=1024)
        #     rm_skill_idxs = jnp.argmax(rm_skills, axis=1)
        #     skill_preds = discrim.apply(discrim_params, rm_next_obs_embeddings, train=False)
            
        #     selected_logits = skill_preds[jnp.arange(config.batch_size), rm_skill_idxs]
        #     rewards = jax.nn.log_softmax(selected_logits + 1e-9) - jnp.log(1 / config.skill_size)
        #     reward_mean = rewards.mean()
        #     ic('Standardized rewards: mean', reward_mean)

        if t_step > config.warmup_steps:   
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

            if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
                # Visits
                x_index = ((obs[:, 0] - min_x) / bin_size_x).astype(int)
                x_index = jnp.minimum(jnp.maximum(x_index, jnp.zeros_like(x_index)), jnp.full_like(x_index, num_bins_x - 1))
                y_index = ((obs[:, 1] - min_y) / bin_size_y).astype(int)
                y_index = jnp.minimum(jnp.maximum(y_index, jnp.zeros_like(y_index)), jnp.full_like(x_index, num_bins_y - 1))

                visits = visits.at[skill_idx, y_index, x_index].add(1)
                visits = visits.at[-1, y_index, x_index].add(1)

                # Goal classification
                done_idx = jnp.argwhere(done).reshape(-1)
                goal_idx = lunar_utils.classify_goal(obs[done_idx, 0])
                goal_skill_mtx = goal_skill_mtx.at[skill_idx[done_idx], goal_idx].add(1)

                if t_step % config.vis_freq == 0:
                    for i in range(config.skill_size):
                        metrics[f"log_visits_s{i}"] = lunar_vis.plot_visits(visits[i], min_x, max_x, min_y, max_y, i, log=True)
                    
                    metrics.update({
                        "log_visits_all": lunar_vis.plot_visits(visits[-1], min_x, max_x, min_y, max_y, log=True),
                        "goal_confusion": lunar_vis.goal_skill_heatmap(goal_skill_mtx),
                        "I(z;s)": lunar_utils.I_zs_score(goal_skill_mtx),
                        "I(z;a)": lunar_utils.I_za_score(qlocal, qlocal_params, e_obs, e_skills, config.batch_size, config.skill_size, config.action_size),
                        "pred_landscape": lunar_vis.plot_pred_landscape(discrim, discrim_params, embedding_fn),
                    })

                    goal_skill_mtx = jnp.zeros_like(goal_skill_mtx)
                    visits = jnp.zeros_like(visits)

        if log:
            wandb.log(metrics, step=t_step*config.num_envs)
            if t_step % config.save_freq == 0:
                checkpoint = {'qlocal': qlocal_params, 'qtarget': qtarget_params, 'discrim': discrim_params}
                save_args = orbax_utils.save_args_from_target(checkpoint)
                orbax_checkpointer.save(f'{os.getcwd()}/data/{run_name}/checkpoint_{t_step*config.num_envs}', checkpoint, save_args=save_args)

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

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
