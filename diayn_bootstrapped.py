#region Imports
import argparse
from collections import deque
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

from craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from dotenv import load_dotenv
from environment_base.wrappers import AutoResetEnvWrapper
from flax import linen as nn
# from flax.training import orbax_utils
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
# import orbax.checkpoint
from tqdm import tqdm
import wandb

from models import QNetBootstrapped, Discriminator, DiscriminatorCraftax
from structures import ReplayBuffer
from utils import grad_norm
import crafter_constants
import crafter_utils
import crafter_vis
import lunar_utils
import lunar_vis
#endregion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

@partial(jit, static_argnames=('discrim'))
def discriminate(discrim, discrim_params, state_embedding, start_state_embedding):
    skill_logits = discrim.apply(discrim_params, state_embedding, start_state_embedding)
    return skill_logits

def train(key, config, run_name, log):
    #region 1. Initialize training state
    key, qlocal_key, qtarget_key, discrim_key, env_key, skill_init_key, head_init_key = jax.random.split(key, num=7)

    DiscriminatorModel = DiscriminatorCraftax if (config.env_name == 'Craftax-Classic-Symbolic-v1' and config.embedding_type == 'identity') else Discriminator

    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    qlocal = QNetBootstrapped(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        num_heads=config.num_heads
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill)
    qlocal_opt = optax.adam(learning_rate=config.policy_lr)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetBootstrapped(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units,
        num_heads=config.num_heads
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill)

    discrim = DiscriminatorModel(
        skill_size=config.skill_size,
        hidden1_size=config.discrim_units,
        hidden2_size=config.discrim_units
    )
    discrim_params = discrim.init(discrim_key, dummy_embedding, dummy_embedding)
    discrim_opt = optax.adam(learning_rate=config.discrim_lr)
    discrim_opt_state = discrim_opt.init(discrim_params)

    buffer = ReplayBuffer.create({
        'obs': onp.zeros((config.state_size,)),
        'action': onp.zeros((1,), dtype=onp.int32),
        'skill': onp.zeros((config.skill_size,)),
        'next_obs': onp.zeros((config.state_size,)),
        'next_obs_embedding': onp.zeros((config.embedding_size,)),
        'start_obs_embedding': onp.zeros((config.embedding_size,)),
        'done': onp.zeros((1,)),
        'flags': onp.zeros((config.num_heads,))
    }, size=config.buffer_size)

    if config.embedding_type == 'identity':
        embedding_fn = lambda next_obs: next_obs
    elif config.embedding_type == '1h':
        embedding_fn = lambda next_obs: lunar_utils.embedding_1he(next_obs[:, 0])
    elif config.embedding_type == 'crafter':
        embedding_model = crafter_utils.new_embedding_model()
        embedding_fn = lambda next_obs: crafter_utils.embedding_crafter(embedding_model, next_obs)
    else:
        raise ValueError(f"Invalid embedding type: {config['embedding_type']}")
    
    # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
        goal_skill_mtx = onp.zeros((3, 3))

        min_x = -1.
        max_x = 1.
        min_y = 0.
        max_y = 1.
        num_bins_x = num_bins_y = 50
        bin_size_x = (max_x - min_x) / num_bins_x
        bin_size_y = (max_y - min_y) / num_bins_y
        visits = onp.zeros((config.skill_size + 1, num_bins_x, num_bins_y))
    elif config.env_name == 'Craftax-Classic-Symbolic-v1':
        achievement_counts = onp.zeros((config.skill_size, 22), dtype=onp.int32)  # refreshes every config.vis_step steps
        achievement_counts_total = onp.zeros((22,), dtype=onp.int32)  # does not refresh
        score_crafter_by_skill = []
        for _ in range(config.skill_size):
            score_crafter_by_skill.append(deque(maxlen=100))
    #endregion

    #region 2. Define model update functions
    # @jit
    def act(key, qlocal_params, obs, skill, head, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs, skill)[:, (head*config.action_size):((head+1)*config.action_size)]),
            random.choice(action_key, jnp.arange(config.action_size), shape=(1,))
        )
        return key, action

    @jit
    def update_q(qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, batch):
        def loss_fn(qlocal_params, batch):
            obs, actions, skills, next_obs, next_obs_embeddings, start_obs_embeddings, dones, flags = batch['obs'], batch['action'], batch['skill'], batch['next_obs'], batch['next_obs_embedding'], batch['start_obs_embedding'], batch['done'], batch['flags']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings, start_obs_embeddings)
            
            selected_logits = skill_preds[jnp.arange(config.batch_size), skill_idxs]
            rewards = (jax.nn.log_softmax(selected_logits) - jnp.log(1 / config.skill_size)).reshape(-1, 1)

            rewards = jnp.repeat(rewards, config.num_heads, axis=0)
            dones = jnp.repeat(dones, config.num_heads, axis=0)
            actions = jnp.repeat(actions, config.num_heads, axis=0)
            flags = flags.reshape(-1, 1)

            Q_targets_next = qtarget.apply(qtarget_params, next_obs, skills).reshape(-1, config.num_heads, config.action_size).reshape(-1, config.action_size).max(axis=1).reshape(-1, 1)
            Q_targets = rewards + (config.gamma * Q_targets_next * (1.0 - dones))
            Q_expected = qlocal.apply(qlocal_params, obs, skills).reshape(-1, config.num_heads, config.action_size).reshape(-1, config.action_size)[jnp.arange(config.batch_size * config.num_heads), actions.reshape(-1)].reshape(-1, 1)

            loss = (optax.squared_error(Q_targets, Q_expected) * flags).mean()

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

    @jit
    def update_discrim(discrim_params, discrim_opt_state, batch):
        def loss_fn(discrim_params, batch):
            next_obs_embeddings, start_obs_embeddings, skills = batch['next_obs_embedding'], batch['start_obs_embedding'], batch['skill']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings, start_obs_embeddings)
            skill_pred_idxs = jnp.argmax(skill_preds, axis=1)
            
            skill_counts = jnp.sum(skills, axis=0)
            skill_freqs = skill_counts / skill_counts.sum()

            loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()
            discrim_skills_ent = jax.scipy.special.entr(skill_freqs).mean()
            return loss, (acc, pplx, discrim_skills_ent)
        
        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(discrim_params, batch)
        updates, discrim_opt_state = discrim_opt.update(grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, updates)

        acc, pplx, discrim_skills_ent = aux_metrics
        discrim_metrics = {
            'discrim_loss': loss,
            'discrim_acc': acc,
            'discrim_pplx': pplx,
            'discrim_skills_ent': discrim_skills_ent,
            'discrim_grad_norm': grad_norm(grads)
        }

        return discrim_params, discrim_opt_state, discrim_metrics    
    #endregion

    # 2. Run training
    if config.gym_np:
        env = gym.make(config.env_name)
        obs = jnp.array(env.reset())
        state, next_state = None, None
    else:
        env = CraftaxClassicSymbolicEnv()
        env_params = env.default_params
        obs, state = env.reset(env_key, env_params)
    start_obs_embedding = embedding_fn(obs.copy().reshape(1, -1)).reshape(-1)
    head = random.randint(head_init_key, minval=0, maxval=config.num_heads, shape=(1,)).item()
    eps = config.eps_start
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(1,)).item()
    skill = jax.nn.one_hot(skill_idx, config.skill_size)
    score_gt = 0.0  # ground truth reward
    score_gt_window = deque(maxlen=100)
    score_pr = 0.0  # pseudo-reward from discriminator logits
    score_pr_window = deque(maxlen=100)
    steps_per_ep = 0
    episodes = 0

    score_gt_window_by_skill = []
    score_pr_window_by_skill = []
    for _ in range(config.skill_size):
        score_gt_window_by_skill.append(deque(maxlen=100))
        score_pr_window_by_skill.append(deque(maxlen=100))

    for t_step in tqdm(range(config.steps)):
        key, step_key, flag_key = random.split(key, num=3)
        key, action = act(key, qlocal_params, obs.reshape(1, -1), skill.reshape(1, -1), head, eps)
        if config.gym_np:
            next_obs, reward_gt, done, _ = env.step(action.item())
        else:
            next_obs, next_state, reward_gt, done, _ = env.step(step_key, state, action.item(), env_params)
        next_obs_embedding = embedding_fn(next_obs.reshape(1, -1)).reshape(-1)
        reward_pr = jax.nn.log_softmax(
            discriminate(discrim, discrim_params, next_obs_embedding.reshape(1, -1), start_obs_embedding.reshape(1, -1))
        )[skill_idx] - jnp.log(1 / config.skill_size)
        flags = random.bernoulli(flag_key, p=config.bootstrapped_dropout_prob, shape=(config.num_heads,))

        steps_per_ep += 1
        score_gt += reward_gt
        score_pr += reward_pr

        metrics = {
            'steps': t_step,
            'eps': eps,
            'episodes': episodes
        }
        
        if done:
            score_gt_window.append(score_gt)
            score_gt_window_by_skill[skill_idx].append(score_gt)
            score_pr_window.append(score_pr)
            score_pr_window_by_skill[skill_idx].append(score_pr)
            episodes += 1
            metrics['score/gt'] = jnp.mean(jnp.array(score_gt_window))
            metrics[f'score/gt_s{skill_idx}'] = jnp.mean(jnp.array(score_gt_window_by_skill[skill_idx]))
            score_pr /= steps_per_ep
            metrics['score/pr'] = jnp.mean(jnp.array(score_pr_window))
            metrics[f'score/pr_s{skill_idx}'] = jnp.mean(jnp.array(score_pr_window_by_skill[skill_idx]))
            metrics[f'steps_per_ep'] = steps_per_ep

        buffer.add_transition({
            'obs': obs,
            'action': action,
            'skill': skill,
            'next_obs': next_obs,
            'next_obs_embedding': next_obs_embedding,
            'start_obs_embedding': start_obs_embedding,
            'done': done,
            'flags': flags
        })

        if t_step > config.warmup_steps:
            batch = buffer.sample(config.batch_size)
            if t_step % config.q_update_freq == 0:
                qlocal_params, qlocal_opt_state, qtarget_params, qlocal_metrics = update_q(qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, batch)
                metrics.update(qlocal_metrics)
            if t_step % config.discrim_update_freq == 0:
                discrim_params, discrim_opt_state, discrim_metrics = update_discrim(discrim_params, discrim_opt_state, batch)
                metrics.update(discrim_metrics)

            if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
                # Visits
                x_index = int((obs[0] - min_x) / bin_size_x)
                x_index = min(max(x_index, 0), num_bins_x - 1)
                y_index = int((obs[1] - min_y) / bin_size_y)
                y_index = min(max(y_index, 0), num_bins_y - 1)
                visits[skill_idx, y_index, x_index] += 1
                visits[-1, y_index, x_index] += 1

                # Goal classification
                goal_idx = lunar_utils.classify_goal(jnp.array(obs[0]).reshape(1, 1))
                goal_skill_mtx[skill_idx, goal_idx] += 1

                if t_step % config.vis_freq == 10:
                    for i in range(config.skill_size):
                        metrics[f"log_visits_s{i}"] = lunar_vis.plot_visits(visits[i], min_x, max_x, min_y, max_y, i, log=True)
                    
                    metrics.update({
                        "log_visits_all": lunar_vis.plot_visits(visits[-1], min_x, max_x, min_y, max_y, log=True),
                        "goal_confusion": lunar_vis.goal_skill_heatmap(goal_skill_mtx),
                        "I(z;s)": lunar_utils.I_zs_score(goal_skill_mtx),
                        # "I(z;a)": lunar_utils.I_za_score(qlocal, qlocal_params, batch['obs'], batch['skill'], config.batch_size, config.skill_size, config.action_size),
                        # "pred_landscape": lunar_vis.plot_pred_landscape(discrim, discrim_params, embedding_fn),
                    })

                    goal_skill_mtx = onp.zeros_like(goal_skill_mtx)
                    visits = onp.zeros_like(visits)
            elif config.env_name == 'Craftax-Classic-Symbolic-v1':
                # Achievement counts
                if done:
                    achievement_counts[skill_idx] += onp.clip(state.achievements, a_min=0, a_max=1)
                    achievement_counts_total += onp.clip(state.achievements, a_min=0, a_max=1)
                    for idx, label in enumerate(crafter_constants.achievement_labels):
                        metrics[f'achievements/{label}'] = achievement_counts_total[idx] / episodes
                    metrics['score/crafter'] = crafter_utils.crafter_score(achievement_counts_total, episodes)
                    metrics[f'score/crafter_s{skill_idx}'] = crafter_utils.crafter_score(achievement_counts, episodes / config.skill_size)
                
                if t_step % config.vis_freq == 100:
                    metrics['achievement_counts'] = crafter_vis.achievement_counts_heatmap(achievement_counts, config.skill_size)
                    metrics['achievement_counts_log'] = crafter_vis.achievement_counts_heatmap(
                        jnp.where(achievement_counts > 0, jnp.log(achievement_counts), 0), config.skill_size)
                    
                    if config.save_rollouts:
                        rollouts = crafter_vis.gather_rollouts_by_skill(key, qlocal, qlocal_params, discrim, discrim_params, embedding_fn, config.skill_size, num_rollouts=config.rollouts_per_skill, max_steps_per_rollout=500)
                        for idx in range(config.skill_size):
                            for rollout_num in range(config.rollouts_per_skill):
                                rollout_filepath = rollouts[idx][rollout_num]
                                metrics[f'rollouts/skill_{idx}_num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=8, format="mp4")

                    entire_rb = buffer.sample_all()
                    all_skills, all_next_obs, all_next_obs_embeddings = entire_rb['skill'], entire_rb['next_obs'], entire_rb['next_obs_embedding']
                    all_features = crafter_utils.obs_to_features(all_next_obs)
                    metrics['feature_divergences'] = crafter_vis.feature_divergences(all_features, all_skills)

                    # tsne_features = crafter_utils.tsne_embeddings(all_next_obs_embeddings)
                    # metrics['embeddings/by_skill'] = crafter_vis.plot_embeddings_by_skill(tsne_features, all_skills, config.skill_size)
                    # metrics['embeddings/by_timestep'] = crafter_vis.plot_embeddings_by_timestep(tsne_features, all_timesteps)

                    achievement_counts = onp.zeros_like(achievement_counts)

        if log and t_step % config.log_freq == 0:
            wandb.log(metrics, step=t_step)
        # if log and config.save_checkpoints and t_step % config.save_freq == 0:
        #     checkpoint = {'qlocal': qlocal_params, 'qtarget': qtarget_params, 'discrim': discrim_params}
        #     save_args = orbax_utils.save_args_from_target(checkpoint)
        #     orbax_checkpointer.save(f'{os.getcwd()}/data/{run_name}/checkpoint_{t_step}', checkpoint, save_args=save_args)

        obs, state = next_obs, next_state
        if t_step > config.warmup_steps:
            eps = max(eps * config.eps_decay, config.eps_end)
        if done or steps_per_ep >= config.max_steps_per_ep:
            key, skill_update_key, env_reset_key = random.split(key, num=3)
            skill_idx = random.randint(skill_update_key, minval=0, maxval=config.skill_size, shape=(1,)).item()
            skill = jax.nn.one_hot(skill_idx, config.skill_size)
            score_gt = 0.0
            score_pr = 0.0
            steps_per_ep = 0
            if config.gym_np:
                obs = jnp.array(env.reset())
            else:
                obs, state = env.reset(env_reset_key, env_params)
            start_obs_embedding = embedding_fn(obs.copy().reshape(1, -1)).reshape(-1)


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
        run_name = '{}_bootstrapped_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
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
