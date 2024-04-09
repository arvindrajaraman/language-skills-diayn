from collections import deque
import math
import numpy as np
import os
import random
from typing import Tuple

import gym
from ml_collections import ConfigDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from lunarlander.captioner import naive_captioner
from lunarlander.metrics import I_zs_confusion_matrix, I_zs_score, I_za_score
from models import QNet, QNetConv, Discriminator, DiscriminatorConv
from structures import ReplayBuffer
from utils import grad_norm, to_numpy, to_torch
from visualization.goal_confusion import confusion_matrix_heatmap
from visualization.pred_landscape import plot_pred_landscape
from visualization.visitations import plot_visitations

# torch.set_float32_matmul_precision('high')

class DIAYN():
    def __init__(
        self,
        config: ConfigDict,
        log: bool = True
    ):
        self.config = config
        self.t_step = 0
        self.stats = {}
        self.log = log

        assert config.buffer_size % config.num_envs == 0, "Buffer size must be divisible by number of environments"

        # Set embedding function
        if config.embedding_type == 'identity':
            self.embedding_fn = lambda x: x
        elif config.embedding_type == 'naive':
            self.embedding_fn = naive_captioner
        else:
            raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

        # Set seed
        random.seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Policy: local and target q-networks
        q_type = QNetConv if config.conv else QNet
        self.qlocal = q_type(config.state_size, config.action_size, config.skill_size, config.policy_units, config.policy_units).to('cuda')
        self.qtarget = q_type(config.state_size, config.action_size, config.skill_size, config.policy_units, config.policy_units).to('cuda')
        self.qlocal.train()
        self.qtarget.train()
        self.qlocal_opt = optim.Adam(self.qlocal.parameters(), lr=config.policy_lr)

        # Discriminator
        discrim_type = DiscriminatorConv if config.conv else Discriminator
        self.discrim = discrim_type(config.embedding_size, config.skill_size, config.discrim_units, config.discrim_units).to('cuda')
        self.discrim.train()
        self.discrim_opt = optim.Adam(self.discrim.parameters(), lr=config.discrim_lr)

        # Replay buffer
        self.memory = ReplayBuffer(config)

        # Visitation distribution
        self.min_x = -1.
        self.max_x = 1.
        self.min_y = 0.
        self.max_y = 1.
        self.num_bins_x = self.num_bins_y = 50
        self.bin_size_x = (self.max_x - self.min_x) / self.num_bins_x
        self.bin_size_y = (self.max_y - self.min_y) / self.num_bins_y
        self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))

        # Goal tracking
        self.goal_skill_pairs = list()
    
    def init_qlocal_from_path(
        self,
        path: str
    ):
        self.qlocal.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qlocal.train()

    def init_qtarget_from_path(
        self,
        path: str
    ):
        self.qtarget.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qtarget.train()

    def init_discrim_from_path(
        self,
        path: str
    ):
        self.discrim.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.discrim.train()
        
    def train(
        self,
        env: gym.Env,
        run_name: str
    ):
        eps = self.config.eps_start
        obs = env.reset()
        skill = np.eye(self.config.skill_size)
        for t in tqdm(range(self.config.episodes)):
            eps = max(eps * self.config.eps_decay, self.config.eps_end)
            action = self.act(obs, skill, eps)
            next_obs, reward, done, _ = env.step(action)
            self.step(obs, action, skill, next_obs, done)
            self.stats.update({
                "reward_ground_truth": reward.mean(),
                "eps": eps,
            })
            done_idxs = np.argwhere(done)
            for done_idx in done_idxs:
                self.goal_skill_pairs.append((obs[done_idx][0], skill[done_idx][0]))
            if t % self.config.log_freq == 0:
                for i in range(self.config.skill_size):
                    self.stats[f"log_visits_rolling_s{i}"] = plot_visitations(self.visitations[i], self.min_x, self.max_x, self.min_y, self.max_y, i, log=True)
                conf_mtx = I_zs_confusion_matrix(self.config, self.goal_skill_pairs)
                self.stats.update({
                    "log_visits_rolling_all": plot_visitations(self.visitations[-1], self.min_x, self.max_x, self.min_y, self.max_y, log=True),
                    "pred_landscape": plot_pred_landscape(self.discrim, self.embedding_fn),
                    "goal_confusion_rolling": confusion_matrix_heatmap(conf_mtx),
                    "I(z;s)_rolling": I_zs_score(conf_mtx),
                    "I(z;a)_rolling": I_za_score(self.config, self),
                })
                self.goal_skill_pairs.clear()
                self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))
                if self.config.save_checkpoints and self.log:
                    torch.save(self.qlocal.state_dict(), f'./data/{run_name}/qlocal_iter{t}.pth')
                    torch.save(self.qtarget.state_dict(), f'./data/{run_name}/qtarget_iter{t}.pth')
                    torch.save(self.discrim.state_dict(), f'./data/{run_name}/discrim_iter{t}.pth')
            if t % 50 == 0 and self.log:
                wandb.log(self.stats, step=t)
                self.stats.clear()
            obs = next_obs

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        skill: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ):
        # Save experience in replay memory
        next_state_embedding = self.embedding_fn(next_state)
        self.memory.add(state, action, skill, next_state, next_state_embedding, done)

        # Record visitation
        x_index = ((state[:, 0] - self.min_x) / self.bin_size_x).astype(int)
        x_index = np.minimum(np.maximum(x_index, np.zeros_like(x_index)), np.full_like(x_index, self.num_bins_x - 1))
        y_index = ((state[:, 1] - self.min_y) / self.bin_size_y).astype(int)
        y_index = np.minimum(np.maximum(y_index, np.zeros_like(y_index)), np.full_like(x_index, self.num_bins_y - 1))

        skill_idx = np.argmax(skill, axis=1)
        self.visitations[skill_idx, y_index, x_index] += 1.
        self.visitations[-1, y_index, x_index] += 1.

        # Perform model updates
        if self.config.discrim_reset_freq != 0 and self.t_step % self.config.discrim_reset_freq == 0:
            self.discrim.fc3.reset_parameters()
        if self.t_step % self.config.qlocal_update_freq == 0 and len(self.memory) >= self.config.batch_size:
            experiences = self.memory.sample()
            self.update_qlocal(experiences)
        if self.t_step % self.config.qtarget_update_freq == 0:
            self.update_qtarget()
        if self.t_step % self.config.discrim_update_freq == 0:
            experiences = self.memory.sample()
            self.update_discrim(experiences)
        self.t_step += 1

    def act(
        self,
        state: np.ndarray,
        skill: np.ndarray,
        eps: float
    ) -> np.ndarray:
        state = to_torch(state)
        skill = to_torch(skill)
        self.qlocal.eval()
        with torch.no_grad():
            action_values = to_numpy(self.qlocal(state, skill))
        self.qlocal.train()

        explore = np.random.rand(self.config.num_envs,)
        rand_action = np.random.randint(0, self.config.action_size, (self.config.num_envs,))
        opt_action = np.argmax(action_values, axis=1)

        action = np.where(explore < eps, rand_action, opt_action)
        return action
    
    def act_probs(
        self,
        state: np.ndarray,
        skill: np.ndarray
    ) -> np.ndarray:
        state = to_torch(state)
        skill = to_torch(skill)
        self.qlocal.eval()
        with torch.no_grad():
            action_probs = to_numpy(F.softmax(self.qlocal(state, skill), dim=1))
        self.qlocal.train()
        return action_probs

    def discriminate(
        self,
        state: np.ndarray
    ) -> torch.Tensor:
        state = to_torch(state)
        self.discrim.eval()
        with torch.no_grad():
            predictions = self.discrim(state)[0]
        self.discrim.train()
        return predictions

    def update_qlocal(
        self,
        experiences: Tuple
    ):
        # @torch.compile
        def _update_qlocal():
            states, actions, skills, next_states, next_state_embeddings, dones = experiences
            skill_idxs = torch.argmax(skills, dim=1, keepdim=True)
            
            skill_preds = self.discrim.forward(next_state_embeddings)
            selected_probs = torch.gather(skill_preds, 1, skill_idxs)
            rewards = torch.log(selected_probs + 1e-5) - math.log(1/self.config.skill_size)

            Q_targets_next = self.qtarget(next_states, skills).detach().max(1)[0].unsqueeze(1)
            Q_targets = rewards + (self.config.gamma * Q_targets_next * ~dones)
            Q_expected = self.qlocal(states, skills).gather(1, actions)

            reg_lambda = 1e-1
            l1_reg = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True).to('cuda')
            for W in self.qlocal.parameters():
                l1_reg = l1_reg + W.norm(1)

            loss = F.mse_loss(Q_expected, Q_targets) + reg_lambda * l1_reg
            self.qlocal_opt.zero_grad()
            loss.backward()
            self.qlocal_opt.step()
            return loss, Q_expected, Q_targets, rewards
        loss, Q_expected, Q_targets, rewards = _update_qlocal()

        self.stats.update({
            "critic_loss": loss.item(),
            "q_values": Q_expected.mean().item(),
            "target_values": Q_targets.mean().item(),
            "qlocal_grad_norm": grad_norm(self.qlocal),
            "reward_skill": rewards.mean().item(),
        })
    
    # @torch.compile
    def update_qtarget(self):
        for target_param, local_param in zip(self.qtarget.parameters(), self.qlocal.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
            # new_target_param_data = self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data
            # target_param.data = new_target_param_data.clone().detach()
    
    def update_discrim(
        self,
        experiences: Tuple
    ):
        # @torch.compile
        def _update_discrim():
            _, _, skills, _, next_state_embeddings, _ = experiences
            skill_preds = self.discrim.forward(next_state_embeddings)

            col_counts = next_state_embeddings.sum(axis=0)
            missing_classes = torch.any(col_counts < 1e-9)
            if missing_classes:
                return float('nan'), float('nan'), float('nan')
            weights = 1. / (col_counts + 1e-9)

            raise NotImplementedError("verify that we should cross entropy vs. softmax cross entropy for the line below")
            loss = F.cross_entropy(skill_preds, skills, weight=weights)
            # loss = F.cross_entropy(skill_preds, skills)
            self.discrim_opt.zero_grad()
            loss.backward()
            self.discrim_opt.step()

            skill_idxs = torch.argmax(skills, dim=1, keepdim=True)
            skill_pred_idxs = torch.argmax(skill_preds, dim=1, keepdim=True)
            acc = torch.sum(skill_pred_idxs == skill_idxs) / skill_idxs.shape[0]
            ent = -torch.sum(skill_preds * torch.log(skill_preds + 1e-10) / math.log(self.config.skill_size), dim=1).mean()
            return loss, acc, ent
        loss, acc, ent = _update_discrim()

        self.stats.update({
            "discrim_loss": loss if math.isnan(loss) else loss.item(),
            "discrim_acc": acc if math.isnan(acc) else acc.item(),
            "discrim_ent": ent if math.isnan(ent) else ent.item(),
            "discrim_grad_norm": 0. if math.isnan(loss) else grad_norm(self.discrim),
        })
