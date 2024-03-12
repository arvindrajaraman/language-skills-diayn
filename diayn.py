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
from lunarlander.metrics import train_mutual_info_score
from models import QNet, QNetConv, Discriminator, DiscriminatorConv
from structures import ReplayBuffer
from utils import grad_norm, to_numpy, to_torch
from visualization.visitations import plot_visitations

torch.set_float32_matmul_precision('high')

class DIAYN():
    def __init__(
        self,
        config: ConfigDict,
        log: bool = True
    ):
        """Initialize the DIAYN agent, including initialization of the policy and discriminator, replay buffer, and other internal structures.
        
        Args:
            config (ConfigDict): Configuration for the DIAYN agent.
            log (bool): Whether to log to wandb.
        """
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
        self.discrim = discrim_type(config.state_size, config.skill_size, config.discrim_units, config.discrim_units).to('cuda')
        self.discrim.train()
        self.discrim_loss_fn = nn.CrossEntropyLoss()
        self.discrim_opt = optim.Adam(self.discrim.parameters(), lr=config.discrim_lr)

        # Replay buffer
        self.memory = ReplayBuffer(config)

        # Visitation distribution
        self.max_x = self.max_y = 1.5
        self.min_x = self.min_y = -1.5
        self.num_bins_x = self.num_bins_y = 50
        self.bin_size_x = (self.max_x - self.min_x) / self.num_bins_x
        self.bin_size_y = (self.max_y - self.min_y) / self.num_bins_y
        self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))

        # Goal tracking
        self.goal_deque = deque(maxlen=config.log_freq)
    
    def init_qlocal_from_path(
        self,
        path: str
    ):
        """Load the policy network from a path.
        
        Args:
            path (str): Path to the policy network state dict.
        """
        self.qlocal.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qlocal.train()

    def init_qtarget_from_path(
        self,
        path: str
    ):
        """Load the target policy network from a path.
        
        Args:
            path (str): Path to the target policy network state dict.
        """
        self.qtarget.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qtarget.train()

    def init_discrim_from_path(
        self,
        path: str
    ):
        """Load the discriminator from a path.
        
        Args:
            path (str): Path to the discriminator state dict.
        """
        self.discrim.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.discrim.train()
        
    def train(
        self,
        env: gym.Env,
        run_name: str
    ):
        """Train the DIAYN agent.
        
        Args:
            env (gym.Env): Environment to train the agent on.
            run_name (str): Name of the run.
        """
        eps = self.config.eps_start
        obs = env.reset()
        skill = np.eye(self.config.skill_size)
        for t in tqdm(range(self.config.episodes)):
            eps = max(eps * self.config.eps_decay, self.config.eps_end)
            action = self.act(obs, skill, eps)
            next_obs, reward, done, _ = env.step(action)
            self.step(obs, action, skill, next_obs, done)
            obs = next_obs
            self.stats.update({
                "reward_ground_truth": reward.mean(),
                "eps": eps,
            })
            done_idxs = np.argwhere(done)
            for done_idx in done_idxs:
                self.goal_deque.append((next_obs[done_idx][0], skill[done_idx][0]))
            if t % self.config.log_freq == 0:
                for i in range(self.config.skill_size):
                    self.stats[f"log_visits_rolling_s{i}"] = plot_visitations(self.visitations[i], self.min_x, self.max_x, self.min_y, self.max_y, i, log=True)
                self.stats.update({
                    "log_visits_rolling_all": plot_visitations(self.visitations[-1], self.min_x, self.max_x, self.min_y, self.max_y, log=True),
                    "mutual_info_rolling": train_mutual_info_score(self.config, self.goal_deque)
                })
                self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))
                if self.config.save_checkpoints:
                    torch.save(self.qlocal.state_dict(), f'./data/{run_name}/qlocal_iter{t}.pth')
                    torch.save(self.qtarget.state_dict(), f'./data/{run_name}/qtarget_iter{t}.pth')
                    torch.save(self.discrim.state_dict(), f'./data/{run_name}/discrim_iter{t}.pth')
            if t % 50 == 0 and self.log:
                wandb.log(self.stats, step=t)
                self.stats.clear()

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        skill: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor
    ):
        """Perform a step in the environment. Record visitations and make model updates if necessary.
        
        Args:
            state (torch.Tensor): Current state.
            action (torch.Tensor): Action taken.
            skill (torch.Tensor): Skill vector.
            next_state (torch.Tensor): Next state.
            done (torch.Tensor): Done flag.
        """
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
    ):
        """Sample an action from the policy network using epsilon-greedy exploration.
        
        Args:
            state (np.ndarray): Current state.
            skill (np.ndarray): Skill vector.
            eps (float): Exploration rate.
        """
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

    def discriminate(
        self,
        state: np.ndarray
    ):
        """Compute the skill probabilities for a specific state, using the discriminator.
        
        Args:
            state (np.ndarray): State to discriminate.
        """
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
        """Update the policy network with a batch of experiences.
        
        Args:
            experiences (Tuple): Batch of experiences.
        """
        @torch.compile
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
    
    @torch.compile
    def update_qtarget(self):
        """Perform Polyak averaging (soft update) on the target policy network with the policy network."""
        for target_param, local_param in zip(self.qtarget.parameters(), self.qlocal.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
    
    def update_discrim(
        self,
        experiences: Tuple
    ):
        """Update the discriminator with a batch of experiences.
        
        Args:
            experiences (Tuple): Batch of experiences.
        """
        self.discrim_opt.zero_grad()

        @torch.compile
        def _update_discrim():
            _, _, skills, _, next_state_embeddings, _ = experiences
            skill_preds = self.discrim.forward(next_state_embeddings)

            loss = self.discrim_loss_fn(skills, skill_preds)
            loss.backward()
            self.discrim_opt.step()

            skill_idxs = torch.argmax(skills, dim=1, keepdim=True)
            skill_pred_idxs = torch.argmax(skill_preds, dim=1, keepdim=True)
            acc = torch.sum(skill_pred_idxs == skill_idxs) / skill_idxs.shape[0]
            ent = -torch.sum(skill_preds * torch.log(skill_preds + 1e-10) / math.log(self.config.skill_size), dim=1).mean()
            return loss, acc, ent
        loss, acc, ent = _update_discrim()

        self.stats.update({
            "discrim_loss": loss.item(),
            "discrim_acc": acc.item(),
            "discrim_ent": ent.item(),
            "discrim_grad_norm": grad_norm(self.discrim),
        })
