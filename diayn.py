from collections import deque
import math
import numpy as np
import os
import random
import time

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
from utils import grad_norm, to_numpy
from visualization import plot_visitations

class DIAYN():
    def __init__(self, config: ConfigDict):
        self.config = config
        self.t_step = 0
        self.stats = {}

        # Set embedding function
        if config.embedding_type == 'identity':
            self.embedding_fn = lambda x: x
        elif config.embedding_type == 'naive':
            self.embedding_fn = naive_captioner
        else:
            raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

        # Set seed
        if "seed" not in config:
            config.seed = int(time.time() * 1000)
        random.seed(config.seed)
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Policy: local and target q-networks
        q_type = QNetConv if config.conv else QNet
        self.qlocal = nn.DataParallel(q_type(config.state_size, config.action_size, config.skill_size, config.policy_units, config.policy_units))
        self.qtarget = nn.DataParallel(q_type(config.state_size, config.action_size, config.skill_size, config.policy_units, config.policy_units))
        self.qlocal.train()
        self.qtarget.train()
        self.qlocal_opt = optim.Adam(self.qlocal.parameters(), lr=config.policy_lr)

        # Discriminator
        discrim_type = DiscriminatorConv if config.conv else Discriminator
        self.discrim = nn.DataParallel(discrim_type(config.state_size, config.skill_size, config.discrim_units, config.discrim_units))
        self.discrim.train()
        self.discrim_loss_fn = nn.CrossEntropyLoss()
        self.discrim_opt = optim.Adam(self.discrim.parameters(), lr=config.discrim_lr)

        # Replay buffer
        self.memory = ReplayBuffer(config.action_size, config.skill_size, config.buffer_size, config.batch_size)

        # Visitation distribution
        self.max_x = self.max_y = 1.5
        self.min_x = self.min_y = -1.5
        self.num_bins_x = self.num_bins_y = 50
        self.bin_size_x = (self.max_x - self.min_x) / self.num_bins_x
        self.bin_size_y = (self.max_y - self.min_y) / self.num_bins_y
        self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))

        # Goal tracking
        self.goal_deque = deque(maxlen=config.log_freq)

    def init_qlocal_from_path(self, path):
        self.qlocal.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qlocal.train()

    def init_qtarget_from_path(self, path):
        self.qtarget.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.qtarget.train()

    def init_discrim_from_path(self, path):
        self.discrim.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
        self.discrim.train()

    def train(self, env, run_name):
        eps = self.config.eps_start
        for episode in tqdm(range(self.config.episodes)):
            eps = max(eps * self.config.eps_decay, self.config.eps_end)
            skill_idx = random.randint(0, self.config.skill_size - 1)
            obs = env.reset()

            for t in tqdm(range(self.config.max_steps_per_episode), leave=False):
                action = self.act(obs, skill_idx, eps)
                next_obs, reward, done, _ = env.step(action)
                self.step(obs, action, skill_idx, next_obs, done)
                obs = next_obs
                if done:
                    self.goal_deque.append((next_obs, skill_idx))
                    self.stats.update({
                        "reward_ground_truth": reward,
                        "eps": eps,
                        "episode": episode,
                        "ep_length": t+1
                    })
                    if episode % self.config.log_freq == 0:
                        torch.save(self.qlocal.state_dict(), f'./data/{run_name}/qlocal_iter{episode}.pth')
                        torch.save(self.qtarget.state_dict(), f'./data/{run_name}/qtarget_iter{episode}.pth')
                        torch.save(self.discrim.state_dict(), f'./data/{run_name}/discrim_iter{episode}.pth')
                        for i in range(self.config.skill_size):
                            self.stats[f"log_visits_rolling_s{i}"] = plot_visitations(self.visitations[i], self.min_x, self.max_x, self.min_y, self.max_y, i, log=True)
                        self.stats.update({
                            "log_visits_rolling_all": plot_visitations(self.visitations[-1], self.min_x, self.max_x, self.min_y, self.max_y, log=True),
                            "mutual_info_rolling": train_mutual_info_score(self.config, self.goal_deque)
                        })
                        self.visitations = np.zeros((self.config.skill_size + 1, self.num_bins_x, self.num_bins_y))
                    wandb.log(self.stats)
                    self.stats.clear()
                    break
            


    def step(self, state, action, skill_idx, next_state, done):
        # Save experience in replay memory
        next_state_embedding = to_numpy(self.embedding_fn(next_state))
        self.memory.add(state, action, skill_idx, next_state, next_state_embedding, done)

        # Record visitation
        x_index = int((state[0] - self.min_x) / self.bin_size_x)
        x_index = min(max(x_index, 0), self.num_bins_x - 1)
        y_index = int((state[1] - self.min_y) / self.bin_size_y)
        y_index = min(max(y_index, 0), self.num_bins_y - 1)

        self.visitations[skill_idx][y_index][x_index] += 1.
        self.visitations[-1][y_index][x_index] += 1.

        # Perform model updates
        if self.t_step % self.qlocal_update_freq == 0 and len(self.memory) >= self.config.batch_size:
            experiences = self.memory.sample()
            self.update_qlocal(experiences, self.config.gamma)
        if self.t_step % self.qtarget_update_freq == 0:
            self.update_qtarget()
        if self.t_step % self.discrim_update_freq == 0:
            experiences = self.memory.sample()
            self.update_discrim(experiences)
        self.t_step += 1

    def act(self, state, skill_idx, eps):
        state = torch.from_numpy(state).float().unsqueeze(0).to('cuda')
        skill = F.one_hot(torch.tensor(skill_idx), self.config.skill_size).reshape(1, -1).float().to('cuda')
        self.qlocal.eval()
        with torch.no_grad():
            action_values = to_numpy(self.qlocal(state, skill))
        self.qlocal.train()

        if random.random() > eps:
            action = np.argmax(action_values)
        else:
            action = random.choice(np.arange(self.config.action_size))
        return action

    def discriminate(self, state):
        state = torch.from_numpy(state).float().to('cuda').view(1, -1)
        self.discrim.eval()
        with torch.no_grad():
            predictions = self.discrim(state)[0]
        self.discrim.train()
        return predictions

    def update_qlocal(self, experiences):
        states, actions, skill_idxs, next_states, next_state_embeddings, dones = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.config.skill_size).float().to('cuda')
        
        skill_preds = self.discrim.forward(next_state_embeddings)
        selected_probs = torch.gather(skill_preds, 1, skill_idxs)
        rewards = torch.log(selected_probs + 1e-5) - math.log(1/self.config.skill_size)

        Q_targets_next = self.qtarget(next_states, skills).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qlocal(states, skills).gather(1, actions)

        reg_lambda = 1e-1
        l1_reg = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True).to('cuda')
        for W in self.qlocal.parameters():
            l1_reg = l1_reg + W.norm(1)

        loss = F.mse_loss(Q_expected, Q_targets) + reg_lambda * l1_reg
        self.qlocal_opt.zero_grad()
        loss.backward()
        self.qlocal_opt.step()

        self.stats.update({
            "critic_loss": loss.item(),
            "q_values": Q_expected.mean().item(),
            "target_values": Q_targets.mean().item(),
            "qlocal_grad_norm": grad_norm(self.qlocal),
            "reward_skill": rewards.mean().item(),
        })
    
    def update_qtarget(self):
        for target_param, local_param in zip(self.qtarget.parameters(), self.qlocal.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1.0 - self.config.tau) * target_param.data)
        
        self.stats.update({
            "qtarget_grad_norm": grad_norm(self.qtarget)
        })
    
    def update_discrim(self, experiences):
        self.discrim_opt.zero_grad()

        _, _, skill_idxs, _, next_state_embeddings, _ = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.config.skill_size).float().to('cuda')
        skill_preds = self.discrim.forward(next_state_embeddings)

        loss = self.discrim_loss_fn(skills, skill_preds)
        loss.backward()
        self.discrim_opt.step()

        skill_idx_preds = torch.argmax(skill_preds, dim=1, keepdim=True)
        acc = torch.sum(skill_idxs == skill_idx_preds) / skill_idxs.shape[0]
        ent = -torch.sum(skill_preds * torch.log(skill_preds + 1e-10) / math.log(self.config.skill_size), dim=1).mean()

        self.stats.update({
            "discrim_loss": loss.item(),
            "discrim_acc": acc.item(),
            "discrim_ent": ent.item(),
            "discrim_grad_norm": grad_norm(self.discrim),
        })

