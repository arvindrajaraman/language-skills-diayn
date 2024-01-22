import numpy as np
import random
from collections import namedtuple, deque
import calendar
import math
import os

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import QSkillNetwork, QConvSkillNetwork, SkillDiscriminatorNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, skill_size, conv, seed,
                 buffer_size, batch_size, gamma, tau, lr, update_every,
                 discriminator_lr):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.skill_size = skill_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.discriminator_lr = discriminator_lr

        # Q-Network
        if conv:
            self.qnetwork_local = QConvSkillNetwork(state_size, action_size, skill_size, seed).to(device)
            self.qnetwork_target = QConvSkillNetwork(state_size, action_size, skill_size, seed).to(device)
        else:
            self.qnetwork_local = QSkillNetwork(state_size, action_size, skill_size, seed).to(device)
            self.qnetwork_target = QSkillNetwork(state_size, action_size, skill_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.skill_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Skill-related
        if conv:
            raise NotImplementedError
        else:
            self.discriminator = SkillDiscriminatorNetwork(state_size, skill_size, seed).to(device)
            self.discriminator.train()
            self.discriminator_criterion = nn.CrossEntropyLoss()
            self.discriminator_optimizer = optim.SGD(self.discriminator.parameters(), lr=discriminator_lr, momentum=0.9)

    def step(self, state, action, skill_idx, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, skill_idx, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                stats = self.learn(experiences, self.gamma)
                stats.update(self.update_discriminator(experiences))
                return stats
            
        return {}

    def act(self, state, skill_idx, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        skill = F.one_hot(torch.tensor(skill_idx), self.skill_size).reshape(1, -1).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, skill)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        return action

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, skill_idxs, next_states, dones = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.skill_size).float().to(device)
        
        # Calculate rewards using current discriminator
        skill_preds = self.discriminator.forward(next_states)
        selected_probs = torch.gather(skill_preds, 1, skill_idxs)
        rewards = torch.log(selected_probs) - math.log(1/self.skill_size)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states, skills).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states, skills).gather(1, actions)

        # Compute L1 term
        reg_lambda = 1e-1
        l1_reg = torch.autograd.Variable(torch.FloatTensor(1), requires_grad=True).to(device)
        for W in self.qnetwork_local.parameters():
            l1_reg = l1_reg + W.norm(1)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets) + reg_lambda * l1_reg
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        total_norm = 0.0
        for p in self.qnetwork_local.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

        return {
            "critic_loss": loss.item(),
            "q_values": Q_expected.mean().item(),
            "target_values": Q_targets.mean().item(),
            "grad_norm": total_norm,
            "rewards": rewards.mean().item(),
        }
    
    def update_discriminator(self, experiences):
        self.discriminator_optimizer.zero_grad()

        _, _, skill_idxs, next_states, _ = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.skill_size).float().to(device)
        skill_preds = self.discriminator.forward(next_states)

        loss = self.discriminator_criterion(skills, skill_preds)
        loss.backward()
        self.discriminator_optimizer.step()

        skill_idx_preds = torch.argmax(skill_preds, dim=1, keepdim=True)
        acc = torch.sum(skill_idxs == skill_idx_preds) / skill_idxs.shape[0]

        total_norm = 0.0
        for p in self.discriminator.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return {
            "discrim_loss": loss.item(),
            "discrim_acc": acc.item(),
            "discrim_grad_norm": total_norm,
        }

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, skill_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.skill_size = skill_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "skill_idx", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, skill_idx, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, skill_idx, next_state.cpu().detach().numpy(), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(self.batch_size * self.skill_size, len(self.memory)))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        skill_idxs = torch.from_numpy(np.vstack([e.skill_idx for e in experiences if e is not None])).long().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, skill_idxs, next_states, dones)

    def sample_for_skill(self, skill_idx):
        """Randomly sample a batch of experiences from memory for a specific skill."""
        experiences = random.sample(self.memory, k=min(self.batch_size * self.skill_size, len(self.memory)))
        experiences = [e for e in experiences if e is not None and e.skill_idx == skill_idx]
        if len(experiences) == 0:
            return None, None, None, None, None

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        skill_idxs = torch.from_numpy(np.vstack([e.skill_idx for e in experiences if e is not None])).long().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, skill_idxs, next_states, dones)


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# def train(agent,
#     n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
#     """Deep Q-Learning.

#     Params
#     ======
#         agent: agent to train
#         n_episodes (int): maximum number of training episodes
#         max_t (int): maximum number of timesteps per episode
#         eps_start (float): starting value of epsilon, for epsilon-greedy action selection
#         eps_end (float): minimum value of epsilon
#         eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
#     """
#     scores = []                        # list containing scores from each episode
#     scores_window = deque(maxlen=100)  # last 100 scores
#     eps = eps_start                    # initialize epsilon
#     savenumber = 0
#     for i_episode in range(1, n_episodes+1):
#         state = env.reset()
#         score = 0
#         for t in range(max_t):
#             action = agent.act(state, eps)
#             next_state, _, done, _ = env.step(action)

#             # Calculating reward has been removed
#             raise NotImplementedError

#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             score += reward
#             if done:
#                 break
#         scores_window.append(score)       # save most recent score
#         scores.append(score)              # save most recent score
#         eps = max(eps_end, eps_decay*eps) # decrease epsilon
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#         if i_episode % 100 == 0:
#             savename = "data/dqn_" + str(savenumber) + ".pth"
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
#             torch.save(agent.qnetwork_local.state_dict(), savename)
#             savenumber += 1
#     return scores

# if __name__ == "__main__":
#     env = gym.make("LunarLander-v2")
#     env.seed(0)
#     agent = Agent(state_size=8, action_size=4, seed=0)
#     train(agent)
