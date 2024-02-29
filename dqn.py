import numpy as np
import random
from collections import namedtuple, deque
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import QSkillNetwork, QConvSkillNetwork, SkillDiscriminatorNetwork, SkillConvDiscriminatorNetwork
from metrics_ll import train_mutual_info_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config: dict, conv: bool, embedding_fn: callable = lambda x: x):
        """Initialize an Agent object.

        Params
        ======
            config (dict): configurations
            conv (bool): whether to use a convolutional discriminator and policy (i.e. if your observations are images)
            embedding_fn (bool): if using embeddings of observations for the discriminator, this is the embedding function
        """
        if "seed" not in config:
            config["seed"] = np.random.randint(0, 2**31 - 1)
        random.seed(config["seed"])
        os.environ['PYTHONHASHSEED'] = str(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.config = config
        self.conv = conv
        self.embedding_fn = embedding_fn

        # Q-Network
        if conv:
            self.qnetwork_local = QConvSkillNetwork(config["state_size"], config["action_size"], config["skill_size"], config["policy_units"], config["policy_units"]).to(device)
            self.qnetwork_target = QConvSkillNetwork(config["state_size"], config["action_size"], config["skill_size"], config["policy_units"], config["policy_units"]).to(device)
        else:
            self.qnetwork_local = QSkillNetwork(config["state_size"], config["action_size"], config["skill_size"], config["policy_units"], config["policy_units"]).to(device)
            self.qnetwork_target = QSkillNetwork(config["state_size"], config["action_size"], config["skill_size"], config["policy_units"], config["policy_units"]).to(device)
        self.qnetwork_local = nn.DataParallel(self.qnetwork_local)
        self.qnetwork_target = nn.DataParallel(self.qnetwork_target)
        self.qnetwork_local.train()
        self.qnetwork_target.train()
        self.qnetwork_optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config["policy_lr"])

        # Replay memory
        self.memory = ReplayBuffer(config["action_size"], config["skill_size"], config["buffer_size"], config["batch_size"])
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        # Visitation distribution
        self.max_x = self.max_y = 1.5
        self.min_x = self.min_y = -1.5
        self.num_bins_x = self.num_bins_y = 50
        self.bin_size_x = (self.max_x - self.min_x) / self.num_bins_x
        self.bin_size_y = (self.max_y - self.min_y) / self.num_bins_y
        self.visitations = np.zeros((self.config["skill_size"] + 1, self.num_bins_x, self.num_bins_y))

        # Setup discriminator
        if conv:
            self.discriminator = SkillConvDiscriminatorNetwork(config["state_size"], config["skill_size"], config["discrim_units"], config["discrim_units"]).to(device)
        else:
            self.discriminator = SkillDiscriminatorNetwork(config["state_size"], config["skill_size"], config["discrim_units"], config["discrim_units"]).to(device)
        self.discriminator = nn.DataParallel(self.discriminator)
        self.discriminator.train()
        self.discriminator_criterion = nn.CrossEntropyLoss()
        self.discriminator_optimizer = optim.SGD(self.discriminator.parameters(), lr=config["discrim_lr"], momentum=config["discrim_momentum"])

        # Goal tracking
        self.last_n_goals = deque(maxlen=100)

    def init_qnetwork_local_from_path(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.qnetwork_local.eval()

    def init_qnetwork_target_from_path(self, path):
        self.qnetwork_target.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.qnetwork_target.eval()

    def init_discriminator_from_path(self, path):
        self.discriminator.load_state_dict(torch.load(path, map_location=torch.device(device)))
        self.discriminator.eval()

    def step(self, state, action, skill_idx, next_state, done):
        stats = {}

        # Save experience in replay memory
        state_embedding = self.embedding_fn(state)
        next_state_embedding = self.embedding_fn(next_state)
        self.memory.add(state, state_embedding, action, skill_idx, next_state, next_state_embedding, done)

        if done and (self.config["skill_size"] == 3 and self.config["env_name"] == "LunarLander-v2"):
            self.last_n_goals.append((next_state, skill_idx))
            stats["mutual_info_train"] = train_mutual_info_score(self.config, self.last_n_goals)

        # Record visitation
        x_index = int((state[0] - self.min_x) / self.bin_size_x)
        x_index = min(max(x_index, 0), self.num_bins_x - 1)
        y_index = int((state[1] - self.min_y) / self.bin_size_y)
        y_index = min(max(y_index, 0), self.num_bins_y - 1)

        self.visitations[skill_idx][y_index][x_index] += 1
        self.visitations[-1][y_index][x_index] += 1

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.config["update_every"]
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.config["batch_size"]:
                experiences = self.memory.sample()
                stats.update(self.learn(experiences, self.config["gamma"]))
                stats.update(self.update_discriminator(experiences))
                return stats
            
        return stats

    def act(self, state, skill_idx, eps):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        skill = F.one_hot(torch.tensor(skill_idx), self.config["skill_size"]).reshape(1, -1).float().to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state, skill)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.config["action_size"]))

        return action

    def discriminate(self, state):
        state = torch.from_numpy(state).float().to(device).view(1, -1)
        self.discriminator.eval()
        with torch.no_grad():
            predictions = self.discriminator(state)[0]
        self.discriminator.train()
        return predictions

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, _, actions, skill_idxs, next_states, _, dones = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.config["skill_size"]).float().to(device)
        
        # Calculate rewards using current discriminator
        skill_preds = self.discriminator.forward(next_states)
        selected_probs = torch.gather(skill_preds, 1, skill_idxs)
        rewards = torch.log(selected_probs + 1e-5) - math.log(1/self.config["skill_size"])

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
        self.qnetwork_optimizer.zero_grad()
        loss.backward()
        self.qnetwork_optimizer.step()

        total_norm = 0.0
        for p in self.qnetwork_local.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.config["tau"])

        return {
            "critic_loss": loss.item(),
            "q_values": Q_expected.mean().item(),
            "target_values": Q_targets.mean().item(),
            "grad_norm": total_norm,
            "reward_skill": rewards.mean().item(),
        }
    
    def update_discriminator(self, experiences):
        self.discriminator_optimizer.zero_grad()

        _, _, _, skill_idxs, _, next_state_embeddings, _ = experiences
        skills = F.one_hot(skill_idxs.view(-1), self.config["skill_size"]).float().to(device)
        skill_preds = self.discriminator.forward(next_state_embeddings)

        loss = self.discriminator_criterion(skills, skill_preds)
        loss.backward()
        self.discriminator_optimizer.step()

        skill_idx_preds = torch.argmax(skill_preds, dim=1, keepdim=True)
        acc = torch.sum(skill_idxs == skill_idx_preds) / skill_idxs.shape[0]

        ent = -torch.sum(skill_preds * torch.log(skill_preds + 1e-10) / math.log(self.config["skill_size"]), dim=1).mean()

        total_norm = 0.0
        for p in self.discriminator.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return {
            "discrim_loss": loss.item(),
            "discrim_acc": acc.item(),
            "discrim_ent": ent.item(),
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

    def __init__(self, action_size, skill_size, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.skill_size = skill_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "state_embedding", "action", "skill_idx", "next_state", "next_state_embedding", "done"])

    def add(self, state, state_embedding, action, skill_idx, next_state, next_state_embedding, done):
        """Add a new experience to memory."""
        e = self.experience(state, state_embedding, action, skill_idx, next_state.cpu().detach().numpy(), next_state_embedding.cpu().detach().numpy(), done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(self.batch_size * self.skill_size, len(self.memory)))

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to(device)
        state_embeddings = torch.from_numpy(np.stack([e.state_embedding for e in experiences if e is not None], axis=0)).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).long().to(device).view(-1, 1)
        skill_idxs = torch.from_numpy(np.stack([e.skill_idx for e in experiences if e is not None], axis=0)).long().to(device).view(-1, 1)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float().to(device)
        next_state_embeddings = torch.from_numpy(np.stack([e.next_state_embedding for e in experiences if e is not None], axis=0)).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=0).astype(np.uint8)).float().to(device).view(-1, 1)

        return (states, state_embeddings, actions, skill_idxs, next_states, next_state_embeddings, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
