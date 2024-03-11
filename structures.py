from collections import deque, namedtuple
import random

import numpy as np
import torch

from utils import to_torch, to_torch_bool, to_torch_long

class ReplayBuffer:
    def __init__(self, config):
        actual_buffer_size = config.buffer_size + config.batch_size
        self.config = config
        self.states = torch.empty(actual_buffer_size, config.state_size, dtype=torch.float32).to('cuda')
        self.actions = torch.empty(actual_buffer_size, dtype=torch.int64).to('cuda')
        self.skills = torch.empty(actual_buffer_size, config.skill_size, dtype=torch.float32).to('cuda')
        self.next_states = torch.empty(actual_buffer_size, config.state_size, dtype=torch.float32).to('cuda')
        self.next_state_embeddings = torch.empty(actual_buffer_size, config.embedding_size, dtype=torch.float32).to('cuda')
        self.dones = torch.empty(actual_buffer_size, dtype=torch.bool).to('cuda')
        self.counter = 0

    def add(self, state, action, skill, next_state, next_state_embedding, done):
        ptr = self.counter % self.config.buffer_size
        self.states[ptr:ptr+self.config.num_envs] = to_torch(state)
        self.actions[ptr:ptr+self.config.num_envs] = to_torch_long(action)
        self.skills[ptr:ptr+self.config.num_envs] = to_torch(skill)
        self.next_states[ptr:ptr+self.config.num_envs] = to_torch(next_state)
        self.next_state_embeddings[ptr:ptr+self.config.num_envs] = to_torch(next_state_embedding)
        self.dones[ptr:ptr+self.config.num_envs] = to_torch_bool(done)
        self.counter += self.config.num_envs

    def sample(self):
        idxs = np.random.randint(0, min(self.config.buffer_size, self.counter), self.config.batch_size)
        states = self.states[idxs]
        actions = self.actions[idxs].view(-1, 1)
        skills = self.skills[idxs]
        next_states = self.next_states[idxs]
        next_state_embeddings = self.next_state_embeddings[idxs]
        dones = self.dones[idxs].view(-1, 1)
        return states, actions, skills, next_states, next_state_embeddings, dones

    def __len__(self):
        return min(self.config.buffer_size, self.counter)
