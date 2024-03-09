from collections import deque, namedtuple
import random

import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, action_size, skill_size, buffer_size, batch_size):
        self.action_size = action_size
        self.skill_size = skill_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "skill_idx", "next_state", "next_state_embedding", "done"])

    def add(self, state, action, skill_idx, next_state, next_state_embedding, done):
        e = self.experience(state, action, skill_idx, next_state, next_state_embedding, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=min(self.batch_size * self.skill_size, len(self.memory)))

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None], axis=0)).float().to('cuda')
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None], axis=0)).long().to('cuda').view(-1, 1)
        skill_idxs = torch.from_numpy(np.stack([e.skill_idx for e in experiences if e is not None], axis=0)).long().to('cuda').view(-1, 1)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None], axis=0)).float().to('cuda')
        next_state_embeddings = torch.from_numpy(np.stack([e.next_state_embedding for e in experiences if e is not None], axis=0)).float().to('cuda')
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None], axis=0).astype(np.uint8)).float().to('cuda').view(-1, 1)

        return (states, actions, skill_idxs, next_states, next_state_embeddings, dones)

    def __len__(self):
        return len(self.memory)
