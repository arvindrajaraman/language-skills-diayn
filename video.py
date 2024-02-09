import argparse
import os
import shutil

from datetime import datetime
import random

import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from dqn_diayn import Agent

from gym.wrappers import RecordVideo

config = {
    "action_size": 4,
    "batch_size": 128,
    "buffer_size": 10000,
    "discrim_lr": 0.01,
    "discrim_momentum": 0.99,
    "env_name": "LunarLander-v2",
    "max_steps_per_episode": 300,
    "policy_lr": 1e-5,
    "skill_size": 5,
    "state_size": 8,
}

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', help='Run name', required=True)
args = parser.parse_args()

run_name = args.name
run_dir = f'./data/{run_name}'

NUM_VIDEOS = 10
VIDEO_FREQ = 500
NUM_SKILLS = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(config=config, conv=False)

for i in range(0, NUM_VIDEOS * VIDEO_FREQ, VIDEO_FREQ):
    discriminator_path = os.path.join(run_dir, f'discriminator_iter{i}.pth')
    qnetwork_local_path = os.path.join(run_dir, f'qnetwork_local_iter{i}.pth')
    qnetwork_target_path = os.path.join(run_dir, f'qnetwork_target_iter{i}.pth')

    agent.init_qnetwork_local_from_path(qnetwork_local_path)
    agent.init_qnetwork_target_from_path(qnetwork_target_path)
    agent.init_discriminator_from_path(discriminator_path)

    for skill_idx in range(NUM_SKILLS):
        print(f'Visualizing video for iter {i}, skill {skill_idx}')
        env = gym.make(config["env_name"])
        env = RecordVideo(env, os.path.join(run_dir, f'iter{i}_skill{skill_idx}'))

        obs = env.reset()
        for _ in range(config["max_steps_per_episode"]):
            action = agent.act(obs, skill_idx, eps=0.)
            next_obs, _, done, _ = env.step(action)
            if done:
                break

            obs = next_obs
        
        env.close()
