import argparse
import os

from datetime import datetime
import random

import gym
import torch
from tqdm import tqdm
import wandb
wandb.login()

from dqn import Agent

import text_crafter

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', '-s', type=str, required=True)
args = parser.parse_args()

def train():
    wandb.init(
        project="language-skills",
        entity="arvind6902"
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = '{}_{}'.format(config["env_name"], int(datetime.now().timestamp()))
    os.makedirs(f'./data/{run_name}')

    # Initialize environment and agent
    env = gym.make(config["env_name"])
    agent = Agent(config=config, conv=True)

    eps = config["eps_start"]
    for episode in tqdm(range(config["episodes"])):
        # Sample skill and initial state
        skill_idx = random.randint(0, config["skill_size"] - 1)
        obs, _ = env.reset()
        obs = obs['obs']

        for t in range(config["max_steps_per_episode"]):
            action = agent.act(obs, skill_idx, eps)
            next_obs, reward, done, _ = env.step(action)
            next_obs = next_obs['obs']
            next_obs = torch.tensor(next_obs).to(device).float()
            stats = agent.step(obs, action, skill_idx, next_obs, done)  # Update policy
            stats["reward_ground_truth"] = reward
            stats["eps"] = eps
            wandb.log(stats)

            obs = next_obs.cpu().detach().numpy()
            if done:
                break

        eps = max(eps * config["eps_decay"], config["eps_end"])

wandb.agent(sweep_id=args.sweep_id, function=train)
