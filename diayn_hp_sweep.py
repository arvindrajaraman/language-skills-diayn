import os

from datetime import datetime
import random

import gym
import torch
from tqdm import tqdm
import wandb
wandb.login()

from dqn_diayn import Agent

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "discrim_acc"},
    "parameters": {
        "eps_start": {"values": [1.0, 0.75, 0.5, 0.25, 0.01]},
        "eps_end": {"values": [0.5, 0.25, 0.01]},
        "eps_decay": {"values": [0.9995, 0.995, 0.95, 0.9]},
        "discrim_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "policy_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "gamma": {"values": [0.7, 0.9, 0.95, 0.99, 0.999, 1.0]},
        "tau": {"values": [1e-4, 1e-3, 1e-2]},
        "update_every": {"values": [1, 2, 4, 8, 16, 32]},
        "discrim_momentum": {"values": [0.9, 0.95, 0.99]},
        "batch_size": {"values": [32, 64, 512]},

        "env_name": {"values": ["LunarLander-v2"]},
        "episodes": {"values": [5000]},
        "max_steps_per_episode": {"values": [300]},
        "state_size": {"values": [8]},
        "action_size": {"values": [4]},
        "skill_size": {"values": [5]},
        "buffer_size": {"values": [10000]}
    }
}

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
    agent = Agent(config=config, conv=False)

    eps = config["eps_start"]
    for episode in tqdm(range(config["episodes"])):
        # Sample skill and initial state
        skill_idx = random.randint(0, config["skill_size"] - 1)
        obs = env.reset()

        for t in range(config["max_steps_per_episode"]):
            action = agent.act(obs, skill_idx, eps)
            next_obs, reward, done, _ = env.step(action)
            next_obs = torch.tensor(next_obs).to(device).float()
            stats = agent.step(obs, action, skill_idx, next_obs, done)  # Update policy
            stats["reward_ground_truth"] = reward
            stats["eps"] = eps
            wandb.log(stats)

            obs = next_obs.cpu().detach().numpy()
            if done:
                break

        eps = max(eps * config["eps_decay"], config["eps_end"])


sweep_id = wandb.sweep(sweep=sweep_config, project="language-skills", entity="arvind6902")
wandb.agent(sweep_id=sweep_id, function=train)
