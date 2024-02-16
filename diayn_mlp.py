from datetime import datetime
import os
import random

import gym
import torch
from tqdm import tqdm
import wandb
wandb.login()

from dqn import Agent

config = {
    "action_size": 4,
    "batch_size": 64,
    "buffer_size": 10000,
    "discrim_lr": 0.01,
    "discrim_momentum": 0.95,
    "env_name": "LunarLander-v2",
    "episodes": 5000,
    "eps_decay": 0.9995,
    "eps_end": 0.01,
    "eps_start": 1.0,
    "gamma": 0.99,
    "max_steps_per_episode": 300,
    "policy_lr": 0.01,
    "skill_size": 2,
    "state_size": 8,
    "tau": 0.001,       # for soft update of target parameters
    "update_every": 1,
    "exp_type": "mlp",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}_{}_{}'.format(config["env_name"], config["exp_type"], config["skill_size"], int(datetime.now().timestamp()))
os.makedirs(f'./data/{run_name}')

# Initialize wandb
run = wandb.init(
    project="language-skills",
    entity="arvind6902",
    name=run_name,
)
wandb.config = config

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
    if episode % 500 == 0:
        torch.save(agent.qnetwork_local.state_dict(), f'./data/{run_name}/qnetwork_local_iter{episode}.pth')
        torch.save(agent.qnetwork_target.state_dict(), f'./data/{run_name}/qnetwork_target_iter{episode}.pth')
        torch.save(agent.discriminator.state_dict(), f'./data/{run_name}/discriminator_iter{episode}.pth')
