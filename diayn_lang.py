from datetime import datetime
import os
import random

import gym
import torch
from tqdm import tqdm
# import wandb
# wandb.login()
from sentence_transformers import SentenceTransformer

from dqn import Agent
from crafter_vis import record_rollouts
from captioner import get_captioner

import text_crafter

config = {
    "action_size": 27,
    "batch_size": 128,
    "buffer_size": 10000,
    "discrim_lr": 1e-4,
    "discrim_momentum": 0.99,
    "env_name": "CrafterReward-v1",
    "episodes": 5000,
    "eps_decay": 0.995,
    "eps_end": 0.01,
    "eps_start": 1.0,
    "gamma": 0.7,
    "max_steps_per_episode": 100,
    "policy_lr": 1e-5,
    "skill_size": 5,
    "state_size": 384,
    "tau": 0.0001,       # for soft update of target parameters
    "update_every": 10,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}'.format(config["env_name"], int(datetime.now().timestamp()))
os.makedirs(f'./data/{run_name}')

# # Initialize wandb
# run = wandb.init(
#     project="language-skills",
#     entity="arvind6902",
#     name=run_name,
# )
# wandb.config = config

# Initialize environment and agent
env = gym.make(config["env_name"])
agent = Agent(config=config, conv=False)

_, caption_state, _ = get_captioner()
caption_embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

eps = config["eps_start"]
for episode in tqdm(range(config["episodes"])):
    if episode % 500 == 0:
        record_stats = record_rollouts(agent, env, config)
    else:
        record_stats = None
    
    # Sample skill and initial state
    skill_idx = random.randint(0, config["skill_size"] - 1)
    obs, info = env.reset()
    obs = obs['obs']

    for t in range(config["max_steps_per_episode"]):
        obs_caption = caption_state(info)
        obs_embedding = caption_embedder.encode(obs_caption)

        action = agent.act(obs_embedding, skill_idx, eps)
        next_obs, reward, done, info = env.step(action)
        next_obs = next_obs['obs']
        next_obs = torch.tensor(next_obs).to(device).float()

        next_obs_caption = caption_state(info)
        next_obs_embedding = caption_embedder.encode(next_obs_caption)

        stats = agent.step(obs_embedding, action, skill_idx, next_obs_embedding, done)  # Update policy
        stats["reward_ground_truth"] = reward
        stats["eps"] = eps
        if record_stats is not None:
            stats.update(record_stats)
        # wandb.log(stats)

        obs = next_obs.cpu().detach().numpy()
        if done:
            break

    eps = max(eps * config["eps_decay"], config["eps_end"])
    if episode % 500 == 0:
        torch.save(agent.qnetwork_local.state_dict(), f'./data/{run_name}/qnetwork_local_iter{episode}.pth')
        torch.save(agent.qnetwork_target.state_dict(), f'./data/{run_name}/qnetwork_target_iter{episode}.pth')
        torch.save(agent.discriminator.state_dict(), f'./data/{run_name}/discriminator_iter{episode}.pth')
