from datetime import datetime
import os

import gym
from gym.wrappers import RecordVideo
import torch
from tqdm import tqdm
import wandb

from dqn_diayn import Agent

config = {
    "env_name": "LunarLander-v2",
    "episodes": 5000,
    "skill_size": 5,
    "max_steps_per_episode": 100,
    "eps_start": 1.0,
    "eps_end": 0.01,
    "eps_decay": 0.995,
    "discrim_lr": 1e-3,
    "discrim_momentum": 0.9,

    "seed": 42,
    "state_size": 8,
    "action_size": 4,

    # RL parameters
    "buffer_size": int(1e5),    # replay buffer size
    "batch_size": 64,           # minibatch size
    "gamma": 0.9,               # discount factor
    "tau": 1e-3,                # for soft update of target parameters
    "policy_lr": 5e-4,          # learning rate
    "update_every": 10          # how often to update the network
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}'.format(config["env_name"], int(datetime.now().timestamp()))
os.makedirs(f'./data/{run_name}')

def generate_test_video(agent, skill_idx, iter):
    # test_env = gym.make(ENV_NAME)
    # test_env.seed(SEED + 1)
    # test_env = RecordVideo(test_env, f'./data/{run_name}/iter{iter}_skill{skill_idx}')

    # obs = test_env.reset()
    # for _ in range(MAX_STEPS_PER_EPISODE):
    #     action = agent.act(obs, skill_idx)
    #     test_env.capture_frame()
    #     next_obs, _, done, _ = test_env.step(action)
    #     if done:
    #         break

    #     obs = next_obs

    # test_env.close()
    pass

# Initialize wandb
# run = wandb.init(
#     project="language-skills",
#     entity="arvind6902"
# )
# wandb.config = config

# Initialize environment and agent
env = gym.make(config["env_name"])
env.seed(config["seed"])
env.observation_space.seed(config["seed"])
env.action_space.seed(config["seed"])
agent = Agent(config=config, conv=False)

eps = config["eps_start"]
for episode in tqdm(range(config["episodes"])):
    # if episode % 500 == 0:
    #     print('Saving videos...')
    #     for i in tqdm(range(config["skill_size"])):
    #         generate_test_video(agent, i, episode)
    
    # Sample skill and initial state
    skill_idx = episode % config["skill_size"]
    obs = env.reset()

    for t in range(config["max_steps_per_episode"]):
        action = agent.act(obs, skill_idx, eps)
        next_obs, reward, done, _ = env.step(action)
        next_obs = torch.tensor(next_obs).to(device).float()
        stats = agent.step(obs, action, skill_idx, next_obs, done)  # Update policy
        stats["reward_ground_truth"] = reward
        # wandb.log(stats)

        obs = next_obs.cpu().detach().numpy()
        if done:
            break

    if episode % 100 == 0:
        torch.save(agent.qnetwork_local.state_dict(), f'./data/{run_name}/qnetwork_local.pth')
        torch.save(agent.qnetwork_target.state_dict(), f'./data/{run_name}/qnetwork_target.pth')
        torch.save(agent.discriminator.state_dict(), f'./data/{run_name}/discriminator.pth')
