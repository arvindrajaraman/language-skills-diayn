from datetime import datetime
import math
import os
import random

import gym
from gym.wrappers import RecordVideo
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from dqn_diayn import Agent

ENV_NAME = "LunarLander-v2"
EPOCHS = 5000
SKILL_SIZE = 5
MAX_STEPS_PER_EPISODE = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
DISCRIMINATOR_LR = 1e-3

SEED = 42
STATE_SIZE = 8
ACTION_SIZE = 4

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.9             # discount factor
TAU = 1e-3              # for soft update of target parameters
POLICY_LR = 5e-4        # learning rate
UPDATE_EVERY = 10       # how often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}'.format(ENV_NAME, int(datetime.now().timestamp()))
os.makedirs(f'./data/{run_name}')

def skill_1he(skill_idx, skill_size):
    z = torch.zeros(skill_size).to(device)
    z[skill_idx] = 1.0
    return z

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
# wandb.config = {
#     "epochs": EPOCHS,
#     "skill_size": SKILL_SIZE,
#     "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
#     "eps_start": EPS_START,
#     "eps_end": EPS_END,
#     "eps_decay": EPS_DECAY,
#     "discriminator_lr": DISCRIMINATOR_LR,
#     "seed": SEED,
#     "state_size": STATE_SIZE,
#     "action_size": ACTION_SIZE,

#     # RL parameters
#     "buffer_size": BUFFER_SIZE,
#     "batch_size": BATCH_SIZE,
#     "gamma": GAMMA,
#     "tau": TAU,
#     "policy_lr": POLICY_LR,
#     "update_every": UPDATE_EVERY
# }

# Initialize environment and agent
env = gym.make(ENV_NAME)
env.seed(SEED)

test_env = gym.make(ENV_NAME)
test_env.seed(SEED + 1)

agent = Agent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    skill_size=SKILL_SIZE,
    conv=False,
    seed=SEED,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    tau=TAU,
    lr=POLICY_LR,
    update_every=UPDATE_EVERY,
    discriminator_lr=DISCRIMINATOR_LR
)

eps = EPS_START
for epoch in tqdm(range(EPOCHS)):
    stats = []
    if epoch % 500 == 0:
        print('Saving videos...')
        for i in tqdm(range(SKILL_SIZE)):
            generate_test_video(agent, i, epoch)
    
    # Sample a random skill
    skill_idx = epoch % SKILL_SIZE

    # Sample an initial state
    obs = env.reset()

    next_obs_list = []
    for t in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(obs, skill_idx, eps)
        next_obs, _, done, _ = env.step(action)
        next_obs = torch.tensor(next_obs).to(device).float()
        next_obs_list.append(next_obs)

        stats = agent.step(obs, action, skill_idx, next_obs, done)  # Update policy

        if done:
            break

        obs = next_obs.cpu().detach().numpy()

    # discriminator.eval()    
    # # Discriminator probability for this traj
    # next_obs_list = torch.stack(next_obs_list)
    # probs = discriminator.forward(next_obs_list)
    # relevant_probs = probs[:, skill_idx]
    # dp_curr = relevant_probs.mean()
    # stats[-1]["dp_curr"] = dp_curr
    
    # for skill_idx in range(SKILL_SIZE):
    #     # Discriminator probability for a replay buffer sample with all of same skill
    #     _, _, _, next_states, _ = agent.memory.sample_for_skill(skill_idx)
    #     if next_states is not None:
    #         probs = discriminator.forward(next_states)
    #         relevant_probs = probs[:, skill_idx]
    #         dp_replay = relevant_probs.mean()
    #         stats[-1][f"dp_replay_skill{skill_idx}"] = dp_replay
            
    #     # Discriminator probability for a new test trajectory
    #     obs = test_env.reset()
    #     next_obs_list = []
    #     for t in range(MAX_STEPS_PER_EPISODE):
    #         action = agent.act(obs, skill_idx, eps)
    #         next_obs, _, done, _ = test_env.step(action)
    #         next_obs = torch.tensor(next_obs).to(device).float()
    #         next_obs_list.append(next_obs)
    #         if done:
    #             break
    #         obs = next_obs.cpu().detach().numpy()

    #     next_obs_list = torch.stack(next_obs_list)
    #     probs = discriminator.forward(next_obs_list)
    #     relevant_probs = probs[:, skill_idx]
    #     dp_test = relevant_probs.mean()
    #     stats[-1][f"dp_replay_skill{skill_idx}"] = dp_test

    # # Decay epsilon (induce more exploitation)
    # eps = max(EPS_END, eps * EPS_DECAY)
    # stats[-1]["eps"] = eps
        
    # # for stat in stats:
    # #     wandb.log(stat)

    if epoch % 100 == 0:
        torch.save(agent.qnetwork_local.state_dict(), f'./data/{run_name}/qnetwork_local.pth')
        torch.save(agent.qnetwork_target.state_dict(), f'./data/{run_name}/qnetwork_target.pth')
        torch.save(agent.discriminator.state_dict(), f'./data/{run_name}/discriminator.pth')
