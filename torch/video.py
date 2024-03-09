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
from dqn import Agent
from gym.wrappers import RecordVideo
import yaml
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='Config', required=True)
parser.add_argument('-n', '--name', help='Run name', required=True)
args = parser.parse_args()

config = yaml.safe_load(Path(os.path.join('config', args.config)).read_text())

run_name = args.name
run_dir = f'./data/{run_name}'

NUM_VIDEOS = 10
VIDEO_FREQ = 500
ROLLOUTS_PER = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = Agent(config=config, conv=False)

for i in range(0, NUM_VIDEOS * VIDEO_FREQ, VIDEO_FREQ):
    discriminator_path = os.path.join(run_dir, f'discriminator_iter{i}.pth')
    qnetwork_local_path = os.path.join(run_dir, f'qnetwork_local_iter{i}.pth')
    qnetwork_target_path = os.path.join(run_dir, f'qnetwork_target_iter{i}.pth')

    agent.init_qnetwork_local_from_path(qnetwork_local_path)
    agent.init_qnetwork_target_from_path(qnetwork_target_path)
    agent.init_discriminator_from_path(discriminator_path)

    for skill_idx in range(config.skill_size):
        for rollout_idx in range(ROLLOUTS_PER):
            print(f'Visualizing video for iter {i}, skill {skill_idx}, rollout_idx {rollout_idx}')
            folder_name = f'iter{i}_skill{skill_idx}_rollout{rollout_idx}'
            env = gym.make(config.env_name)
            env = RecordVideo(env, os.path.join(run_dir, 'rollouts', folder_name))
            discriminator_probs = []

            obs = env.reset()
            for _ in range(config.max_steps_per_episode):
                action = agent.act(obs, skill_idx, eps=0.)
                next_obs, _, done, _ = env.step(action)
                discriminator_probs.append(agent.discriminate(next_obs)[skill_idx].item())
                if done:
                    break

                obs = next_obs
            
            env.close()
            with open(os.path.join(run_dir, 'rollouts', folder_name, 'discriminator_probs.txt'), 'a') as f:
                f.write(f'{discriminator_probs}\n')
