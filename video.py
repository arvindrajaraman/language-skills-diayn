import argparse
from datetime import datetime
import os
from pathlib import Path
import random
import shutil

import gym
from gym.wrappers import RecordVideo
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import yaml

from diayn import DIAYN

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
agent = DIAYN(config=config)

for i in range(0, NUM_VIDEOS * VIDEO_FREQ, VIDEO_FREQ):
    discrim_path = os.path.join(run_dir, f'discrim_iter{i}.pth')
    qlocal_path = os.path.join(run_dir, f'qlocal_iter{i}.pth')
    qtarget_path = os.path.join(run_dir, f'qtarget_iter{i}.pth')

    agent.init_qlocal_from_path(qlocal_path)
    agent.init_qtarget_from_path(qtarget_path)
    agent.init_discrim_from_path(discrim_path)

    for skill_idx in range(config.skill_size):
        for rollout_idx in range(ROLLOUTS_PER):
            print(f'Visualizing video for iter {i}, skill {skill_idx}, rollout_idx {rollout_idx}')
            folder_name = f'iter{i}_skill{skill_idx}_rollout{rollout_idx}'
            env = gym.make(config.env_name)
            env = RecordVideo(env, os.path.join(run_dir, 'rollouts', folder_name))
            discrim_probs = []

            obs = env.reset()
            for _ in range(config.max_steps_per_episode):
                action = agent.act(obs, skill_idx, eps=0.)
                next_obs, _, done, _ = env.step(action)
                discrim_probs.append(agent.discriminate(next_obs)[skill_idx].item())
                if done:
                    break

                obs = next_obs
            
            env.close()
            with open(os.path.join(run_dir, 'rollouts', folder_name, 'discrim_probs.txt'), 'a') as f:
                f.write(f'{discrim_probs}\n')
