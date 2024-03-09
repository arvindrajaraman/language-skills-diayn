import argparse
from datetime import datetime
import os
import random
import yaml
from pathlib import Path
import gym
import torch
import numpy as np
from tqdm import tqdm
import wandb
from dotenv import load_dotenv
from diayn import DIAYN
from visualization.visitation_distribution import plot_visitations
from lunarlander.captioner import naive_captioner, language_captioner

load_dotenv()
wandb.login(key=os.getenv("WANDB_API_KEY"))

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, required=True)
args = parser.parse_args()

config = yaml.safe_load(Path(os.path.join('config', args.config)).read_text())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
os.makedirs(f'./data/{run_name}')

# Initialize wandb
run = wandb.init(
    project="language-skills",
    entity="arvind6902",
    name=run_name,
)
wandb.config = config

# Choose embedding function
if config.embedding_type == "identity":
    embedding_fn = lambda x: x
elif config.embedding_type == "naive":
    embedding_fn = naive_captioner
elif config.embedding_type == "language":
    embedding_fn = language_captioner
else:
    raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

# Initialize environment and agent
env = gym.make(config.env_name)
agent = DIAYN(config=config, embedding_fn=embedding_fn)

eps = config.eps_start
for episode in tqdm(range(config.episodes)):  
    # Sample skill and initial state
    skill_idx = random.randint(0, config.skill_size - 1)
    obs = env.reset()

    for t in range(config.max_steps_per_episode):
        action = agent.act(obs, skill_idx, eps)
        next_obs, reward, done, _ = env.step(action)
        stats = agent.step(obs, action, skill_idx, next_obs, done)  # Update policy
        stats["reward_ground_truth"] = reward
        stats["eps"] = eps
        stats["episode"] = episode
        if episode % 500 == 0:
            for i in range(config.skill_size):
                stats[f"log_visits_s{i}"] = plot_visitations(agent.visitations[i], agent.min_x, agent.max_x, agent.min_y, agent.max_y, i, log=True)
            stats["log_visits_all"] = plot_visitations(agent.visitations[-1], agent.min_x, agent.max_x, agent.min_y, agent.max_y, log=True)
            agent.visitations = np.zeros((agent.config.skill_size + 1, agent.num_bins_x, agent.num_bins_y))
        wandb.log(stats)

        obs = next_obs.cpu().detach().numpy()
        if done:
            break

    eps = max(eps * config.eps_decay, config.eps_end)
    if episode % 500 == 0:
        torch.save(agent.qlocal.state_dict(), f'./data/{run_name}/qlocal_iter{episode}.pth')
        torch.save(agent.qtarget.state_dict(), f'./data/{run_name}/qtarget_iter{episode}.pth')
        torch.save(agent.discrim.state_dict(), f'./data/{run_name}/discrim_iter{episode}.pth')
