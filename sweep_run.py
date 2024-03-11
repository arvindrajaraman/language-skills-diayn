import argparse

from dotenv import load_dotenv
import gym
from ml_collections import FrozenConfigDict
import os

import wandb

from diayn import DIAYN

def train():
    run = wandb.init(
        project="language-skills",
        entity="arvind6902"
    )
    config = wandb.config
    env = gym.vector.make(config.env_name, num_envs=config.num_envs, asynchronous=True)
    agent = DIAYN(config)
    agent.train(env, None)
    run.finish()

if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', '-s', type=str, required=True)
    args = parser.parse_args()

    wandb.agent(sweep_id=args.sweep_id, function=train)
