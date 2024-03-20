import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
import gym
from ml_collections import FrozenConfigDict
import os
import torch

import wandb
import yaml

from diayn import DIAYN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--nolog', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(Path(os.path.join('config', args.config)).read_text())
    if "seed" not in config:
        config['seed'] = int(datetime.now().timestamp())
    config = FrozenConfigDict(config)

    if not args.nolog:
        run_name = '{}_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
        os.makedirs(f'./data/{run_name}')

        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        run = wandb.init(
            project="language-skills",
            entity="arvind6902",
            name=run_name,
            config=config
        )
        wandb.config = config
    else:
        run_name = None

    env = gym.vector.make(config.env_name, num_envs=config.num_envs, asynchronous=True)
    agent = DIAYN(config, log=not args.nolog)
    agent.train(env, run_name)
    run.finish()
