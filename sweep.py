import argparse
from datetime import datetime
from dotenv import load_dotenv
from ml_collections import ConfigDict
import os
import wandb
from jax import random

import diayn

def agent():
    run = wandb.init(
        project="language-skills",
        entity="arvind6902"
    )
    config = wandb.config
    if "seed" not in config:
        config['seed'] = int(datetime.now().timestamp())
    config = ConfigDict(config)

    key = random.PRNGKey(config.seed)
    diayn.train(key, config, None, log=True)
    run.finish()

if __name__ == "__main__":
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', '-i', type=str, required=True)
    args = parser.parse_args()

    wandb.agent(
        sweep_id=args.id,
        function=agent,
        count=100,
        project="language-skills",
        entity="arvind6902"
    )
