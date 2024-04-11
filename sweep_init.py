import argparse
from pathlib import Path
import os

from dotenv import load_dotenv
import wandb
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    args = parser.parse_args()

    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    config = yaml.safe_load(Path(os.path.join('config', args.config)).read_text())    
    sweep_id = wandb.sweep(
        sweep=config,
        project="language-skills",
        entity="arvind6902"
    )
    print('Sweep id:', sweep_id)
