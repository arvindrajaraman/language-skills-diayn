import os

from dotenv import load_dotenv
import wandb
import yaml

if __name__ == '__main__':
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    with open('sweep_config.yaml') as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="language-skills",
        entity="arvind6902"
    )
    print('Sweep id:', sweep_id)
