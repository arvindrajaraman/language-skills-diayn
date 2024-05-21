import wandb
import os
import ast
from tqdm import tqdm
import numpy as np

# List all files in data directory
files = os.listdir("ellm/data")

achievement_names = [
    'collect_coal',
    'collect_diamond',
    'collect_drink',
    'collect_iron',
    'collect_sapling',
    'collect_stone',
    'collect_wood',
    'defeat_skeleton',
    'defeat_zombie',
    'eat_cow',
    'eat_plant',
    'make_iron_pickaxe',
    'make_iron_sword',
    'make_stone_pickaxe',
    'make_stone_sword',
    'make_wood_pickaxe',
    'make_wood_sword',
    'place_furnace',
    'place_plant',
    'place_stone',
    'place_table',
    'wake_up'
]

for file in files:
    filepath = os.path.join(os.getcwd(), 'ellm/data', file)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        run_name = 'ellm_{}'.format(file.split('_')[1][:-4])
        
        wandb.login(key='c8fc98b4e8461d1de58e3e8a09d13af0d96e72b2')
        
        config = {
            'name': 'ellm'
        }
        run = wandb.init(
            project="language-skills",
            entity="arvind6902",
            name=run_name,
            config=config
        )
        wandb.config = config
        
        for episodes, line in tqdm(enumerate(lines)):
            step, achievement_counts = line.strip().split(' [')
            achievement_counts = '[' + achievement_counts
            
            step = int(step)
            achievement_counts = ast.literal_eval(achievement_counts)
            
            achievement_counts = np.array(achievement_counts)
            achievement_counts *= step
            
            achievement_counts /= (episodes + 1)
            
            achievement_success_rates = (achievement_counts * 100.0)
            crafter_score = np.nan_to_num(
                np.exp(np.log(achievement_success_rates + 1.0).mean()) - 1.0,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            metrics = dict()
            metrics['score/crafter'] = crafter_score
            
            for achievement_name, achievement_freq in zip(achievement_names, achievement_counts):
                metrics[f'achievements/{achievement_name}'] = achievement_freq
                
            wandb.log(metrics, step=step)
            
        run.finish()