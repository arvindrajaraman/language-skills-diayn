import wandb
wandb.login()

sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "discrim_acc"},
    "parameters": {
        "eps_start": {"values": [1.0]},
        "eps_end": {"values": [0.25, 0.01]},
        "eps_decay": {"values": [0.9995, 0.995]},
        "discrim_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "policy_lr": {"values": [1e-5, 1e-4, 1e-3, 1e-2]},
        "gamma": {"values": [0.7, 0.9, 0.95, 0.99, 0.999, 1.0]},
        "tau": {"values": [1e-4, 1e-3, 1e-2]},
        "update_every": {"values": [1, 2, 4, 8, 16, 32]},
        "discrim_momentum": {"values": [0.9, 0.95, 0.99]},
        "batch_size": {"values": [32, 64, 128, 256, 512]},

        "env_name": {"values": ["LunarLander-v2"]},
        "exp_type": {"values": ["mlp"]},
        "embedding_type": {"values": ["identity"]},
        "episodes": {"values": [5000]},
        "max_steps_per_episode": {"values": [300]},
        "state_size": {"values": [8]},
        "action_size": {"values": [4]},
        "skill_size": {"values": [3]},
        "embedding_size": {"values": [3]},
        "buffer_size": {"values": [10000]},
        "discrim_units": {"values": [512]},
        "policy_units": {"values": [512]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="language-skills", entity="arvind6902")
print('Sweep ID:', sweep_id)
