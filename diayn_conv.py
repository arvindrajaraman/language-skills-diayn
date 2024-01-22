from datetime import datetime
import math
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb

from dqn_diayn import Agent
from models import SkillConvDiscriminatorNetwork

import text_crafter

ENV_NAME = "CrafterReward-v1"
EPOCHS = 5000
SKILL_NUM = 5
MAX_STEPS_PER_EPISODE = 100
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995
DISCRIMINATOR_LR = 1e-4

SEED = 0
STATE_SIZE = (64, 64, 3)
ACTION_SIZE = 27

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = '{}_{}'.format(ENV_NAME, int(datetime.now().timestamp()))

def skill_1he(skill_num):
    z = torch.zeros(SKILL_NUM).to(device)
    z[skill_num] = 1.0
    return z

def generate_test_video(agent, skill_num, skill, iter):
    test_env = gym.make(ENV_NAME)
    test_env.seed(SEED + 1)
    test_env = text_crafter.Recorder(test_env, f'./data/{run_name}/iter{iter}_skill{skill_num}')

    agent.set_curr_skill(skill)

    state, _ = test_env.reset()
    for _ in range(MAX_STEPS_PER_EPISODE):
        action = agent.act(state['obs'], eps=0.)
        # action = test_env.action_space.sample()
        print(action)
        
        next_state, _, done, _ = test_env.step(action)
        if done:
            break

        state = next_state

    test_env._video_save()

# Initialize wandb
run = wandb.init(
    project="language-skills",
    entity="arvind6902"
)
wandb.config = {
    "epochs": EPOCHS,
    "skill_num": SKILL_NUM,
    "max_steps_per_episode": MAX_STEPS_PER_EPISODE,
    "eps_start": EPS_START,
    "eps_end": EPS_END,
    "eps_decay": EPS_DECAY,
    "discriminator_lr": DISCRIMINATOR_LR
}

# Initialize environment and agent
env = gym.make(ENV_NAME)
env.seed(SEED)
agent = Agent(
    state_size=STATE_SIZE,
    action_size=ACTION_SIZE,
    skill_size=SKILL_NUM,
    conv=True,
    seed=SEED
)

# Initialize the discriminator
discriminator = SkillConvDiscriminatorNetwork(
    state_shape=STATE_SIZE,
    skill_size=SKILL_NUM,
    seed=SEED
)
discriminator.to(device)
discriminator.train()
discriminator_criterion = nn.CrossEntropyLoss()
discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=DISCRIMINATOR_LR, momentum=0.9)

eps = EPS_START
for epoch in tqdm(range(EPOCHS)):
    if epoch % 100 == 0:
        print('Saving videos...')
        for i in tqdm(range(SKILL_NUM)):
            generate_test_video(agent, i, skill_1he(i), epoch)
    
    # Sample a random skill
    skill_num = random.randint(0, SKILL_NUM - 1)
    print('Skill:', skill_num)
    z = skill_1he(skill_num)
    agent.set_curr_skill(z)

    # Sample an initial state
    state, _ = env.reset()

    for t in range(MAX_STEPS_PER_EPISODE):
        discriminator_optimizer.zero_grad()
        discriminator.train()

        action = agent.act(state['obs'], eps=eps)
        next_state, _, done, _ = env.step(action)
        next_obs = torch.tensor(next_state['obs']).to(device).float()

        skill_pred = discriminator.forward(next_obs)
        skill_reward = math.log(skill_pred[skill_num]) - math.log(1/SKILL_NUM)
        
        agent.step(state['obs'], action, skill_reward, next_obs, done)  # Update policy
        if agent.t_step == 0:
            loss = discriminator_criterion(z, skill_pred)
            loss.backward()
            print('Discrim Loss: {}, Reward: {}'.format(loss.item(), skill_reward))
            discriminator_optimizer.step()
            wandb.log({
                "discrim_loss": loss.item(),
                "reward": skill_reward,
                "discrim_loss_skill_{}".format(skill_num): loss.item(),
                "reward_skill_{}".format(skill_num): skill_reward,
                "eps": eps
            })

        if done:
            break

        state = next_state

    # Decay epsilon (induce more exploitation)
    eps = max(EPS_END, eps * EPS_DECAY)


# Questions:
# Should we vary epsilon?
# Get rid of replay buffer?
# Only update discriminator/policy every N steps?
