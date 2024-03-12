import cv2
import gym
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import torch
import wandb

from diayn import DIAYN

def get_frame(
    env: gym.Env,
    action: int
):
    """
    Compute a frame of the environment for a given action.

    Args:
        env (gym.Env): Environment to render
        action (int): Action to take
    """
    img = env.render((512, 512))
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = np.asarray(img, dtype=np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 30), 'Took action: ' +  env.get_action_name(action) + f' - unknown',(255,255,255))
    img = np.asarray(img)
    return img

def record_rollouts(
    agent: DIAYN,
    config: dict,
    env: gym.Env,
    device: torch.device
):
    """
    Record rollouts for the given agent and environment.

    Args:
        agent (DIAYN): Agent to record rollouts for
        config (dict): Configuration dictionary
        env (gym.Env): Environment to record rollouts for
        device (torch.device): Device to use for recording rollouts
    """
    agent.qlocal.eval()
    stats = dict()
    for skill_idx in range(0, config.skill_size):
        print(f'Visualizing video for skill {skill_idx}')
        frames = []
        obs, _ = env.reset()
        obs = obs['obs']
        for _ in range(config.max_steps_per_episode):
            action = agent.act(obs, skill_idx, eps=0.)
            next_obs, _, done, _ = env.step(action)
            next_obs = next_obs['obs']
            next_obs = torch.tensor(next_obs).to(device).float()
            obs = next_obs.cpu().detach().numpy()
            frames.append(get_frame(env, action))
            if done:
                break
        env.close()

        frames = np.stack(frames) # THWC
        frames = np.moveaxis(frames, [3], [1])
        stats[f'skill_{skill_idx}_video'] = wandb.Video(frames)
    agent.qlocal.train()
    return stats
