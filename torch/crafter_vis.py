import cv2
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import torch
import wandb

def get_frame(env, action):
    img = env.render((512, 512))
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    img = np.asarray(img, dtype=np.uint8)
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    draw.text((0, 30), 'Took action: ' +  env.get_action_name(action) + f' - unknown',(255,255,255))
    img = np.asarray(img)
    return img

def record_rollouts(agent, env, config, device):
    agent.qnetwork_local.eval()
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
    agent.qnetwork_local.train()
    return stats
