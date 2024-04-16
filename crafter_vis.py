from PIL import Image, ImageDraw

from craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
from craftax_classic.renderer import render_craftax_pixels
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp
import wandb

from crafter_constants import achievement_labels, action_names

def achievement_counts_heatmap(achievement_counts, skill_size):
    plt.close()

    skill_labels = [f'Skill {i}' for i in range(skill_size)]

    plt.figure(figsize=(4, 7))
    plt.imshow(achievement_counts.T)
    plt.xticks(range(skill_size), skill_labels, rotation=45)
    plt.yticks(range(len(achievement_labels)), achievement_labels)
    plt.title('Achievement Counts by Skill')
    plt.tight_layout()
    plt.colorbar()

    for i in range(len(achievement_labels)):
        for j in range(skill_size):
            plt.text(j, i, round(achievement_counts[j, i], 1), ha="center", va="center", color="w")
    
    return wandb.Image(plt)

test_env = CraftaxClassicSymbolicEnv()
env_params = test_env.default_params

def gather_rollouts(key, qlocal, qlocal_params, discrim, discrim_params, embedding_fn, skill_size,
                    num_rollouts=5, max_steps_per_rollout=500):
    skill_rollouts = []
    for skill_idx in range(skill_size):
        skill = jax.nn.one_hot(skill_idx, skill_size).reshape(1, -1)
        rollouts = []
        for _ in range(num_rollouts):
            key, env_init_key = random.split(key, num=2)
            obs, state = test_env.reset(env_init_key, env_params)

            rollout = []
            for t in range(max_steps_per_rollout):
                key, env_step_key = random.split(key, num=2)
                
                qvalues = qlocal.apply(qlocal_params, obs.reshape(1, -1), skill, train=False)
                action_probs = jax.nn.softmax(qvalues)[0]
                action = jnp.argmax(qvalues, axis=-1).item()

                next_obs, state, reward_gt, done, info = test_env.step(env_step_key, state, action,
                                                                    env_params)
                next_obs_embedding = embedding_fn(next_obs.reshape(1, -1))
                discrim_logits = discrim.apply(discrim_params, next_obs_embedding, train=False)
                discrim_probs = jax.nn.softmax(discrim_logits)[0]

                frame = render_craftax_pixels(state, 64) / 255.0
                img = Image.fromarray(onp.uint8(frame * 255)).convert('RGB')

                action_name = action_names[action]
                action_prob = round(action_probs[action].item(), 4)
                discrim_prob = round(discrim_probs[skill_idx].item(), 4)
                reward_gt = round(reward_gt, 4)

                draw = ImageDraw.Draw(img)
                draw.text((10, 410), f"t={t}", (255, 255, 255),
                          font_size=24, stroke_width=1)
                draw.text((10, 10), f"Action: {action_name}", (255, 255, 255),
                          font_size=24, stroke_width=1)
                draw.text((10, 40), f"Action Prob: {action_prob}", (255, 255, 255),
                          font_size=24, stroke_width=1)
                draw.text((300, 10), f"Skill {skill_idx} Prob: {discrim_prob}", (255, 255, 255),
                          font_size=24, stroke_width=1)
                draw.text((300, 40), f"Reward GT: {reward_gt}", (255, 255, 255),
                          font_size=24, stroke_width=1)

                rollout.append(onp.array(img))

                if done:
                    break

            rollout_tensor = onp.array(rollout).transpose(0, 3, 1, 2)
            rollouts.append(rollout_tensor)
        skill_rollouts.append(rollouts)
    return skill_rollouts
