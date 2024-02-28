import numpy as np
import os

import gym

def _compute_mutual_info(mi_matrix):
    mi = 0.0
    for i in range(3):
        for j in range(3):
            mi += mi_matrix[i][j] * np.log(mi_matrix[i][j] / (mi_matrix[i].sum() * mi_matrix[:, j].sum()))
    return mi

def _classify_goal(next_state):
    if -1.5 <= next_state[0] <= -0.5:
        return 0
    elif -0.5 <= next_state[0] <= 0.5:
        return 1
    else:
        return 2

def train_mutual_info_score(config, last_n_goals):
    assert config["skill_size"] == 3 and config["env_name"] == "LunarLander-v2", "Not the right settings for this metric"

    mi_matrix = np.zeros((3, 3))
    for next_state, skill_idx in last_n_goals:
        goal_idx = _classify_goal(next_state)
        mi_matrix[goal_idx][skill_idx] += 1

    return _compute_mutual_info(mi_matrix)


def test_mutual_info_score(agent, config, rollouts_per_skill=20):
    assert config["skill_size"] == 3 and config["env_name"] == "LunarLander-v2", "Not the right settings for this metric"

    mi_matrix = np.zeros((3, 3))
    env = gym.make(config["env_name"])
    for skill_idx in [0, 1, 2]:
        for _ in range(rollouts_per_skill):
            obs = env.reset()
            for _ in range(config["max_steps_per_episode"]):
                action = agent.act(obs, skill_idx, eps=0.)
                next_obs, _, done, _ = env.step(action)
                if done:
                    goal_idx = _classify_goal(next_obs)
                    mi_matrix[goal_idx][skill_idx] += 1
                    break

                obs = next_obs
            
            env.close()

    return _compute_mutual_info(mi_matrix)