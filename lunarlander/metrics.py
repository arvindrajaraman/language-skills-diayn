import numpy as np
import torch

from utils import to_numpy

def _classify_goal(obs):
    if -1.0 <= obs[0] < -0.33:
        return 0
    elif -0.33 <= obs[0] < 0.33:
        return 1
    else:
        return 2

def I_zs_confusion_matrix(config, goal_skill_pairs):
    assert config.skill_size == 3 and config.env_name == "LunarLander-v2", "Not the right settings for this metric"

    freq_matrix = np.zeros((3, 3))
    for obs, skill in goal_skill_pairs:
        goal_idx = _classify_goal(obs)
        skill_idx = np.argmax(skill)
        freq_matrix[skill_idx][goal_idx] += 1

    if np.sum(freq_matrix) < 1e-9:
        return np.zeros((3, 3))

    conf_matrix = freq_matrix / (np.sum(freq_matrix, axis=1, keepdims=True) + 1e-9)
    return conf_matrix

def I_zs_score(freq_matrix):
    total_freq = np.sum(freq_matrix)
    if total_freq < 1e-9:
        return 0.0
    
    p_y = np.sum(freq_matrix, axis=0) / total_freq
    p_yhat = np.sum(freq_matrix, axis=1) / total_freq
    p_y_yhat = freq_matrix / total_freq
    
    mi = 0.0
    for i in range(3):
        for j in range(3):
            if p_y_yhat[i, j] > 0 and p_y[i] > 0 and p_yhat[j] > 0:
                mi += p_y_yhat[i, j] * np.log(p_y_yhat[i, j] / (p_y[i] * p_yhat[j] + 1e-9))

    return mi

def I_za_score(config, agent):
    if agent.t_step == 0:
        return 0.0
    
    states, _, skills_gt, _, _, _ = agent.memory.get_all()
    states, skills_gt = to_numpy(states), to_numpy(skills_gt)

    def _H_a():
        all_probs = np.empty((config.skill_size, states.shape[0], config.action_size))
        for i in range(config.skill_size):
            skill = np.zeros((1, config.skill_size))
            skill[0][i] = 1.
            skills = np.repeat(skill, states.shape[0], axis=0)
            all_probs[i] = agent.act_probs(states, skills)
        probs = all_probs.mean(axis=0)
        return -np.multiply(probs, np.log(probs + 1e-9)).sum(axis=1).mean()
    
    def _H_az():
        probs = agent.act_probs(states, skills_gt)
        return -np.multiply(probs, np.log(probs + 1e-9)).sum(axis=1).mean()
    
    H_a = _H_a()
    H_az = _H_az()

    return H_a - H_az

def std_a_score(config, agent):
    if agent.t_step == 0:
        return 0.0
    
    states, _, skills_gt, _, _, _ = agent.memory.get_all()
    states, skills_gt = to_numpy(states), to_numpy(skills_gt)

    all_probs = np.empty((config.skill_size, states.shape[0], config.action_size))
    for i in range(config.skill_size):
        skill = np.zeros((1, config.skill_size))
        skill[0][i] = 1.
        skills = np.repeat(skill, states.shape[0], axis=0)
        all_probs[i] = agent.act_probs(states, skills)

    std = all_probs.std(axis=0).mean()
    return std
