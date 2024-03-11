import gym
import numpy as np

def _compute_mutual_info(freq_matrix):
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
                mi += p_y_yhat[i, j] * np.log(p_y_yhat[i, j] / (p_y[i] * p_yhat[j]))
    
    return mi

def _classify_goal(next_state):
    if -1.5 <= next_state[0] <= -0.25:
        return 0
    elif -0.25 <= next_state[0] <= 0.25:
        return 1
    else:
        return 2

def train_mutual_info_score(config, last_n_goals):
    assert config.skill_size == 3 and config.env_name == "LunarLander-v2", "Not the right settings for this metric"

    freq_matrix = np.zeros((3, 3))
    for next_state, skill in last_n_goals:
        goal_idx = _classify_goal(next_state)
        skill_idx = np.argmax(skill)
        freq_matrix[skill_idx][goal_idx] += 1

    return _compute_mutual_info(freq_matrix)


# def test_mutual_info_score(agent, config, rollouts_per_skill=20):
#     assert config.skill_size == 3 and config.env_name == "LunarLander-v2", "Not the right settings for this metric"

#     freq_matrix = np.zeros((3, 3))
#     env = gym.make(config.env_name)
#     for skill_idx in [0, 1, 2]:
#         for _ in range(rollouts_per_skill):
#             obs = env.reset()
#             for _ in range(config.max_steps_per_episode):
#                 action = agent.act(obs, skill_idx, eps=0.)
#                 next_obs, _, done, _ = env.step(action)
#                 if done:
#                     goal_idx = _classify_goal(next_obs)
#                     freq_matrix[skill_idx][goal_idx] += 1
#                     break

#                 obs = next_obs
            
#             env.close()
    
#     return _compute_mutual_info(freq_matrix)
