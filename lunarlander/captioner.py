import numpy as np

def naive_captioner(obs):
    result = np.zeros((obs.shape[0], 3))
    
    result[(obs[:, 0] >= -1.5) & (obs[:, 0] < -0.33), 0] = 1.
    result[(obs[:, 0] >= -0.33) & (obs[:, 0] < 0.33), 1] = 1.
    result[(obs[:, 0] >= 0.33) & (obs[:, 0] < 1.5), 2] = 1.

    return result
