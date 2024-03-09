import torch

def naive_captioner(obs):
    if -1.5 <= obs[0] <= -0.25:
        return torch.Tensor([1.0, 0.0, 0.0])
    elif -0.25 <= obs[0] <= 0.25:
        return torch.Tensor([0.0, 1.0, 0.0])
    else:
        return torch.Tensor([0.0, 0.0, 1.0])
