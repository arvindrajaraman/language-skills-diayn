import torch

def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def to_torch(ndarray):
    return torch.from_numpy(ndarray).to('cuda').float()

def to_torch_bool(ndarray):
    return torch.from_numpy(ndarray).to('cuda').bool()

def to_torch_long(ndarray):
    return torch.from_numpy(ndarray).to('cuda').long()
