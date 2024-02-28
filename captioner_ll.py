import torch
from sentence_transformers import SentenceTransformer
from functools import lru_cache

embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

@lru_cache(maxsize=3)
def naive_captioner(obs):
    if -1.5 <= obs[0] <= -0.5:
        return torch.Tensor([1.0, 0.0, 0.0])
    elif -0.5 <= obs[0] <= 0.5:
        return torch.Tensor([0.0, 1.0, 0.0])
    else:
        return torch.Tensor([0.0, 0.0, 1.0])

@lru_cache(maxsize=128)
def language_captioner(obs):
    if -1.5 <= obs[0] <= -0.5:
        return embedding_model.encode("The rocket is to the left.")
    elif -0.5 <= obs[0] <= 0.5:
        return embedding_model.encode("The rocket is in the middle.")
    else:
        return embedding_model.encode("The rocket is to the right.")
