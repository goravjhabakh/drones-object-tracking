import torch
import os

TARGET_PATH = "app/model/target_embedding.pt"

_os_dir = os.path.dirname(TARGET_PATH)
os.makedirs(_os_dir, exist_ok=True)

_cached_embedding = None


def save_target_embedding(embedding: torch.Tensor):
    global _cached_embedding
    _cached_embedding = embedding.detach().cpu()

    torch.save(_cached_embedding, TARGET_PATH)
    return True


def get_target_embedding():
    global _cached_embedding

    # Return cached version if available
    if _cached_embedding is not None:
        return _cached_embedding.to("cuda")

    # Load from disk if exists
    if os.path.exists(TARGET_PATH):
        emb = torch.load(TARGET_PATH, map_location="cuda")
        _cached_embedding = emb
        return emb

    # No embedding yet
    return None


def clear_target():
    global _cached_embedding
    _cached_embedding = None

    if os.path.exists(TARGET_PATH):
        os.remove(TARGET_PATH)