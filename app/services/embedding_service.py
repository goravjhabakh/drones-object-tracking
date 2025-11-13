from sentence_transformers import SentenceTransformer
from core.config import CLIP_MODEL_NAME, DEVICE
import torch

clip_model = SentenceTransformer(CLIP_MODEL_NAME, device=DEVICE)

def get_image_embedding(image):
  return clip_model.encode(image, convert_to_tensor=True)

def encode_images(images, batch_size=32):
    """
    Encode list of PIL images into a torch.Tensor (N, D)
    """
    if len(images) == 0:
        return None

    embeddings = clip_model.encode(
        images,
        batch_size=batch_size,
        convert_to_numpy=False,   # return torch tensors
        show_progress_bar=False
    )

    # If embeddings is a list -> convert to torch tensor
    if isinstance(embeddings, list):
        embeddings = torch.stack(embeddings).to(DEVICE)
    else:
        embeddings = embeddings.to(DEVICE)

    # Normalize
    embeddings = torch.nn.functional.normalize(embeddings, dim=1)

    return embeddings