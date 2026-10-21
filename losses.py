import torch
from torch import Tensor


def retrieval_loss(
    embeddings: Tensor,
    n_queries: int,
    positives: Tensor,
    temperature: float = 0.05,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if positives.shape != (n_queries, len(embeddings) - n_queries) or not positives.any(1).all():
        raise ValueError("Each query needs at least one positive among the candidates")
    queries, documents = embeddings[:n_queries].float(), embeddings[n_queries:].float()
    logits = queries @ documents.T / temperature
    # Sum the probability of every known positive instead of treating another answer as a negative.
    return (logits.logsumexp(-1) - logits.masked_fill(~positives, -torch.inf).logsumexp(-1)).mean()
