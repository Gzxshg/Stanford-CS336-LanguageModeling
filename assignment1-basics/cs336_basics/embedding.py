import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            device:torch.device | None = None,

    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device)
        )

        nn.init.trunc_normal_(self.weight, mean=0, std=0.02, a=-2.0, b=2.0)

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        """
        Args:
            token_ids: A tensor of shape (...,) containing integer token IDs in the range [0, num_embeddings-1]
        Returns:
            A tensor of shape (..., embedding_dim) containing the corresponding embeddings for each token ID.
        """
        return self.weight[token_ids]